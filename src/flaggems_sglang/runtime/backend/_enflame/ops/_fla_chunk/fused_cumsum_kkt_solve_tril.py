# Copyright (c) 2025 FlagGems. All rights reserved.
# Fused cumsum + KKT + solve_tril for chunk_gated_delta_rule. Returns g_out, A_inv; w_u is separate.
# License: Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flaggems_sglang.runtime.backend._enflame.ops._fla_chunk.index import (
    prepare_chunk_indices,
)
from flaggems_sglang.utils import libentry, libtuner


@libentry()
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_G": lambda args: args.get("USE_G", True),
    }
)
@libtuner(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [64, 128]
        for num_warps in [1, 2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel(
    g_in,
    g_out,
    k,
    beta,
    A_inv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_G: tl.constexpr,
    USE_TMA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # ---------- cumsum ----------
    p_g_in = tl.make_block_ptr(
        g_in + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    p_g_out = tl.make_block_ptr(
        g_out + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_g = tl.load(p_g_in, boundary_check=(0,)).to(tl.float32)
    b_g = tl.cumsum(b_g, axis=0)
    tl.store(p_g_out, b_g.to(p_g_out.dtype.element_ty), boundary_check=(0,))

    # ---------- KKT (compute L in registers) ----------
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        # the second operand needs [BK, BT]: load the transposed tile through
        # a K-major block pointer (same form as the h/o kernels) instead of
        # materializing a tl.trans layout conversion on the dot operand
        p_kt = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (K, T),
            (1, Hg * K),
            (i_k * BK, i_t * BT),
            (BK, BT),
            (0, 1),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kt = tl.load(p_kt, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), b_kt)
    if USE_G:
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * tl.exp(b_g_diff)
    m_A_kkt = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A_kkt, b_A, 0)

    # ---------- solve_tril (registers -> A_inv) ----------
    # GCU: two rewrites of the upstream solve. (1) The 16x16 block-merge
    # formulation chains [16,16] fp32 ieee tl.dot ops that fail gcu.matmul
    # attribute conversion on every fresh compile (varlen specialization,
    # cleared caches, the standalone solve_tril entry). (2) The plain
    # row-recurrence forward substitution compiles (and was verified against
    # solve_triangular on GCU, maxdiff <= 7.3e-4) but is a 62-step serial
    # chain per program -- ~90% of this kernel's wall time. Instead use the
    # nilpotent factorization: for strictly-lower A (A^BT == 0),
    #   (I + A)^-1 = (I-A)(I+A^2)(I-A^4)(I+A^8)(I-A^16)(I+A^32),
    # i.e. log2(BT)-1 squarings plus as many factor products of [BT, BT]
    # ieee dots. Rows/cols past the sequence end load as zero, so partial
    # chunks yield the exact live-block inverse padded with identity.
    # The masked KKT result feeds the inverse directly from registers: the
    # historical global round-trip of A through DRAM was write-only dead
    # storage (only A_inv leaves this kernel), so it is gone.
    o_i = tl.arange(0, BT)
    m_I = o_i[:, None] == o_i[None, :]
    b_eye = tl.where(m_I, 1.0, 0.0)
    b_L = b_A
    b_Y = b_eye - b_L
    b_P = tl.dot(b_L, b_L, input_precision="ieee")
    b_Y = tl.dot(b_Y, b_eye + b_P, input_precision="ieee")
    if BT >= 32:
        b_P = tl.dot(b_P, b_P, input_precision="ieee")
        b_Y = tl.dot(b_Y, b_eye - b_P, input_precision="ieee")
    if BT >= 64:
        b_P = tl.dot(b_P, b_P, input_precision="ieee")
        b_Y = tl.dot(b_Y, b_eye + b_P, input_precision="ieee")
    if BT >= 128:
        b_P = tl.dot(b_P, b_P, input_precision="ieee")
        b_Y = tl.dot(b_Y, b_eye - b_P, input_precision="ieee")
    b_Ai = b_Y
    A_inv_base = A_inv + (bos * H + i_h) * BT
    p_Ai = tl.make_block_ptr(
        A_inv_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(
        p_Ai,
        b_Ai.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


def chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril(
    g: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype | None = None,
    *,
    use_g_in_kkt: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused kernel: cumsum(g) + KKT(L) + solve_tril(L -> inv). Returns (g_out, A_inv).
    w_u stays a separate kernel (e.g. recompute_w_u_fwd) for HGMMA."""
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    output_dtype = output_dtype or k.dtype
    # GCU: Triton reads int64 buffers with garbage values; normalize to int32
    # (sequence offsets are < 2^31 here).
    if cu_seqlens is not None and cu_seqlens.dtype == torch.int64:
        cu_seqlens = cu_seqlens.to(torch.int32)
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT)
        if cu_seqlens is not None
        else None
    )
    if chunk_indices is not None and chunk_indices.dtype == torch.int64:
        chunk_indices = chunk_indices.to(torch.int32)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    g_out = torch.empty_like(g)
    # Every (i_t, i_bh) program stores its full [BT, BT] tile clipped to T, so
    # all rows [0, T) of A_inv are written exactly once: no zero fill needed.
    A_inv = torch.empty(B, T, H, BT, device=g.device, dtype=output_dtype)

    def grid(meta):
        # Varlen kernels derive the sequence from chunk_indices. The batch id
        # from i_bh is unused there, so H programs are sufficient for B > 1 too.
        return (NT, H if cu_seqlens is not None else B * H)

    chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel[grid](
        g_in=g,
        g_out=g_out,
        k=k,
        beta=beta,
        A_inv=A_inv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
        IS_VARLEN=cu_seqlens is not None,
        USE_G=use_g_in_kkt,
        USE_TMA=False,
        DOT_PRECISION="ieee",
    )
    return g_out, A_inv
