# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
# Portions adapted from SGLang (Copyright 2023-2026 SGLang Team, Apache-2.0).
"""SGLang v0.5.18 compressed-candidate scoring entry point."""

import torch
import triton
import triton.language as tl
from flaggems_sglang_plugin.compat.dsv41.fp4_prefill import dense_fp8_index_logits
from flaggems_sglang.runtime.backend._nvidia.hopper.ops.dsv41_fp8_prefill_candidates import (
    candidate_prefill_score,
    supports_candidate_prefill,
)

INDEX_HEAD_DIM = 128


@triton.jit
def _fp8_index_logits_prefill_kernel(
    q_ptr,  # [B, H, D] e4m3
    w_ptr,  # [B, H] bf16
    k_ptr,  # [L, D] e4m3
    lens_ptr,  # [B] int64
    out_ptr,  # [B, OUT_L] fp32
    candidate_blocks_ptr,
    stride_candidates,
    L,
    OUT_L,
    stride_qb,
    stride_qh,
    stride_kl,
    stride_wb,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    out_cols = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    if USE_CANDIDATES:
        blocks = tl.load(
            candidate_blocks_ptr
            + b * stride_candidates
            + out_cols // CANDIDATE_BLOCK_SIZE,
            mask=out_cols < OUT_L,
            other=-1,
        ).to(tl.int64)
        offs_l = blocks * CANDIDATE_BLOCK_SIZE + out_cols % CANDIDATE_BLOCK_SIZE
    else:
        offs_l = out_cols
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    n_vis = tl.load(lens_ptr + b)
    valid = (offs_l >= 0) & (offs_l < tl.minimum(n_vis, L)) & (out_cols < OUT_L)
    q = tl.load(q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :])
    k = tl.load(
        k_ptr + offs_l[:, None] * stride_kl + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    acc = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
    # Preserve the reference post-dot rounding and reduction points.
    s = acc.to(tl.bfloat16).to(tl.float32)
    s = tl.maximum(s, 0.0)
    w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
    s = (s * w[:, None]).to(tl.bfloat16).to(tl.float32)
    logit = tl.sum(s, axis=0).to(tl.bfloat16).to(tl.float32)
    logit = tl.where(valid, logit, float("-inf"))
    tl.store(out_ptr + b * OUT_L + out_cols, logit, mask=out_cols < OUT_L)


def fp8_index_logits_prefill(
    q: torch.Tensor,
    weights: torch.Tensor,
    keys: torch.Tensor,
    lens: torch.Tensor,
    *,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 8,
) -> torch.Tensor:
    """Score E4M3 queries against E4M3 keys with FP32 accumulation.

    Output rows are padded to four floats for the fused ragged top-k kernel;
    padding remains unreachable because it is initialized to ``-inf``.
    """
    if candidate_blocks is None:
        return dense_fp8_index_logits(q, weights, keys, lens)
    if candidate_block_size == 8 and supports_candidate_prefill(
        q, weights, keys, lens, candidate_blocks
    ):
        return candidate_prefill_score(q, weights, keys, lens, candidate_blocks)
    assert q.dtype == torch.float8_e4m3fn and q.shape[-1] == INDEX_HEAD_DIM
    rows, heads, _ = q.shape
    width = keys.shape[0]
    assert keys.dtype == torch.float8_e4m3fn and keys.shape[1:] == (INDEX_HEAD_DIM,)
    assert weights.shape == (rows, heads)
    assert lens.shape == (rows,)
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    keys = keys.contiguous()
    use_candidates = candidate_blocks is not None
    if use_candidates:
        assert candidate_blocks.ndim == 2 and candidate_blocks.shape[0] == rows
        assert candidate_blocks.dtype in (torch.int32, torch.int64)
        assert candidate_block_size % 4 == 0
        out_width = candidate_blocks.shape[1] * candidate_block_size
    else:
        out_width = ((width + 3) // 4) * 4
    out = torch.empty((rows, out_width), dtype=torch.float32, device=q.device)
    if rows == 0 or out_width == 0:
        return out
    # On H100, the 32-head source path is limited by repeated cached loads
    # and small CTAs, rather than HBM traffic. Widen only the key dimension:
    # grouping query heads would change the tensor-core accumulation path.
    # Candidate gathers need a smaller tile to retain enough resident warps.
    block_l = 64
    if heads == 32 and out_width >= 4096 and rows * out_width >= 262144:
        block_l = 128 if use_candidates else 256
    _fp8_index_logits_prefill_kernel[(rows, triton.cdiv(out_width, block_l))](
        q,
        weights,
        keys,
        lens.to(torch.int64).contiguous(),
        out,
        candidate_blocks if use_candidates else lens,
        candidate_blocks.stride(0) if use_candidates else 0,
        width,
        out_width,
        q.stride(0),
        q.stride(1),
        keys.stride(0),
        weights.stride(0),
        H=heads,
        D=INDEX_HEAD_DIM,
        BLOCK_L=block_l,
        USE_CANDIDATES=use_candidates,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        num_warps=4,
    )
    return out
