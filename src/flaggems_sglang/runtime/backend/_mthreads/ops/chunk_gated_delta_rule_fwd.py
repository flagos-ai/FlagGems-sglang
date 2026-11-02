# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chunked gated delta-rule forward (vLLM-Ascend chunk.py semantics).

Two-kernel design:

1. `_gdn_prep_kernel` - T-parallel L2 normalization of q and k. The reference
   normalizes q and k in float32 (rsqrt(sum(x*x) + 1e-6)) before multiplying q
   by `scale`. Doing that inside the sequential recurrent kernel forces two
   full cross-thread reductions per timestep; hoisting it into a small
   T-parallel pass removes them from the serial chain entirely. The normalized
   values are kept in float32, so no 16-bit round trip is introduced.

2. `_gdn_fwd_kernel` - fused recurrent scan. One program owns the full
   [BLOCK_K, BLOCK_V] float32 state of one (batch-or-segment, head, V-tile)
   pair and walks the sequence once, applying the exact reference recurrence:
   decay the state by exp(g), predict v from the decayed state, correct it by
   beta, rank-1 accumulate k*v_corr (state is [K, V], so outer(k, x)), then read
   out with q * scale after the update.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_prep_kernel(
    q_ptr,
    k_ptr,
    qo_ptr,
    ko_ptr,
    scale,
    T,
    H,
    Kd,
    s_qb,
    s_qt,
    s_qh,
    s_ob,
    s_ot,
    s_oh,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.program_id(2)

    rows = c * BT + tl.arange(0, BT)
    offs = tl.arange(0, BK)
    rm = rows < T
    km = offs < Kd
    m2 = rm[:, None] & km[None, :]

    src = b * s_qb + h * s_qh + rows[:, None] * s_qt + offs[None, :]
    qv = tl.load(q_ptr + src, mask=m2, other=0.0).to(tl.float32)
    kv = tl.load(k_ptr + src, mask=m2, other=0.0).to(tl.float32)

    qn = tl.rsqrt(tl.sum(qv * qv, axis=1) + 1e-6)
    kn = tl.rsqrt(tl.sum(kv * kv, axis=1) + 1e-6)

    dst = b * s_ob + h * s_oh + rows[:, None] * s_ot + offs[None, :]
    tl.store(qo_ptr + dst, qv * qn[:, None] * scale, mask=m2)
    tl.store(ko_ptr + dst, kv * kn[:, None], mask=m2)


@triton.jit
def _gdn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    b_ptr,
    o_ptr,
    init_ptr,
    fin_ptr,
    cu_ptr,
    scale,
    T,
    H,
    Kd,
    Vd,
    s_qb,
    s_qt,
    s_qh,
    s_vb,
    s_vt,
    s_vh,
    s_gb,
    s_gt,
    s_gh,
    s_ob,
    s_ot,
    s_oh,
    s_n,
    s_h,
    s_k,
    NV,
    HAS_INIT: tl.constexpr,
    HAS_CU: tl.constexpr,
    PRE_NORM: tl.constexpr,
    OUT_FINAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    nv = pid % NV
    bh = pid // NV
    if HAS_CU:
        seg = bh // H
        h = bh % H
        b = 0
    else:
        b = bh // H
        h = bh % H
        seg = b

    offs_k = tl.arange(0, BLOCK_K)
    offs_v = nv * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = offs_k < Kd
    v_mask = offs_v < Vd
    km2 = k_mask[:, None] & v_mask[None, :]

    if HAS_INIT:
        S = tl.load(
            init_ptr
            + seg * s_n
            + h * s_h
            + offs_k[:, None] * s_k
            + offs_v[None, :],
            mask=km2,
            other=0.0,
        ).to(tl.float32)
    else:
        S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)

    if HAS_CU:
        t0 = tl.load(cu_ptr + seg)
        t1 = tl.load(cu_ptr + seg + 1)
    else:
        t0 = 0
        t1 = T

    qbase = q_ptr + b * s_qb + h * s_qh
    kbase = k_ptr + b * s_qb + h * s_qh
    vbase = v_ptr + b * s_vb + h * s_vh
    gbase = g_ptr + b * s_gb + h * s_gh
    bbase = b_ptr + b * s_gb + h * s_gh
    obase = o_ptr + b * s_ob + h * s_oh

    for t in range(t0, t1):
        kt = tl.load(kbase + t * s_qt + offs_k, mask=k_mask, other=0.0).to(
            tl.float32
        )
        qt = tl.load(qbase + t * s_qt + offs_k, mask=k_mask, other=0.0).to(
            tl.float32
        )
        vt = tl.load(vbase + t * s_vt + offs_v, mask=v_mask, other=0.0).to(
            tl.float32
        )
        gt = tl.load(gbase + t * s_gt).to(tl.float32)
        bt = tl.load(bbase + t * s_gt).to(tl.float32)

        if not PRE_NORM:
            qt = qt * scale

        S = S * tl.exp(gt)
        x = (vt - tl.sum(S * kt[:, None], axis=0)) * bt
        S = S + kt[:, None] * x[None, :]
        ot = tl.sum(S * qt[:, None], axis=0)
        tl.store(
            obase + t * s_ot + offs_v,
            ot.to(o_ptr.dtype.element_ty),
            mask=v_mask,
        )

    if OUT_FINAL:
        tl.store(
            fin_ptr
            + seg * s_n
            + h * s_h
            + offs_k[:, None] * s_k
            + offs_v[None, :],
            S,
            mask=km2,
        )


def chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
):
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise AssertionError("q, k and v must share a dtype")
    if q.dtype == torch.float32:
        raise AssertionError(
            "chunk_gated_delta_rule does not support float32; use bfloat16"
        )
    if beta.dim() != 3:
        raise AssertionError("beta must be of shape [B, T, H]")

    if scale is None:
        scale = k.shape[-1] ** -0.5

    B, T, H, Kd = q.shape
    Vd = v.shape[-1]

    has_cu = cu_seqlens is not None
    if has_cu:
        if B != 1:
            raise ValueError(
                "batch size is expected to be 1 when using cu_seqlens; "
                f"got {B}"
            )
        n_seg = cu_seqlens.shape[0] - 1
        if initial_state is not None and initial_state.shape[0] != n_seg:
            raise ValueError(
                "the number of initial states must equal the number of "
                f"sequences, i.e. {n_seg} rather than {initial_state.shape[0]}"
            )
    else:
        n_seg = B

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()
    if not g.is_contiguous():
        g = g.contiguous()
    if not beta.is_contiguous():
        beta = beta.contiguous()
    if initial_state is not None and not initial_state.is_contiguous():
        initial_state = initial_state.contiguous()
    if has_cu and not cu_seqlens.is_contiguous():
        cu_seqlens = cu_seqlens.contiguous()

    o = torch.empty((B, T, H, Vd), dtype=q.dtype, device=q.device)
    final_state = (
        torch.empty((n_seg, H, Kd, Vd), dtype=torch.float32, device=q.device)
        if output_final_state
        else None
    )
    # Every pointer slot the kernel might not use still receives a real,
    # consistently-shaped fp32/int64 buffer of the natural dtype and shape.
    # Passing an unused 16-bit-typed pointer slot (or aliasing a dead slot onto
    # o) was measured to corrupt the o output on this backend.
    scratch = torch.empty(
        (n_seg, H, Kd, Vd), dtype=torch.float32, device=q.device
    )
    fin_ptr = final_state if final_state is not None else scratch
    init_ptr = initial_state if initial_state is not None else scratch
    cu_ptr = (
        cu_seqlens
        if has_cu
        else torch.zeros(1, dtype=torch.int64, device=q.device)
    )

    sk = max(16, triton.next_power_of_2(Kd))
    # Block-size / warp dispatch measured on the target (scan-only timings of the
    # eight timing shapes, 60-SM MTT S5000):
    #   K=128,V=128, segment length >= 2048 : BLOCK_V=2,  num_warps=8
    #                                          (3.8 ms vs 4.0 ms at BLOCK_V=4/w16)
    #   K=128,V=128, segment length >= 1024 : BLOCK_V=8,  num_warps=8
    #                                          (2.96 ms vs 3.05 ms at BLOCK_V=4/w4)
    #   K=128,V=128, shorter segments       : BLOCK_V=16, num_warps=4
    #   otherwise                           : BLOCK_V=4,  num_warps=4
    #                                          (1.9x on K=V=64, T=256)
    # Narrow V-tiles multiply the number of independent programs, which the
    # latency-bound serial scan needs; very long segments additionally need more
    # warps per program to overlap the carried-state reduction chain. The
    # threshold must use the per-segment length, not the packed T: with
    # cu_seqlens, T is the sum over segments and a large T can still mean short
    # segments, where the narrow-tile geometry measured ~2x slower.
    nv_full = triton.next_power_of_2(Vd)
    seg_len = T if n_seg == 1 else T // n_seg
    if Kd >= 128 and Vd >= 128:
        if seg_len >= 2048:
            bv, num_warps = 2, 8
        elif seg_len >= 1024:
            bv, num_warps = 8, 8
        else:
            bv, num_warps = 16, 4
    else:
        bv, num_warps = 4, 4
    bv = max(1, min(bv, nv_full))
    nv = (Vd + bv - 1) // bv

    if use_qk_l2norm_in_kernel:
        qn = torch.empty((B, T, H, Kd), dtype=torch.float32, device=q.device)
        kn = torch.empty((B, T, H, Kd), dtype=torch.float32, device=q.device)
        bt_rows = 64
        _gdn_prep_kernel[(B, H, triton.cdiv(T, bt_rows))](
            q,
            k,
            qn,
            kn,
            float(scale),
            T,
            H,
            Kd,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            qn.stride(0),
            qn.stride(1),
            qn.stride(2),
            BT=bt_rows,
            BK=sk,
            num_warps=4,
        )
        q_ptr, k_ptr = qn, kn
    else:
        q_ptr, k_ptr = q, k

    grid = (n_seg * H * nv,)
    _gdn_fwd_kernel[grid](
        q_ptr,
        k_ptr,
        v,
        g,
        beta,
        o,
        init_ptr,
        fin_ptr,
        cu_ptr,
        float(scale),
        T,
        H,
        Kd,
        Vd,
        q_ptr.stride(0),
        q_ptr.stride(1),
        q_ptr.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        fin_ptr.stride(0),
        fin_ptr.stride(1),
        fin_ptr.stride(2),
        nv,
        HAS_INIT=initial_state is not None,
        HAS_CU=has_cu,
        PRE_NORM=bool(use_qk_l2norm_in_kernel),
        OUT_FINAL=bool(output_final_state),
        BLOCK_K=sk,
        BLOCK_V=bv,
        num_warps=num_warps,
    )
    return o, final_state


__all__ = ["chunk_gated_delta_rule_fwd"]
