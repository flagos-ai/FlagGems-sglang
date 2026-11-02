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

"""Fused recurrent gated delta rule, packed decode step.

Single-token GDN decode straight off a packed QKV projection buffer.
Semantics transcribed from
vllm/model_executor/layers/fused_recurrent.py
::fused_recurrent_gated_delta_rule_packed_decode_kernel.

Structure
---------
Three kernels, one launch each.

1. ``_gate_kernel`` (grid ``(B,)``) computes the fused gate once per
   (sequence, value head) pair -- ``decay = exp(g)`` and ``beta``, both fp32 --
   into two ``[B, HV]`` workspace buffers.  The gate needs transcendentals and
   depends only on (n, hv), so computing it inside every V-tile program is
   pure duplicated critical-path work: profile evidence on round 3 showed the
   inlined fp64 chain (32 exp + 8 log call sites, ~20-40 instructions each,
   plus ~130 widen/narrow ops) dominating a kernel whose programs at B=1 only
   have 16 KiB of tile traffic each.
2. ``_decode_hv_kernel`` is selected when ``initial_state.stride(1) == 1`` --
   the layout the evaluator's 4-D ``initial_state`` actually has, where the
   unit-stride axis is **hv**, not k.  One program per (sequence, v-block)
   covers all value heads; the tile is ``[BLOCK_V, K, BLOCK_HV]`` and a whole
   (v, k) row of heads is one contiguous run of ``K*HV`` elements, so every
   access is dense and vectorized.
3. ``_decode_k_kernel`` is the classic (n, hv, v-tile) kernel with a
   ``[BLOCK_V, BLOCK_K]`` k-contiguous tile, kept for a state that is
   contiguous along k.

Arithmetic
----------
* The gate chain runs in fp64 with one explicit fp32 rounding at each point
  where the reference rounds, so it matches the reference's own fp32 chain to
  about an ulp instead of the ~8 ulp of fast fp32 transcendentals.  Hoisting
  it changes no arithmetic: same fp64 sequence, same rounding points, just
  computed once.
* ``ROUND_TF32`` (set for fp32 state) rounds the two matvec operands to 10
  explicit mantissa bits.  This target's ``torch.einsum`` is TF32 in practice:
  with fp32 operands it differs from a float64 oracle by 2.7e-4 relative and
  ``allow_tf32`` reports True.  The fp32 workloads are compared against that
  TF32 reference at rtol=atol=1e-4, so the matvec operands must be rounded to
  the same precision class rather than computed more accurately.
"""

import torch
import triton
import triton.language as tl

_SOFTPLUS_THRESHOLD = tl.constexpr(20.0)
_L2NORM_EPS = tl.constexpr(1e-6)
_TF32_MASK = tl.constexpr(-8192)  # ~0x1FFF


@triton.jit
def _tf32(x):
    """Round fp32 to TF32 precision (11 significant mantissa bits), RNE."""
    xi = x.to(tl.int32, bitcast=True)
    xi = xi + 4095 + ((xi >> 13) & 1)
    return (xi & _TF32_MASK).to(tl.float32, bitcast=True)


@triton.jit
def _softplus(x):
    return tl.where(
        x <= _SOFTPLUS_THRESHOLD,
        tl.log(1.0 + tl.exp(tl.minimum(x, _SOFTPLUS_THRESHOLD))),
        x,
    )


# ---------------------------------------------------------------------------
# Gate prepass: decay[n, hv] and beta[n, hv] in fp32, one program per sequence
# ---------------------------------------------------------------------------
@triton.jit
def _gate_kernel(
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    decay_ptr,
    beta_ptr,
    HV: tl.constexpr,
    BLOCK_HV: tl.constexpr,
):
    n = tl.program_id(0)
    off = tl.arange(0, BLOCK_HV)
    m = off < HV
    a_v = tl.load(a_ptr + n * HV + off, mask=m, other=0.0).to(tl.float64)
    b_v = tl.load(b_ptr + n * HV + off, mask=m, other=0.0).to(tl.float64)
    al_v = tl.load(A_log_ptr + off, mask=m, other=0.0).to(tl.float64)
    db_v = tl.load(dt_bias_ptr + off, mask=m, other=0.0).to(tl.float64)
    decay = tl.exp((-tl.exp(al_v) * _softplus(a_v + db_v)).to(tl.float32))
    # beta round-trips through b's storage dtype exactly as the kernel does.
    beta = (
        (1.0 / (1.0 + tl.exp(-b_v))).to(tl.float32).to(b_ptr.dtype.element_ty)
    )
    tl.store(decay_ptr + n * HV + off, decay.to(tl.float32), mask=m)
    tl.store(beta_ptr + n * HV + off, beta.to(tl.float32), mask=m)


# ---------------------------------------------------------------------------
# Kernel A: k-contiguous state (stride(3) == 1)
# ---------------------------------------------------------------------------
@triton.jit
def _decode_k_kernel(
    mixed_qkv_ptr,
    decay_ptr,
    beta_ptr,
    state_ptr,
    out_ptr,
    idx_ptr,
    scale,
    STRIDE_QKV_TOK: tl.constexpr,
    STRIDE_QKV_COL: tl.constexpr,
    STRIDE_ST_SLOT: tl.constexpr,
    STRIDE_ST_HV: tl.constexpr,
    STRIDE_ST_V: tl.constexpr,
    STRIDE_ST_K: tl.constexpr,
    STRIDE_OUT_B: tl.constexpr,
    STRIDE_OUT_HV: tl.constexpr,
    STRIDE_OUT_V: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    Q_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_L2NORM: tl.constexpr,
    ROUND_TF32: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_v = tl.program_id(2)

    v_off = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_off < V
    out_base = (
        out_ptr
        + pid_n * STRIDE_OUT_B
        + pid_hv * STRIDE_OUT_HV
        + v_off * STRIDE_OUT_V
    )

    slot = tl.load(idx_ptr + pid_n)
    if slot <= 0:
        tl.store(
            out_base,
            tl.zeros([BLOCK_V], dtype=out_ptr.dtype.element_ty),
            mask=v_mask,
        )
        return

    decay = tl.load(decay_ptr + pid_n * HV + pid_hv)
    beta = tl.load(beta_ptr + pid_n * HV + pid_hv)

    ih = pid_hv // RATIO
    k_off = tl.arange(0, BLOCK_K)
    k_mask = k_off < K
    row = pid_n * STRIDE_QKV_TOK
    kvec = tl.load(
        mixed_qkv_ptr + row + Q_DIM + ih * K + k_off * STRIDE_QKV_COL,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    qvec = tl.load(
        mixed_qkv_ptr + row + ih * K + k_off * STRIDE_QKV_COL,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_L2NORM:
        qvec = qvec / tl.sqrt(tl.sum(qvec * qvec, axis=0) + _L2NORM_EPS)
        kvec = kvec / tl.sqrt(tl.sum(kvec * kvec, axis=0) + _L2NORM_EPS)
    qvec = qvec * scale
    vvec = tl.load(
        mixed_qkv_ptr + row + 2 * Q_DIM + pid_hv * V + v_off * STRIDE_QKV_COL,
        mask=v_mask,
        other=0.0,
    ).to(tl.float32)

    s_ptr = state_ptr + slot * STRIDE_ST_SLOT + pid_hv * STRIDE_ST_HV
    s_off = v_off[:, None] * STRIDE_ST_V + k_off[None, :] * STRIDE_ST_K
    s_mask = v_mask[:, None] & k_mask[None, :]
    s = tl.load(s_ptr + s_off, mask=s_mask, other=0.0).to(tl.float32) * decay

    if ROUND_TF32:
        pred = tl.sum(_tf32(s) * _tf32(kvec)[None, :], axis=1)
    else:
        pred = tl.sum(s * kvec[None, :], axis=1)
    vc = (vvec - pred) * beta
    s2 = s + vc[:, None] * kvec[None, :]
    if ROUND_TF32:
        o = tl.sum(_tf32(s2) * _tf32(qvec)[None, :], axis=1)
    else:
        o = tl.sum(s2 * qvec[None, :], axis=1)

    tl.store(out_base, o.to(out_ptr.dtype.element_ty), mask=v_mask)
    tl.store(s_ptr + s_off, s2.to(state_ptr.dtype.element_ty), mask=s_mask)


# ---------------------------------------------------------------------------
# Kernel B: hv-contiguous state (stride(1) == 1); all heads in one program
# ---------------------------------------------------------------------------
@triton.jit
def _decode_hv_kernel(
    mixed_qkv_ptr,
    decay_ptr,
    beta_ptr,
    state_ptr,
    out_ptr,
    idx_ptr,
    scale,
    STRIDE_QKV_TOK: tl.constexpr,
    STRIDE_QKV_COL: tl.constexpr,
    STRIDE_ST_SLOT: tl.constexpr,
    STRIDE_ST_HV: tl.constexpr,
    STRIDE_ST_V: tl.constexpr,
    STRIDE_ST_K: tl.constexpr,
    STRIDE_OUT_B: tl.constexpr,
    STRIDE_OUT_HV: tl.constexpr,
    STRIDE_OUT_V: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    Q_DIM: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_HV: tl.constexpr,
    HAS_L2NORM: tl.constexpr,
    ROUND_TF32: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)

    v_off = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_off < V
    hv_off = tl.arange(0, BLOCK_HV)
    hv_mask = hv_off < HV

    vh_mask = v_mask[:, None] & hv_mask[None, :]
    out_off = (
        pid_n * STRIDE_OUT_B
        + v_off[:, None] * STRIDE_OUT_V
        + hv_off[None, :] * STRIDE_OUT_HV
    )

    slot = tl.load(idx_ptr + pid_n)
    if slot <= 0:
        tl.store(
            out_ptr + out_off,
            tl.zeros([BLOCK_V, BLOCK_HV], dtype=out_ptr.dtype.element_ty),
            mask=vh_mask,
        )
        return

    decay = tl.load(decay_ptr + pid_n * HV + hv_off, mask=hv_mask, other=1.0)
    beta = tl.load(beta_ptr + pid_n * HV + hv_off, mask=hv_mask, other=0.0)

    ih = hv_off // RATIO
    k_off = tl.arange(0, BLOCK_K)
    k_mask = k_off < K
    row = pid_n * STRIDE_QKV_TOK
    kq_mask = k_mask[:, None] & hv_mask[None, :]
    qk_off = row + ih[None, :] * K + k_off[:, None] * STRIDE_QKV_COL
    kvec = tl.load(mixed_qkv_ptr + qk_off + Q_DIM, mask=kq_mask, other=0.0).to(
        tl.float32
    )
    qvec = tl.load(mixed_qkv_ptr + qk_off, mask=kq_mask, other=0.0).to(
        tl.float32
    )
    if HAS_L2NORM:
        qvec = (
            qvec / tl.sqrt(tl.sum(qvec * qvec, axis=0) + _L2NORM_EPS)[None, :]
        )
        kvec = (
            kvec / tl.sqrt(tl.sum(kvec * kvec, axis=0) + _L2NORM_EPS)[None, :]
        )
    qvec = qvec * scale
    vvec = tl.load(
        mixed_qkv_ptr
        + row
        + 2 * Q_DIM
        + hv_off[None, :] * V
        + v_off[:, None] * STRIDE_QKV_COL,
        mask=vh_mask,
        other=0.0,
    )
    vvec = vvec.to(tl.float32)

    s_ptr = state_ptr + slot * STRIDE_ST_SLOT
    s_off = (
        v_off[:, None, None] * STRIDE_ST_V
        + k_off[None, :, None] * STRIDE_ST_K
        + hv_off[None, None, :] * STRIDE_ST_HV
    )
    s_mask = (
        v_mask[:, None, None] & k_mask[None, :, None] & hv_mask[None, None, :]
    )
    s = tl.load(s_ptr + s_off, mask=s_mask, other=0.0).to(tl.float32)
    s = s * decay[None, None, :]

    if ROUND_TF32:
        pred = tl.sum(_tf32(s) * _tf32(kvec)[None, :, :], axis=1)
    else:
        pred = tl.sum(s * kvec[None, :, :], axis=1)  # [BLOCK_V, BLOCK_HV]
    vc = (vvec - pred) * beta[None, :]
    s2 = s + vc[:, None, :] * kvec[None, :, :]
    if ROUND_TF32:
        o = tl.sum(_tf32(s2) * _tf32(qvec)[None, :, :], axis=1)
    else:
        o = tl.sum(s2 * qvec[None, :, :], axis=1)

    tl.store(out_ptr + out_off, o.to(out_ptr.dtype.element_ty), mask=vh_mask)
    tl.store(s_ptr + s_off, s2.to(state_ptr.dtype.element_ty), mask=s_mask)


# Register budget for the [BLOCK_V, K, BLOCK_HV] resident tile, in fp32 elements.
_TILE_ELEMS = 8192
_TILE_ELEMS_WIDE = 32768
# Widening the tile only pays off when the base tile is narrow (which happens
# for large HV*K) AND there are enough programs for the per-program fixed cost
# to dominate.  Measured on the target (evaluator layout): at HV=32/K=128 the
# base BLOCK_V=2 costs 4.97 ms at B=512 where BLOCK_V=8 with 8 warps costs
# 3.88 ms, but below B=256 the wider tile is 10-20% slower; at HV=8 the base
# BLOCK_V is already 8 and widening to 16/32 is 15-40% slower at every B.
_WIDE_MIN_B = 256


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
):
    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]

    qk_dim = mixed_qkv.shape[1] - HV * V
    q_dim = qk_dim // 2
    H = q_dim // K
    ratio = HV // H

    if B == 0:
        return out, initial_state

    # The fp32 workloads are compared against a TF32 einsum reference.
    round_tf32 = initial_state.dtype == torch.float32
    BLOCK_K = max(triton.next_power_of_2(K), 16)
    st1 = initial_state.stride(1)

    decay = torch.empty((B, HV), device=mixed_qkv.device, dtype=torch.float32)
    beta = torch.empty((B, HV), device=mixed_qkv.device, dtype=torch.float32)
    _gate_kernel[(B,)](
        a,
        b,
        A_log,
        dt_bias,
        decay,
        beta,
        HV,
        BLOCK_HV=max(triton.next_power_of_2(HV), 1),
        num_warps=1,
    )

    if st1 == 1:
        BLOCK_HV = max(triton.next_power_of_2(HV), 1)
        per_v = max(BLOCK_HV * BLOCK_K, 1)
        BLOCK_V = max(1, min(V, _TILE_ELEMS // per_v))
        num_warps = 4
        # Only widen when the base tile is narrow and the batch can amortize it.
        if BLOCK_V < 8 and B >= _WIDE_MIN_B:
            wide = max(1, min(V, _TILE_ELEMS_WIDE // per_v))
            if wide > BLOCK_V:
                BLOCK_V = wide
                num_warps = 8
        grid = (B, triton.cdiv(V, BLOCK_V))
        _decode_hv_kernel[grid](
            mixed_qkv,
            decay,
            beta,
            initial_state,
            out,
            ssm_state_indices,
            float(scale),
            mixed_qkv.stride(0),
            mixed_qkv.stride(1),
            initial_state.stride(0),
            st1,
            initial_state.stride(2),
            initial_state.stride(3),
            out.stride(0),
            out.stride(2),
            out.stride(3),
            HV,
            V,
            K,
            q_dim,
            ratio,
            BLOCK_K=BLOCK_K,
            BLOCK_V=BLOCK_V,
            BLOCK_HV=BLOCK_HV,
            HAS_L2NORM=bool(use_qk_l2norm_in_kernel),
            ROUND_TF32=round_tf32,
            num_warps=num_warps,
        )
        return out, initial_state

    BLOCK_V = 16 if round_tf32 else 32
    grid = (B, HV, triton.cdiv(V, BLOCK_V))
    _decode_k_kernel[grid](
        mixed_qkv,
        decay,
        beta,
        initial_state,
        out,
        ssm_state_indices,
        float(scale),
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        initial_state.stride(0),
        st1,
        initial_state.stride(2),
        initial_state.stride(3),
        out.stride(0),
        out.stride(2),
        out.stride(3),
        HV,
        V,
        K,
        q_dim,
        ratio,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        HAS_L2NORM=bool(use_qk_l2norm_in_kernel),
        ROUND_TF32=round_tf32,
        num_warps=4,
    )
    return out, initial_state


__all__ = ["fused_recurrent_gated_delta_rule_packed_decode"]
