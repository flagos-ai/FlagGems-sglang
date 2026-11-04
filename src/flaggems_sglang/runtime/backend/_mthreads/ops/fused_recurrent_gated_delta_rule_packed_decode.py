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

"""Fused recurrent gated-delta-rule packed decode (single-token GDN decode step).

Triton implementation of
vllm/model_executor/layers/fla/ops/fused_recurrent.py
::fused_recurrent_gated_delta_rule_packed_decode_kernel

One launch, grid = B * HV * NV; each program owns (sequence n, value head ihv,
V-block iv) and the full K axis of the paged recurrent state tile.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _tf32_trunc(x):
    """Reproduce the target's fp32 matmul operand rounding.

    On this target torch's fp32 ``einsum``/``matmul`` lowers to a tensor-core
    path that truncates each operand to tf32 (round-toward-zero on the 13 low
    bits of the fp32 pattern) while accumulating in fp32.  A target debug job
    confirmed that this emulation reproduces the device result to ~1e-7 of a
    3.4-magnitude reduction, whereas an exact fp32 sum of the unrounded
    operands differs by ~3e-3, which the fp32 contract (rtol = atol = 1e-4)
    rejects.  The truncation is only ever enabled for fp32 operands: fp16
    (10 explicit mantissa bits) and bf16 (8) survive it unchanged.
    """
    return (x.to(tl.int32, bitcast=True) & -8192).to(tl.float32, bitcast=True)


@triton.jit
def _gdn_packed_decode_kernel(
    mq_ptr,
    a_ptr,
    b_ptr,
    al_ptr,
    dt_ptr,
    st_ptr,
    o_ptr,
    idx_ptr,
    scale,
    mq_s0,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    al_s0,
    dt_s0,
    st_s0,
    st_s1,
    st_s2,
    st_s3,
    o_s0,
    o_s2,
    o_s3,
    H,
    HV,
    V,
    K,
    RATIO,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    L2NORM: tl.constexpr,
    EVEN_V: tl.constexpr,
    EVEN_K: tl.constexpr,
    TF32_EMULATE: tl.constexpr,
):
    pid = tl.program_id(0)
    iv = pid % NV
    t = pid // NV
    ihv = t % HV
    n = t // HV
    ih = ihv // RATIO

    o_v = iv * BV + tl.arange(0, BV)
    o_k = tl.arange(0, BK)

    slot = tl.load(idx_ptr + n).to(tl.int64)

    o_ptrs = o_ptr + n * o_s0 + ihv * o_s2 + o_v * o_s3
    o_dt = o_ptr.dtype.element_ty
    m_out = o_v < V

    if slot <= 0:
        # NULL_BLOCK_ID / PAD_SLOT_ID: zero this row of out, leave state alone.
        if EVEN_V:
            tl.store(o_ptrs, tl.zeros((BV,), dtype=o_dt))
        else:
            tl.store(o_ptrs, tl.zeros((BV,), dtype=o_dt), mask=m_out)
    else:
        # ---- fused gate (fp32 scalars) ----
        a_val = tl.load(a_ptr + n * a_s0 + ihv * a_s1).to(tl.float32)
        dt_val = tl.load(dt_ptr + ihv * dt_s0).to(tl.float32)
        x = a_val + dt_val
        xc = tl.minimum(x, 20.0)
        # softplus; log1p(exp(x)) as log(1+exp(min(x,20))), abs err ~1e-7 in a
        # term that is scaled by exp(A_log) and then exponentiated.
        sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(xc)), x)
        g = -tl.exp(tl.load(al_ptr + ihv * al_s0).to(tl.float32)) * sp
        eg = tl.exp(g)
        b_val = tl.load(b_ptr + n * b_s0 + ihv * b_s1)
        beta = (
            tl.sigmoid(b_val.to(tl.float32))
            .to(b_ptr.dtype.element_ty)
            .to(tl.float32)
        )

        # ---- q | k | v ----
        mq_row = mq_ptr + n * mq_s0
        m_k = o_k < K
        if EVEN_K:
            k_vec = tl.load(mq_row + H * K + ih * K + o_k).to(tl.float32)
            q_vec = tl.load(mq_row + ih * K + o_k).to(tl.float32)
        else:
            k_vec = tl.load(
                mq_row + H * K + ih * K + o_k, mask=m_k, other=0.0
            ).to(tl.float32)
            q_vec = tl.load(mq_row + ih * K + o_k, mask=m_k, other=0.0).to(
                tl.float32
            )
        if EVEN_V:
            v_vec = tl.load(mq_row + 2 * H * K + ihv * V + o_v).to(tl.float32)
        else:
            v_vec = tl.load(
                mq_row + 2 * H * K + ihv * V + o_v, mask=m_out, other=0.0
            ).to(tl.float32)

        if L2NORM:
            q_vec = q_vec / tl.sqrt(tl.sum(q_vec * q_vec, axis=0) + 1e-6)
            k_vec = k_vec / tl.sqrt(tl.sum(k_vec * k_vec, axis=0) + 1e-6)
        q_vec = q_vec * scale

        # ---- paged state tile ----
        st_ptrs = (
            st_ptr
            + slot * st_s0
            + ihv * st_s1
            + o_v[:, None] * st_s2
            + o_k[None, :] * st_s3
        )
        m_st = (o_v[:, None] < V) & (o_k[None, :] < K)
        if EVEN_V and EVEN_K:
            S = tl.load(st_ptrs).to(tl.float32)
        else:
            S = tl.load(st_ptrs, mask=m_st, other=0.0).to(tl.float32)

        # ---- delta-rule recurrence ----
        # The reference's two reductions are framewise `einsum` calls, which this
        # target lowers to a tf32-truncated matmul.  That operand rounding is
        # observable for fp32 operands and has to be reproduced; for fp16/bf16
        # operands the truncation is a bit-exact no-op, so it is compiled out.
        S = S * eg
        if TF32_EMULATE:
            pred = tl.sum(_tf32_trunc(S) * _tf32_trunc(k_vec)[None, :], axis=1)
        else:
            pred = tl.sum(S * k_vec[None, :], axis=1)
        vc = (v_vec - pred) * beta
        S = S + vc[:, None] * k_vec[None, :]
        if TF32_EMULATE:
            o = tl.sum(_tf32_trunc(S) * _tf32_trunc(q_vec)[None, :], axis=1)
        else:
            o = tl.sum(S * q_vec[None, :], axis=1)

        st_dt = st_ptr.dtype.element_ty
        if EVEN_V:
            tl.store(o_ptrs, o.to(o_dt))
        else:
            tl.store(o_ptrs, o.to(o_dt), mask=m_out)
        if EVEN_V and EVEN_K:
            tl.store(st_ptrs, S.to(st_dt))
        else:
            tl.store(st_ptrs, S.to(st_dt), mask=m_st)


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
    H = (qk_dim // 2) // K
    ratio = HV // H

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), max(16, 8192 // BK))
    BV = max(BV, 16)
    while BV > 16 and B * HV * triton.cdiv(V, BV) < 120:
        BV //= 2
    NV = triton.cdiv(V, BV)

    grid = (B * HV * NV,)
    _gdn_packed_decode_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        ssm_state_indices,
        float(scale),
        mixed_qkv.stride(0),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        A_log.stride(0),
        dt_bias.stride(0),
        initial_state.stride(0),
        initial_state.stride(1),
        initial_state.stride(2),
        initial_state.stride(3),
        out.stride(0),
        out.stride(2),
        out.stride(3),
        H,
        HV,
        V,
        K,
        ratio,
        BK=BK,
        BV=BV,
        NV=NV,
        L2NORM=bool(use_qk_l2norm_in_kernel),
        EVEN_V=(V % BV == 0),
        EVEN_K=(BK == K),
        TF32_EMULATE=(initial_state.dtype == torch.float32),
        num_warps=16,
    )
    return out, initial_state


__all__ = ["fused_recurrent_gated_delta_rule_packed_decode"]
