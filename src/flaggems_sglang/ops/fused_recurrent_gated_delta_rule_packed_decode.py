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

"""Fused recurrent gated-delta-rule packed decode (single token).

Single Triton kernel that consumes the packed QKV projection directly and
performs the whole straight-line recurrence (gate, delta-rule update, readout,
state writeback) in one pass over the paged state.
"""

import triton
import triton.language as tl


@triton.jit
def _packed_decode_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    out_ptr,
    idx_ptr,
    scale,
    HV: tl.constexpr,
    H: tl.constexpr,
    stride_qkv_tok: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_hv: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_out_n: tl.constexpr,
    stride_out_hv: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    EVEN_V: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_L2NORM: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_nh = tl.program_id(1)
    i_n = pid_nh // HV
    i_hv = pid_nh % HV

    v_idx = pid_v * BV + tl.arange(0, BV)
    k_idx = tl.arange(0, BK)

    out_base = out_ptr + i_n * stride_out_n + i_hv * stride_out_hv

    slot = tl.load(idx_ptr + i_n)

    if slot <= 0:
        # NULL_BLOCK_ID / PAD_SLOT_ID: zero `out`, leave the state untouched.
        zval = tl.zeros([BV], dtype=tl.float32).to(out_ptr.dtype.element_ty)
        if EVEN_V:
            tl.store(out_base + v_idx, zval)
        else:
            tl.store(out_base + v_idx, zval, mask=v_idx < V)
    else:
        ratio = HV // H
        i_h = i_hv // ratio

        a_v = tl.load(a_ptr + i_n * HV + i_hv).to(tl.float32)
        b_raw = tl.load(b_ptr + i_n * HV + i_hv)
        # deliberate round-trip through b's dtype, as in the reference kernel
        beta = tl.sigmoid(b_raw.to(tl.float32)).to(b_raw.dtype).to(tl.float32)
        alog = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dtb = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

        x = a_v + dtb
        sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))), x)
        decay = tl.exp(-tl.exp(alog) * sp)

        row = mixed_qkv_ptr + i_n * stride_qkv_tok
        qk_half = H * K
        q_base = row + i_h * K
        k_base = row + qk_half + i_h * K
        v_base = row + 2 * qk_half + i_hv * V

        if EVEN_K:
            q_vec = tl.load(q_base + k_idx).to(tl.float32)
            k_vec = tl.load(k_base + k_idx).to(tl.float32)
        else:
            km = k_idx < K
            q_vec = tl.load(q_base + k_idx, mask=km, other=0.0).to(tl.float32)
            k_vec = tl.load(k_base + k_idx, mask=km, other=0.0).to(tl.float32)
        if EVEN_V:
            v_vec = tl.load(v_base + v_idx).to(tl.float32)
        else:
            v_vec = tl.load(v_base + v_idx, mask=v_idx < V, other=0.0).to(
                tl.float32
            )

        if USE_L2NORM:
            q_vec = q_vec / tl.sqrt(tl.sum(q_vec * q_vec) + 1e-6)
            k_vec = k_vec / tl.sqrt(tl.sum(k_vec * k_vec) + 1e-6)
        q_vec = q_vec * scale

        s_base = state_ptr + slot * stride_state_slot + i_hv * stride_state_hv
        offs = (
            v_idx[:, None] * stride_state_v + k_idx[None, :] * stride_state_k
        )

        if EVEN_V and EVEN_K:
            h = tl.load(s_base + offs).to(tl.float32)
        elif EVEN_V:
            h = tl.load(
                s_base + offs, mask=(k_idx < K)[None, :], other=0.0
            ).to(tl.float32)
        elif EVEN_K:
            h = tl.load(
                s_base + offs, mask=(v_idx < V)[:, None], other=0.0
            ).to(tl.float32)
        else:
            h = tl.load(
                s_base + offs,
                mask=(v_idx < V)[:, None] & (k_idx < K)[None, :],
                other=0.0,
            ).to(tl.float32)

        # pred = sum_k (decay*h)[v,k]*k[k] = decay * sum_k h[v,k]*k[k]
        hk = tl.sum(h * k_vec[None, :], axis=1)
        hq = tl.sum(h * q_vec[None, :], axis=1)
        kq = tl.sum(k_vec * q_vec, axis=0)

        pred = decay * hk
        vc = (v_vec - pred) * beta
        o = decay * hq + vc * kq
        h2 = decay * h + vc[:, None] * k_vec[None, :]

        o_val = o.to(out_ptr.dtype.element_ty)
        if EVEN_V:
            tl.store(out_base + v_idx, o_val)
        else:
            tl.store(out_base + v_idx, o_val, mask=v_idx < V)

        h2_val = h2.to(state_ptr.dtype.element_ty)
        if EVEN_V and EVEN_K:
            tl.store(s_base + offs, h2_val)
        elif EVEN_V:
            tl.store(s_base + offs, h2_val, mask=(k_idx < K)[None, :])
        elif EVEN_K:
            tl.store(s_base + offs, h2_val, mask=(v_idx < V)[:, None])
        else:
            tl.store(
                s_base + offs,
                h2_val,
                mask=(v_idx < V)[:, None] & (k_idx < K)[None, :],
            )


def _pick_block(V, K):
    BK = max(16, triton.next_power_of_2(K))
    BV = min(64, max(16, triton.next_power_of_2(V)))
    return BV, BK


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

    BV, BK = _pick_block(V, K)
    nw = 8 if BV * BK >= 8192 else 4

    grid = (triton.cdiv(V, BV), B * HV)
    _packed_decode_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        ssm_state_indices,
        float(scale),
        int(HV),
        int(H),
        int(mixed_qkv.stride(0)),
        int(initial_state.stride(0)),
        int(initial_state.stride(1)),
        int(initial_state.stride(2)),
        int(initial_state.stride(3)),
        int(out.stride(0)),
        int(out.stride(2)),
        V=int(V),
        K=int(K),
        BV=BV,
        BK=BK,
        EVEN_V=(V % BV) == 0,
        EVEN_K=(BK == K),
        USE_L2NORM=bool(use_qk_l2norm_in_kernel),
        num_warps=nw,
    )
    return out, initial_state


__all__ = ["fused_recurrent_gated_delta_rule_packed_decode"]
