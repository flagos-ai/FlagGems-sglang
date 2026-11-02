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

"""Fused recurrent gated delta rule, packed decode step (single token).

Triton implementation of
vllm/model_executor/layers/fla/ops/fused_recurrent.py
::fused_recurrent_gated_delta_rule_packed_decode_kernel.

One program handles one (sequence n, value head i_hv, v-tile) triple.
"""

import triton
import triton.language as tl


@triton.jit
def _gdn_packed_decode_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    out_ptr,
    ssm_idx_ptr,
    scale,
    q_dim,
    hv_dim,
    v_dim,
    k_dim,
    ratio,
    stride_mq_tok,
    stride_a_tok,
    stride_state_slot,
    stride_state_hv,
    stride_state_v,
    stride_out_n,
    stride_out_hv,
    stride_out_v,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_L2NORM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_vb = tl.cdiv(v_dim, BLOCK_V)

    vb = pid % num_vb
    tmp = pid // num_vb
    i_hv = tmp % hv_dim
    n = tmp // hv_dim

    v_off = vb * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_off < v_dim
    k_off = tl.arange(0, BLOCK_K)
    k_mask = k_off < k_dim

    o_ptrs = (
        out_ptr
        + n * stride_out_n
        + i_hv * stride_out_hv
        + v_off * stride_out_v
    )

    slot = tl.load(ssm_idx_ptr + n).to(tl.int64)
    if slot <= 0:
        tl.store(
            o_ptrs,
            tl.zeros([BLOCK_V], dtype=out_ptr.dtype.element_ty),
            mask=v_mask,
        )
        return

    # Grouped value attention: value head i_hv reads query/key head i_hv // ratio.
    i_h = i_hv // ratio
    tok_base = mixed_qkv_ptr + n * stride_mq_tok
    q_vec = tl.load(tok_base + i_h * k_dim + k_off, mask=k_mask, other=0.0).to(
        tl.float32
    )
    k_vec = tl.load(
        tok_base + q_dim + i_h * k_dim + k_off, mask=k_mask, other=0.0
    ).to(tl.float32)

    if USE_L2NORM:
        q_vec = q_vec / tl.sqrt(tl.sum(q_vec * q_vec) + 1e-6)
        k_vec = k_vec / tl.sqrt(tl.sum(k_vec * k_vec) + 1e-6)
    q_vec = q_vec * scale

    a_val = tl.load(a_ptr + n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b_ptr + n * stride_a_tok + i_hv).to(tl.float32)
    A_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
    d_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

    x = a_val + d_val
    sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))), x)
    decay = tl.exp(-tl.exp(A_val) * sp)
    beta = tl.sigmoid(b_val).to(b_ptr.dtype.element_ty).to(tl.float32)

    s_ptrs = (
        state_ptr
        + slot * stride_state_slot
        + i_hv * stride_state_hv
        + v_off[:, None] * stride_state_v
        + k_off[None, :]
    )
    s_mask = v_mask[:, None] & k_mask[None, :]
    state = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    state = state * decay
    pred = tl.sum(state * k_vec[None, :], axis=1)
    v_vec = tl.load(
        tok_base + 2 * q_dim + i_hv * v_dim + v_off, mask=v_mask, other=0.0
    ).to(tl.float32)
    vc = (v_vec - pred) * beta
    state = state + vc[:, None] * k_vec[None, :]

    o = tl.sum(state * q_vec[None, :], axis=1)

    tl.store(o_ptrs, o.to(out_ptr.dtype.element_ty), mask=v_mask)
    tl.store(s_ptrs, state.to(state_ptr.dtype.element_ty), mask=s_mask)


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

    BLOCK_K = max(16, triton.next_power_of_2(K))
    BLOCK_V = min(128, max(16, triton.next_power_of_2(V)))

    nvb = triton.cdiv(V, BLOCK_V)
    grid = (B * HV * nvb,)

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
        q_dim,
        HV,
        V,
        K,
        ratio,
        mixed_qkv.stride(0),
        a.stride(0),
        initial_state.stride(0),
        initial_state.stride(1),
        initial_state.stride(2),
        out.stride(0),
        out.stride(2),
        out.stride(3),
        BLOCK_V=BLOCK_V,
        BLOCK_K=BLOCK_K,
        USE_L2NORM=bool(use_qk_l2norm_in_kernel),
    )

    return out, initial_state


__all__ = ["fused_recurrent_gated_delta_rule_packed_decode"]
