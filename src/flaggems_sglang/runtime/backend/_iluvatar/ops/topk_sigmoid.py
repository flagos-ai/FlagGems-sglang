# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import triton
import triton.language as tl

_PACK_MAX_TOKENS = 8


@triton.jit
def _topk_sigmoid_pack_kernel(
    gating_ptr,
    weights_ptr,
    ids_ptr,
    stride_gm,
    n_experts,
    k,
    scale,
    renorm: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    e_off = tl.arange(0, BLOCK_E)
    mask_e = e_off < n_experts
    logits = tl.load(
        gating_ptr + row * stride_gm + e_off, mask=mask_e, other=float("-inf")
    )

    lbits = logits.to(tl.int32, bitcast=True)
    ordered = tl.where(
        lbits >= 0, lbits ^ tl.cast(-(1 << 31), tl.int32), ~lbits
    )
    key = ((ordered.to(tl.int64) - 0x80000000) << 32) + (
        0x7FFFFFFF - e_off.to(tl.int64)
    )
    top = tl.topk(key, BLOCK_K)

    rk = tl.arange(0, BLOCK_K)
    idx = (0x7FFFFFFF - (top & 0x7FFFFFFF)).to(tl.int32)
    mask_k = rk < k
    wl = tl.load(
        gating_ptr + row * stride_gm + idx, mask=mask_k, other=0.0
    ).to(tl.float32)
    w = 1.0 / (1.0 + tl.exp(-wl))

    if renorm:
        total = tl.sum(w, axis=0) + 1e-20
        w = w * scale / total

    tl.store(weights_ptr + row * k + rk, w, mask=mask_k)
    tl.store(ids_ptr + row * k + rk, idx, mask=mask_k)


@triton.jit
def _topk_sigmoid_argmax_kernel(
    gating_ptr,
    weights_ptr,
    ids_ptr,
    stride_gm,
    n_experts,
    scale,
    renorm: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    e_off = tl.arange(0, BLOCK_E)
    mask_e = e_off < n_experts
    logits = tl.load(
        gating_ptr + row * stride_gm + e_off, mask=mask_e, other=float("-inf")
    )

    rk = tl.arange(0, BLOCK_K)
    w_acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    id_acc = tl.zeros([BLOCK_K], dtype=tl.int32)
    for i in tl.static_range(0, K):
        val = tl.max(logits, axis=0)
        eq = logits == val
        idx = tl.min(tl.where(eq, e_off, BLOCK_E), axis=0)
        logits = tl.where(eq, float("-inf"), logits)
        w_acc = tl.where(rk == i, val, w_acc)
        id_acc = tl.where(rk == i, idx.to(tl.int32), id_acc)

    mask_k = rk < K
    wl = tl.load(
        gating_ptr + row * stride_gm + id_acc, mask=mask_k, other=0.0
    ).to(tl.float32)
    w_acc = tl.where(mask_k, 1.0 / (1.0 + tl.exp(-wl)), 0.0)

    if renorm:
        total = tl.sum(w_acc, axis=0) + 1e-20
        w_acc = w_acc * scale / total

    tl.store(weights_ptr + row * K + rk, w_acc, mask=mask_k)
    tl.store(ids_ptr + row * K + rk, id_acc, mask=mask_k)


def topk_sigmoid(
    topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor
):
    num_tokens, num_experts = gating_output.shape
    k = topk_weights.shape[1]
    if gating_output.stride(1) != 1:
        gating_output = gating_output.contiguous()

    if num_tokens <= _PACK_MAX_TOKENS:
        _topk_sigmoid_pack_kernel[(num_tokens,)](
            gating_output,
            topk_weights,
            topk_ids,
            gating_output.stride(0),
            num_experts,
            k,
            float(routed_scaling_factor),
            renorm=bool(renormalize),
            BLOCK_E=triton.next_power_of_2(num_experts),
            BLOCK_K=triton.next_power_of_2(k),
            num_warps=4,
            num_stages=1,
        )
    else:
        _topk_sigmoid_argmax_kernel[(num_tokens,)](
            gating_output,
            topk_weights,
            topk_ids,
            gating_output.stride(0),
            num_experts,
            float(routed_scaling_factor),
            renorm=bool(renormalize),
            K=k,
            BLOCK_E=triton.next_power_of_2(num_experts),
            BLOCK_K=triton.next_power_of_2(k),
            num_warps=1,
            num_stages=1,
        )
    return topk_weights, topk_ids


__all__ = ["topk_sigmoid"]
