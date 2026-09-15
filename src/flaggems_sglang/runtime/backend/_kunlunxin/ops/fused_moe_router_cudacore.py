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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _logits_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    logits_ptr,
    H: tl.constexpr,
    E: tl.constexpr,
    STRIDE_X_B: tl.constexpr,
    STRIDE_X_H: tl.constexpr,
    STRIDE_W_E: tl.constexpr,
    STRIDE_W_H: tl.constexpr,
    CAP: tl.constexpr,
    HAS_CAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    N_H: tl.constexpr,
):
    token = tl.program_id(0)
    expert_block = tl.program_id(1)
    offs_e = expert_block * BLOCK_E + tl.arange(0, BLOCK_E)
    e_live = offs_e < E
    totals = tl.zeros((BLOCK_E,), dtype=tl.float32)
    offs_h = tl.arange(0, BLOCK_H)
    for h_block in tl.static_range(0, N_H):
        h = h_block * BLOCK_H + offs_h
        if H % BLOCK_H == 0 and E % BLOCK_E == 0:
            x = tl.load(x_ptr + token * STRIDE_X_B + h * STRIDE_X_H).to(
                tl.float32
            )
            w = tl.load(
                w_ptr + offs_e[:, None] * STRIDE_W_E + h[None, :] * STRIDE_W_H,
            ).to(tl.float32)
        else:
            safe_e = tl.minimum(offs_e, E - 1)
            safe_h = tl.minimum(h, H - 1)
            x = tl.load(x_ptr + token * STRIDE_X_B + safe_h * STRIDE_X_H).to(
                tl.float32
            )
            w = tl.load(
                w_ptr
                + safe_e[:, None] * STRIDE_W_E
                + safe_h[None, :] * STRIDE_W_H,
            ).to(tl.float32)
            x = tl.where(h < H, x, 0.0)
            w = tl.where(e_live[:, None] & (h[None, :] < H), w, 0.0)
        totals += tl.sum(w * x[None, :], axis=1)
    if HAS_CAP:
        totals = (2.0 * tl.sigmoid(2.0 * (totals / CAP)) - 1.0) * CAP
    if HAS_BIAS:
        totals += tl.load(bias_ptr + offs_e, mask=e_live, other=0.0).to(
            tl.float32
        )
    tl.store(logits_ptr + token * E + offs_e, totals, mask=e_live)


@triton.jit
def _topk_kernel(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NEG: tl.constexpr,
):
    token = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    live = offs < E
    logits = tl.load(logits_ptr + token * E + offs, mask=live, other=NEG)
    logits = tl.where(live, logits, NEG)
    row_max = tl.max(logits, axis=0)
    inv = 1.0 / tl.sum(tl.exp(logits - row_max), axis=0)
    remaining = logits
    for slot in tl.static_range(0, TOPK):
        selected = tl.max(remaining, axis=0)
        expert = tl.min(tl.where(remaining == selected, offs, BLOCK_E), axis=0)
        tl.store(ids_ptr + token * TOPK + slot, expert)
        tl.store(
            weights_ptr + token * TOPK + slot, tl.exp(selected - row_max) * inv
        )
        remaining = tl.where(offs == expert, NEG, remaining)


def _next_pow2(value):
    result = 1
    while result < value:
        result *= 2
    return result


def fused_moe_router_cudacore(
    x, router_weight, topk, moe_softcapping, correction_bias=None
):
    x = x.contiguous()
    weights = router_weight.contiguous()
    bs, hidden = x.shape
    experts = weights.shape[0]
    topk = int(topk)
    cap = float(moe_softcapping)
    has_cap = cap != 0.0
    has_bias = correction_bias is not None
    logits = torch.empty((bs, experts), dtype=torch.float32, device=x.device)
    topk_weights = torch.empty(
        (bs, topk), dtype=torch.float32, device=x.device
    )
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
    bias_arg = correction_bias if has_bias else x
    sx = x.stride()
    sw = weights.stride()
    neg = -3.0e38

    block_e = 4 if experts >= 4 else _next_pow2(max(experts, 1))
    block_h = 2048
    while block_h > 32 and (hidden + block_h - 1) // block_h > 8:
        block_h = block_h // 2
    if hidden < block_h:
        block_h = _next_pow2(hidden)
    n_h = (hidden + block_h - 1) // block_h
    grid_n = (experts + block_e - 1) // block_e

    _logits_kernel[(bs, grid_n)](
        x,
        weights,
        bias_arg,
        logits,
        hidden,
        experts,
        sx[0],
        sx[1],
        sw[0],
        sw[1],
        cap,
        has_cap,
        has_bias,
        block_e,
        block_h,
        n_h,
        num_warps=8,
        num_stages=1,
    )
    _topk_kernel[(bs,)](
        logits,
        topk_weights,
        topk_ids,
        experts,
        topk,
        _next_pow2(experts),
        neg,
        num_warps=1,
        num_stages=1,
    )
    return topk_weights, topk_ids


__all__ = ["fused_moe_router_cudacore"]
