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
    BS: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    STRIDE_X_B: tl.constexpr,
    STRIDE_X_H: tl.constexpr,
    STRIDE_W_E: tl.constexpr,
    STRIDE_W_H: tl.constexpr,
    CAP: tl.constexpr,
    HAS_CAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    STRIDE_ORD: tl.constexpr,
    ROT_ORD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_e = pid_n * BLOCK_E + tl.arange(0, BLOCK_E)
    m_live = offs_m < BS
    e_live = offs_e < E
    acc = tl.zeros((BLOCK_M, BLOCK_E), dtype=tl.float32)
    offs_h0 = tl.arange(0, BLOCK_H)
    even = (H % BLOCK_H == 0) and (BS % BLOCK_M == 0) and (E % BLOCK_E == 0)
    for i in range(0, N_BLOCKS):
        h_block = (i * STRIDE_ORD + ROT_ORD) % N_BLOCKS
        h = h_block * BLOCK_H + offs_h0
        if even:
            x = tl.load(
                x_ptr + offs_m[:, None] * STRIDE_X_B + h[None, :] * STRIDE_X_H,
            ).to(tl.float32)
            w = tl.load(
                w_ptr + offs_e[:, None] * STRIDE_W_E + h[None, :] * STRIDE_W_H,
            ).to(tl.float32)
        else:
            safe_m = tl.minimum(offs_m, BS - 1)
            safe_e = tl.minimum(offs_e, E - 1)
            safe_h = tl.minimum(h, H - 1)
            x = tl.load(
                x_ptr
                + safe_m[:, None] * STRIDE_X_B
                + safe_h[None, :] * STRIDE_X_H,
            ).to(tl.float32)
            w = tl.load(
                w_ptr
                + safe_e[:, None] * STRIDE_W_E
                + safe_h[None, :] * STRIDE_W_H,
            ).to(tl.float32)
            x = tl.where((offs_m[:, None] < BS) & (h[None, :] < H), x, 0.0)
            w = tl.where((offs_e[:, None] < E) & (h[None, :] < H), w, 0.0)
        acc += tl.sum(x[:, None, :] * w[None, :, :], axis=2)
    if HAS_CAP:
        acc = (2.0 * tl.sigmoid(2.0 * (acc / CAP)) - 1.0) * CAP
    if HAS_BIAS:
        if E % BLOCK_E == 0:
            acc += tl.load(bias_ptr + offs_e).to(tl.float32)[None, :]
        else:
            safe_e = tl.minimum(offs_e, E - 1)
            bias = tl.load(bias_ptr + safe_e).to(tl.float32)
            acc += tl.where(e_live, bias, 0.0)[None, :]
    tl.store(
        logits_ptr + offs_m[:, None] * E + offs_e[None, :],
        acc,
        mask=m_live[:, None] & e_live[None, :],
    )


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
    offs_e = tl.arange(0, BLOCK_E)
    e_live = offs_e < E
    logits = tl.load(logits_ptr + token * E + offs_e, mask=e_live, other=NEG)
    logits = tl.where(e_live, logits, NEG)
    row_max = tl.max(logits, axis=0)
    inv = 1.0 / tl.sum(tl.exp(logits - row_max), axis=0)
    remaining = logits
    for slot in range(0, TOPK):
        selected = tl.max(remaining, axis=0)
        expert = tl.min(
            tl.where(remaining == selected, offs_e, BLOCK_E), axis=0
        )
        tl.store(ids_ptr + token * TOPK + slot, expert)
        tl.store(
            weights_ptr + token * TOPK + slot, tl.exp(selected - row_max) * inv
        )
        remaining = tl.where(offs_e == expert, NEG, remaining)


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
    block_h = 8
    n_blocks = (hidden + block_h - 1) // block_h
    if bs >= 64 and bs % 64 == 0:
        block_m = 64
    elif bs >= 32 and bs % 32 == 0:
        block_m = 32
    elif bs >= 16 and bs % 16 == 0:
        block_m = 16
    else:
        block_m = 1
    if experts >= 32:
        block_e = 32
    elif experts >= 16:
        block_e = 16
    else:
        block_e = _next_pow2(max(experts, 8))
    token_blocks = (bs + block_m - 1) // block_m
    expert_blocks = (experts + block_e - 1) // block_e
    if token_blocks * expert_blocks > 65535:
        block_e = max(
            1,
            (experts + 65535 // max(token_blocks, 1) - 1)
            // max(1, 65535 // max(token_blocks, 1)),
        )
        block_e = _next_pow2(block_e)
        expert_blocks = (experts + block_e - 1) // block_e
    _logits_kernel[(token_blocks, expert_blocks)](
        x,
        weights,
        bias_arg,
        logits,
        bs,
        hidden,
        experts,
        sx[0],
        sx[1],
        sw[0],
        sw[1],
        cap,
        has_cap,
        has_bias,
        block_m,
        block_e,
        block_h,
        n_blocks,
        45 if n_blocks % 3 != 0 and n_blocks % 5 != 0 else 1,
        12,
        num_warps=4,
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
