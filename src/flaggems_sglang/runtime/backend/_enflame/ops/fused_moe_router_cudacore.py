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
def _router_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    BS: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    STRIDE_X_B: tl.constexpr,
    STRIDE_X_H: tl.constexpr,
    STRIDE_W_E: tl.constexpr,
    STRIDE_W_H: tl.constexpr,
    CAP: tl.constexpr,
    HAS_CAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    e_live = offs_e < E
    t0 = pid_m * BLOCK_M
    t1 = t0 + 1
    t2 = t0 + 2
    t3 = t0 + 3
    acc0 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    even = (H % BLOCK_H == 0) and (BS % BLOCK_M == 0)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        if even:
            w = tl.load(
                w_ptr
                + offs_e[:, None] * STRIDE_W_E
                + offs_h[None, :] * STRIDE_W_H,
                mask=e_live[:, None],
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(x_ptr + t0 * STRIDE_X_B + offs_h * STRIDE_X_H).to(
                tl.float32
            )
            x1 = tl.load(x_ptr + t1 * STRIDE_X_B + offs_h * STRIDE_X_H).to(
                tl.float32
            )
            x2 = tl.load(x_ptr + t2 * STRIDE_X_B + offs_h * STRIDE_X_H).to(
                tl.float32
            )
            x3 = tl.load(x_ptr + t3 * STRIDE_X_B + offs_h * STRIDE_X_H).to(
                tl.float32
            )
        else:
            h_live = offs_h < H
            w = tl.load(
                w_ptr
                + offs_e[:, None] * STRIDE_W_E
                + offs_h[None, :] * STRIDE_W_H,
                mask=e_live[:, None] & h_live[None, :],
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(
                x_ptr + t0 * STRIDE_X_B + offs_h * STRIDE_X_H,
                mask=(t0 < BS) & h_live,
                other=0.0,
            ).to(tl.float32)
            x1 = tl.load(
                x_ptr + t1 * STRIDE_X_B + offs_h * STRIDE_X_H,
                mask=(t1 < BS) & h_live,
                other=0.0,
            ).to(tl.float32)
            x2 = tl.load(
                x_ptr + t2 * STRIDE_X_B + offs_h * STRIDE_X_H,
                mask=(t2 < BS) & h_live,
                other=0.0,
            ).to(tl.float32)
            x3 = tl.load(
                x_ptr + t3 * STRIDE_X_B + offs_h * STRIDE_X_H,
                mask=(t3 < BS) & h_live,
                other=0.0,
            ).to(tl.float32)
        acc0 += tl.sum(w * x0[None, :], axis=1)
        acc1 += tl.sum(w * x1[None, :], axis=1)
        acc2 += tl.sum(w * x2[None, :], axis=1)
        acc3 += tl.sum(w * x3[None, :], axis=1)
    if HAS_CAP:
        acc0 = (2.0 * tl.sigmoid(2.0 * (acc0 / CAP)) - 1.0) * CAP
        acc1 = (2.0 * tl.sigmoid(2.0 * (acc1 / CAP)) - 1.0) * CAP
        acc2 = (2.0 * tl.sigmoid(2.0 * (acc2 / CAP)) - 1.0) * CAP
        acc3 = (2.0 * tl.sigmoid(2.0 * (acc3 / CAP)) - 1.0) * CAP
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=e_live, other=0.0).to(
            tl.float32
        )
        acc0 += bias
        acc1 += bias
        acc2 += bias
        acc3 += bias
    logits0 = tl.where(e_live, acc0, NEG)
    top0 = tl.max(logits0, axis=0)
    inv0 = 1.0 / tl.sum(tl.exp(logits0 - top0), axis=0)
    rem0 = logits0
    for slot in range(0, TOPK):
        sel = tl.max(rem0, axis=0)
        expert = tl.min(tl.where(rem0 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t0 * TOPK + slot, expert, mask=t0 < BS)
        tl.store(
            weights_ptr + t0 * TOPK + slot,
            tl.exp(sel - top0) * inv0,
            mask=t0 < BS,
        )
        rem0 = tl.where(offs_e == expert, NEG, rem0)

    logits1 = tl.where(e_live, acc1, NEG)
    top1 = tl.max(logits1, axis=0)
    inv1 = 1.0 / tl.sum(tl.exp(logits1 - top1), axis=0)
    rem1 = logits1
    for slot in range(0, TOPK):
        sel = tl.max(rem1, axis=0)
        expert = tl.min(tl.where(rem1 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t1 * TOPK + slot, expert, mask=t1 < BS)
        tl.store(
            weights_ptr + t1 * TOPK + slot,
            tl.exp(sel - top1) * inv1,
            mask=t1 < BS,
        )
        rem1 = tl.where(offs_e == expert, NEG, rem1)

    logits2 = tl.where(e_live, acc2, NEG)
    top2 = tl.max(logits2, axis=0)
    inv2 = 1.0 / tl.sum(tl.exp(logits2 - top2), axis=0)
    rem2 = logits2
    for slot in range(0, TOPK):
        sel = tl.max(rem2, axis=0)
        expert = tl.min(tl.where(rem2 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t2 * TOPK + slot, expert, mask=t2 < BS)
        tl.store(
            weights_ptr + t2 * TOPK + slot,
            tl.exp(sel - top2) * inv2,
            mask=t2 < BS,
        )
        rem2 = tl.where(offs_e == expert, NEG, rem2)

    logits3 = tl.where(e_live, acc3, NEG)
    top3 = tl.max(logits3, axis=0)
    inv3 = 1.0 / tl.sum(tl.exp(logits3 - top3), axis=0)
    rem3 = logits3
    for slot in range(0, TOPK):
        sel = tl.max(rem3, axis=0)
        expert = tl.min(tl.where(rem3 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t3 * TOPK + slot, expert, mask=t3 < BS)
        tl.store(
            weights_ptr + t3 * TOPK + slot,
            tl.exp(sel - top3) * inv3,
            mask=t3 < BS,
        )
        rem3 = tl.where(offs_e == expert, NEG, rem3)


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
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    expert_block = tl.program_id(1)
    offs_e = expert_block * BLOCK_E + tl.arange(0, BLOCK_E)
    e_live = offs_e < E
    t0 = pid_m * BLOCK_M

    acc0 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    even = (H % BLOCK_H == 0) and (E % BLOCK_E == 0) and (BS % BLOCK_M == 0)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        if even:
            w = tl.load(
                w_ptr
                + offs_e[:, None] * STRIDE_W_E
                + offs_h[None, :] * STRIDE_W_H,
            ).to(tl.float32)
            x0 = tl.load(x_ptr + t0 * STRIDE_X_B + offs_h * STRIDE_X_H).to(
                tl.float32
            )
        else:
            h_live = offs_h < H
            w = tl.load(
                w_ptr
                + offs_e[:, None] * STRIDE_W_E
                + offs_h[None, :] * STRIDE_W_H,
                mask=e_live[:, None] & h_live[None, :],
                other=0.0,
            ).to(tl.float32)
            x0 = tl.load(
                x_ptr + t0 * STRIDE_X_B + offs_h * STRIDE_X_H,
                mask=(t0 < BS) & h_live,
                other=0.0,
            ).to(tl.float32)
        acc0 += tl.sum(w * x0[None, :], axis=1)
    if HAS_CAP:
        acc0 = (2.0 * tl.sigmoid(2.0 * (acc0 / CAP)) - 1.0) * CAP
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=e_live, other=0.0).to(
            tl.float32
        )
        acc0 += bias
    tl.store(logits_ptr + t0 * E + offs_e, acc0, mask=e_live & (t0 < BS))


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
    bias_arg = correction_bias if has_bias else x
    sx = x.stride()
    sw = weights.stride()
    neg = -3.0e38
    if _next_pow2(experts) <= 32:
        topk_weights = torch.empty(
            (bs, topk), dtype=torch.float32, device=x.device
        )
        topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
        sx = x.stride()
        sw = router_weight.stride()
        block_e = _next_pow2(experts)
        block_h = 32768 // block_e
        if block_h > hidden:
            block_h = _next_pow2(hidden)
        if block_h < 16:
            block_h = 16
        block_m = 4
        _router_kernel[((bs + block_m - 1) // block_m,)](
            x,
            router_weight,
            correction_bias if has_bias else x,
            topk_weights,
            topk_ids,
            bs,
            hidden,
            experts,
            topk,
            sx[0],
            sx[1],
            sw[0],
            sw[1],
            cap,
            has_cap,
            has_bias,
            -3.0e38,
            block_e,
            block_h,
            block_m,
            num_warps=1,
            num_stages=2,
        )
    else:
        logits = torch.empty(
            (bs, experts), dtype=torch.float32, device=x.device
        )
        topk_weights = torch.empty(
            (bs, topk), dtype=torch.float32, device=x.device
        )
        topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
        bias_arg = correction_bias if has_bias else x
        sx = x.stride()
        sw = weights.stride()
        neg = -3.0e38
        block_m = 1
        block_e = 8 if experts >= 8 else _next_pow2(max(experts, 1))
        block_h = (
            4096
            if hidden >= 4096
            else (
                2048
                if hidden >= 2048
                else (1024 if hidden >= 1024 else _next_pow2(hidden))
            )
        )
        grid_n = (experts + block_e - 1) // block_e
        grid_m = (bs + block_m - 1) // block_m
        if grid_n > 255:
            block_e = _next_pow2((experts + 254) // 255)
            grid_n = (experts + block_e - 1) // block_e
        row_start = 0
        while row_start < grid_m:
            row_count = min(grid_m - row_start, 65535)
            _logits_kernel[(row_count, grid_n)](
                x[row_start * block_m :] if row_start else x,
                weights,
                bias_arg,
                logits[row_start * block_m :] if row_start else logits,
                bs - row_start * block_m,
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
                block_m,
                num_warps=1,
                num_stages=2,
            )
            row_start += row_count

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
