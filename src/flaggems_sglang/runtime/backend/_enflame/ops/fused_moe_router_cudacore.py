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

# GCU (2026-11-08 工具链) 免疫形态说明:
# - 20:22 工具链回归使 2D tile 访存 ([:, None]/[None, :] 仿射) 首编挂
#   memref_ext.slice_start/slice_pad_start (见 compiler_issues/ 家族档案);
#   BLOCK_H > 2048 的单条 1D load 另有静默错值面 (探针实证 err~0.2)。
# - 因此本实现全部访存为 1D 仿射: w 行逐条 load 一次服务 BLOCK_M 个 token,
#   x 以 bf16 驻留 (BLOCK_M=2), 累加器为窄 [BLOCK_E] 向量跨 h 循环携带。
#
# 2026-11-12 更新: P01a 定案 (2026-11-10) 后 2D tile / tl.dot fresh 编译
# 通道已恢复 (外部 gcu-compiler-opt 对齐 8 段 operandSegmentSizes;
# fused_moe_gemm / bmm_chunk 11-11 重放全绿先例)。E>32 的 logits GEMM
# 增设 `_logits_dot_kernel` MMA 臂 (fused_moe_gemm 同款全 K 单 dot +
# static_range N-chunk 形态, bf16 操作数 + fp32 累加), 1D 树形 reduce
# 形态保留为回退臂 (非 bf16 / H>4096 / E>8160 / bs==1 / dot 编译失败)。
# 配参 (探针三轮定标): PAD_M=32, NCHUNKS=1, nw=1; bs>=64 且 E>=256 且
# E%64==0 时 BLOCK_N=64, 否则 BLOCK_N=32。
# 实测 (bench core 官方口径两轮, E=256/H=4096/topk=8/bf16): bs=8/64/512/
# 4096 = 27.8/102.1/719.2/7006.6µs → 11.9/16.8/74.9/486.9µs
# (2.33/6.09/9.60/14.39×, 全档反超 torch 4.6/3.3/2.3/1.16×);
# bs=1 dot 无优势 (MMA 行填充 1/32), 保持 1D 臂。


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
):
    pid_m = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    e_live = offs_e < E
    t0 = pid_m * 4
    t1 = t0 + 1
    t2 = t0 + 2
    t3 = t0 + 3
    acc0 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        h_live = offs_h < H
        xb0 = tl.load(x_ptr + t0 * STRIDE_X_B + offs_h * STRIDE_X_H, mask=(t0 < BS) & h_live, other=0.0)
        xb1 = tl.load(x_ptr + t1 * STRIDE_X_B + offs_h * STRIDE_X_H, mask=(t1 < BS) & h_live, other=0.0)
        xb2 = tl.load(x_ptr + t2 * STRIDE_X_B + offs_h * STRIDE_X_H, mask=(t2 < BS) & h_live, other=0.0)
        xb3 = tl.load(x_ptr + t3 * STRIDE_X_B + offs_h * STRIDE_X_H, mask=(t3 < BS) & h_live, other=0.0)
        for e in tl.static_range(BLOCK_E):
            w_e = tl.load(w_ptr + e * STRIDE_W_E + offs_h * STRIDE_W_H, mask=(e < E) & h_live, other=0.0)
            sel = offs_e == e
            acc0 += tl.where(sel, tl.sum(w_e.to(tl.float32) * xb0.to(tl.float32), axis=0), 0.0)
            acc1 += tl.where(sel, tl.sum(w_e.to(tl.float32) * xb1.to(tl.float32), axis=0), 0.0)
            acc2 += tl.where(sel, tl.sum(w_e.to(tl.float32) * xb2.to(tl.float32), axis=0), 0.0)
            acc3 += tl.where(sel, tl.sum(w_e.to(tl.float32) * xb3.to(tl.float32), axis=0), 0.0)
    if HAS_CAP:
        acc0 = (2.0 * tl.sigmoid(2.0 * (acc0 / CAP)) - 1.0) * CAP
        acc1 = (2.0 * tl.sigmoid(2.0 * (acc1 / CAP)) - 1.0) * CAP
        acc2 = (2.0 * tl.sigmoid(2.0 * (acc2 / CAP)) - 1.0) * CAP
        acc3 = (2.0 * tl.sigmoid(2.0 * (acc3 / CAP)) - 1.0) * CAP
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=e_live, other=0.0).to(tl.float32)
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
        tl.store(weights_ptr + t0 * TOPK + slot, tl.exp(sel - top0) * inv0, mask=t0 < BS)
        rem0 = tl.where(offs_e == expert, NEG, rem0)

    logits1 = tl.where(e_live, acc1, NEG)
    top1 = tl.max(logits1, axis=0)
    inv1 = 1.0 / tl.sum(tl.exp(logits1 - top1), axis=0)
    rem1 = logits1
    for slot in range(0, TOPK):
        sel = tl.max(rem1, axis=0)
        expert = tl.min(tl.where(rem1 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t1 * TOPK + slot, expert, mask=t1 < BS)
        tl.store(weights_ptr + t1 * TOPK + slot, tl.exp(sel - top1) * inv1, mask=t1 < BS)
        rem1 = tl.where(offs_e == expert, NEG, rem1)

    logits2 = tl.where(e_live, acc2, NEG)
    top2 = tl.max(logits2, axis=0)
    inv2 = 1.0 / tl.sum(tl.exp(logits2 - top2), axis=0)
    rem2 = logits2
    for slot in range(0, TOPK):
        sel = tl.max(rem2, axis=0)
        expert = tl.min(tl.where(rem2 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t2 * TOPK + slot, expert, mask=t2 < BS)
        tl.store(weights_ptr + t2 * TOPK + slot, tl.exp(sel - top2) * inv2, mask=t2 < BS)
        rem2 = tl.where(offs_e == expert, NEG, rem2)

    logits3 = tl.where(e_live, acc3, NEG)
    top3 = tl.max(logits3, axis=0)
    inv3 = 1.0 / tl.sum(tl.exp(logits3 - top3), axis=0)
    rem3 = logits3
    for slot in range(0, TOPK):
        sel = tl.max(rem3, axis=0)
        expert = tl.min(tl.where(rem3 == sel, offs_e, BLOCK_E), axis=0)
        tl.store(ids_ptr + t3 * TOPK + slot, expert, mask=t3 < BS)
        tl.store(weights_ptr + t3 * TOPK + slot, tl.exp(sel - top3) * inv3, mask=t3 < BS)
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
    acc1 = tl.zeros((BLOCK_E,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        h_live = offs_h < H
        xb0 = tl.load(x_ptr + t0 * STRIDE_X_B + offs_h * STRIDE_X_H, mask=(t0 < BS) & h_live, other=0.0)
        if BLOCK_M == 2:
            t1 = t0 + 1
            xb1 = tl.load(x_ptr + t1 * STRIDE_X_B + offs_h * STRIDE_X_H, mask=(t1 < BS) & h_live, other=0.0)
        for e in tl.static_range(BLOCK_E):
            e_abs = expert_block * BLOCK_E + e
            w_e = tl.load(w_ptr + e_abs * STRIDE_W_E + offs_h * STRIDE_W_H, mask=(e_abs < E) & h_live, other=0.0)
            sel = offs_e == e_abs
            acc0 += tl.where(sel, tl.sum(w_e.to(tl.float32) * xb0.to(tl.float32), axis=0), 0.0)
            if BLOCK_M == 2:
                acc1 += tl.where(sel, tl.sum(w_e.to(tl.float32) * xb1.to(tl.float32), axis=0), 0.0)
    if HAS_CAP:
        acc0 = (2.0 * tl.sigmoid(2.0 * (acc0 / CAP)) - 1.0) * CAP
        acc1 = (2.0 * tl.sigmoid(2.0 * (acc1 / CAP)) - 1.0) * CAP
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=e_live, other=0.0).to(tl.float32)
        acc0 += bias
        acc1 += bias
    tl.store(logits_ptr + t0 * E + offs_e, acc0, mask=e_live & (t0 < BS))
    if BLOCK_M == 2:
        t1 = t0 + 1
        tl.store(logits_ptr + t1 * E + offs_e, acc1, mask=e_live & (t1 < BS))


@triton.jit
def _logits_dot_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    logits_ptr,
    BS: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    SXB: tl.constexpr,
    SXH: tl.constexpr,
    SWE: tl.constexpr,
    SWH: tl.constexpr,
    SLB: tl.constexpr,
    SLE: tl.constexpr,
    CAP: tl.constexpr,
    HAS_CAP: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NCHUNKS: tl.constexpr,
    PAD_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * PAD_M
    p_a = tl.make_block_ptr(
        x_ptr + m0 * SXB,
        (BS - m0, H),
        (SXB, SXH),
        (0, 0),
        (PAD_M, BLOCK_K),
        (1, 0),
    )
    a = tl.load(p_a, boundary_check=(0, 1))
    n_start = pid_n * BLOCK_N * NCHUNKS
    for c in tl.static_range(NCHUNKS):
        n0 = n_start + c * BLOCK_N
        p_b = tl.make_block_ptr(
            w_ptr,
            (H, E),
            (SWH, SWE),
            (0, n0),
            (BLOCK_K, BLOCK_N),
            (0, 1),
        )
        b = tl.load(p_b, boundary_check=(0, 1))
        acc = tl.dot(a, b, out_dtype=tl.float32)
        if HAS_CAP:
            acc = (2.0 * tl.sigmoid(2.0 * (acc / CAP)) - 1.0) * CAP
        if HAS_BIAS:
            offs_nb = n0 + tl.arange(0, BLOCK_N)
            bias = tl.load(
                bias_ptr + offs_nb, mask=offs_nb < E, other=0.0
            ).to(tl.float32)
            acc += bias[None, :]
        offs_m = m0 + tl.arange(0, PAD_M)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        tl.store(
            logits_ptr + offs_m[:, None] * SLB + offs_n[None, :] * SLE,
            acc,
            mask=(offs_m[:, None] < BS) & (offs_n[None, :] < E),
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


@triton.jit
def _topk_kernel_m(
    logits_ptr,
    weights_ptr,
    ids_ptr,
    BS: tl.constexpr,
    E: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_E: tl.constexpr,
    NEG: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    e_live = offs_e < E
    for m in tl.static_range(BLOCK_M):
        token = pid * BLOCK_M + m
        live = token < BS
        logits = tl.load(logits_ptr + token * E + offs_e, mask=e_live & live, other=NEG)
        logits = tl.where(e_live, logits, NEG)
        row_max = tl.max(logits, axis=0)
        inv = 1.0 / tl.sum(tl.exp(logits - row_max), axis=0)
        remaining = logits
        for slot in range(0, TOPK):
            selected = tl.max(remaining, axis=0)
            expert = tl.min(
                tl.where(remaining == selected, offs_e, BLOCK_E), axis=0
            )
            tl.store(ids_ptr + token * TOPK + slot, expert, mask=live)
            tl.store(
                weights_ptr + token * TOPK + slot,
                tl.exp(selected - row_max) * inv,
                mask=live,
            )
            remaining = tl.where(offs_e == expert, NEG, remaining)


def _next_pow2(value):
    result = 1
    while result < value:
        result *= 2
    return result


_DOT_PAD_M = 32
_dot_arm_broken = False


def _launch_logits_dot(
    x, weights, bias_arg, logits, bs, hidden, experts, cap, has_cap, has_bias
):
    """MMA logits arm for the E > 32 path (bf16 operands, fp32 accumulate).

    Full-K single ``tl.dot`` per static_range N-chunk (fused_moe_gemm proven
    form; runtime-loop dots miscompile on this backend). Returns False when
    the shape is outside the arm's domain or the kernel fails to build, so
    the caller keeps the all-1D tree-reduce kernel as fallback.
    """
    global _dot_arm_broken
    if (
        _dot_arm_broken
        or bs < 2
        or (bs + _DOT_PAD_M - 1) // _DOT_PAD_M > 65535
        or hidden > 4096
        or experts > 8160
        or x.dtype != torch.bfloat16
        or weights.dtype != torch.bfloat16
    ):
        return False
    if bs >= 64 and experts >= 256 and experts % 64 == 0:
        block_n = 64
    else:
        block_n = 32
    sx = x.stride()
    sw = weights.stride()
    try:
        _logits_dot_kernel[
            (
                (bs + _DOT_PAD_M - 1) // _DOT_PAD_M,
                (experts + block_n - 1) // block_n,
            )
        ](
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
            logits.stride(0),
            logits.stride(1),
            cap,
            has_cap,
            has_bias,
            block_n,
            _next_pow2(hidden),
            1,
            _DOT_PAD_M,
            num_warps=1,
            num_stages=1,
        )
        return True
    except Exception:
        _dot_arm_broken = True
        return False


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
    topk_weights = torch.empty(
        (bs, topk), dtype=torch.float32, device=x.device
    )
    topk_ids = torch.empty((bs, topk), dtype=torch.int32, device=x.device)
    if _next_pow2(experts) <= 32:
        block_e = _next_pow2(experts)
        block_h = 32768 // block_e
        if block_h > hidden:
            block_h = _next_pow2(hidden)
        if block_h > 2048:
            block_h = 2048
        if block_h < 16:
            block_h = 16
        _router_kernel[((bs + 3) // 4,)](
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
            num_warps=1,
            num_stages=1,
        )
    else:
        logits = torch.empty(
            (bs, experts), dtype=torch.float32, device=x.device
        )
        bias_arg = correction_bias if has_bias else x
        if not _launch_logits_dot(
            x,
            weights,
            bias_arg,
            logits,
            bs,
            hidden,
            experts,
            cap,
            has_cap,
            has_bias,
        ):
            block_m = 1 if bs == 1 else 2
            block_e = 8 if experts >= 8 else _next_pow2(max(experts, 1))
            if bs == 1 or bs >= 64:
                block_h = 2048
            else:
                block_h = 1024
            if block_h > _next_pow2(hidden):
                block_h = _next_pow2(hidden)
            if block_h < 16:
                block_h = 16
            grid_n = (experts + block_e - 1) // block_e
            if grid_n > 255:
                block_e = _next_pow2((experts + 254) // 255)
                grid_n = (experts + block_e - 1) // block_e
            row_start = 0
            total_rows = (bs + block_m - 1) // block_m
            while row_start < total_rows:
                row_count = min(total_rows - row_start, 65535)
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
                    num_stages=1,
                )
                row_start += row_count

        if bs >= 2048:
            topk_block_m = 4
            _topk_kernel_m[((bs + topk_block_m - 1) // topk_block_m,)](
                logits,
                topk_weights,
                topk_ids,
                bs,
                experts,
                topk,
                _next_pow2(experts),
                neg,
                topk_block_m,
                num_warps=1,
                num_stages=1,
            )
        else:
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
