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

"""rmsnorm_hf: HuggingFace LlamaRMSNorm 语义的纯 Triton 实现."""


import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["hidden", "stride_m"])
def _rmsnorm_hf_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    eps,
    hidden,
    stride_m,
    BLOCK: tl.constexpr,
):
    # 单行 kernel: 每 program 一行, 适合并行度足够的 shape.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < hidden

    x = tl.load(x_ptr + row * stride_m + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    # 行内均方: fp32 求和 / hidden, 与 reference 的 mean(-1) 一致
    ms = tl.sum(x * x, axis=0) / hidden
    rstd = tl.math.rsqrt(ms + eps)

    # 先 cast 回输入 dtype (HF 语义), 再以 fp32 与 weight 相乘
    ty = y_ptr.dtype.element_ty
    y = (x * rstd).to(ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * stride_m + cols, (w * y).to(ty), mask=mask)


@triton.jit(do_not_specialize=["hidden", "num_rows", "stride_m"])
def _rmsnorm_hf_rows_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    eps,
    hidden,
    num_rows,
    stride_m,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    EVEN: tl.constexpr,
):
    # 多行 kernel: 每 program 串行 R 行, weight 一次加载行间复用.
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    cmask = cols < hidden

    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    ty = y_ptr.dtype.element_ty
    for i in tl.static_range(R):
        row = pid * R + i
        if EVEN:
            mask = cmask
        else:
            mask = cmask & (row < num_rows)
        base = row * stride_m
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        ms = tl.sum(x * x, axis=0) / hidden
        rstd = tl.math.rsqrt(ms + eps)
        y = (x * rstd).to(ty).to(tl.float32)
        tl.store(y_ptr + base + cols, (w * y).to(ty), mask=mask)


@triton.jit(do_not_specialize=["hidden", "num_rows", "stride_m"])
def _rmsnorm_hf_tile_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    eps,
    hidden,
    num_rows,
    stride_m,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    EVEN: tl.constexpr,
):
    # 2D tile kernel: 每 program 处理 [R, BLOCK] 的 tile, 行维一次 2D load,
    # 行内归约 axis=1, 再 2D store. HF 双 round 语义与 rows kernel 完全一致.
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    cols = tl.arange(0, BLOCK)
    cmask = cols < hidden
    if EVEN:
        rmask = rows < num_rows + R  # 恒真, 仅为让两条分支 IR 形状一致
    else:
        rmask = rows < num_rows
    mask = rmask[:, None] & cmask[None, :]
    off = rows[:, None] * stride_m + cols[None, :]

    x = tl.load(x_ptr + off, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=1) / hidden
    rstd = tl.math.rsqrt(ms + eps)
    ty = y_ptr.dtype.element_ty
    y = (x * rstd[:, None]).to(ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
    tl.store(y_ptr + off, (w[None, :] * y).to(ty), mask=mask)


def _pow2(n):
    """向上取 2 的幂 (纯 Python, 避免每调用进 triton.next_power_of_2)."""
    b = 1
    while b < n:
        b <<= 1
    return b


def _num_warps_single(hidden):
    if hidden <= 512:
        return 4
    if hidden <= 2048:
        return 8
    return 16


def _pick_launch(hidden, num_tokens):
    """按 hidden / M 选择 (kernel 结构, BLOCK, R, num_warps)."""
    block = _pow2(hidden)
    if num_tokens < 16:
        return "single", block, 1, _num_warps_single(hidden)
    if block <= 256:
        return "rows", block, 16, 4
    if block <= 4096:
        if num_tokens >= 4096:
            # bs4096_h4096: tile R=4 nw=4 实测 96us vs rows R=8 106us
            return "tile", block, 4, 4
        return "rows", block, 4 if num_tokens <= 1024 else 8, 1
    # block 8192 (h 4097..8192)
    if num_tokens >= 4096:
        # bs4096_h7168: tile R=2 nw=4 193us vs rows R=8 197us
        return "tile", block, 2, 4
    return "rows", block, 2 if num_tokens <= 1024 else 8, 8


def rmsnorm_hf(input, weight, eps):
    assert (
        input.ndim == 2
    ), "rmsnorm_hf expects a 2D [num_tokens, hidden] input"
    num_tokens, hidden = input.shape
    out = torch.empty_like(input)
    if num_tokens == 0 or hidden == 0:
        return out

    variant, block, r, nw = _pick_launch(hidden, num_tokens)
    if variant == "tile" and num_tokens >= r:
        _rmsnorm_hf_tile_kernel[((num_tokens + r - 1) // r,)](
            input,
            weight,
            out,
            eps,
            hidden,
            num_tokens,
            input.stride(0),
            BLOCK=block,
            R=r,
            EVEN=num_tokens % r == 0,
            num_warps=nw,
        )
    elif r > 1:
        _rmsnorm_hf_rows_kernel[((num_tokens + r - 1) // r,)](
            input,
            weight,
            out,
            eps,
            hidden,
            num_tokens,
            input.stride(0),
            BLOCK=block,
            R=r,
            EVEN=num_tokens % r == 0,
            num_warps=nw,
        )
    else:
        _rmsnorm_hf_kernel[(num_tokens,)](
            input,
            weight,
            out,
            eps,
            hidden,
            input.stride(0),
            BLOCK=block,
            num_warps=nw,
        )
    return out


__all__ = ["rmsnorm_hf"]
