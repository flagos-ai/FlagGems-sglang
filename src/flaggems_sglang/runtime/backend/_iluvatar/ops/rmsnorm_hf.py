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

import torch
import triton
import triton.language as tl

_MAX_ROW_IN_REGS = 8192

_ROW_SEQ_TOKENS = 512

_ALWAYS_ROW_BLOCK_N = 2048


def _configs():
    return (
        triton.Config({"XBLOCK": 1}, num_warps=2),
        triton.Config({"XBLOCK": 1}, num_warps=4),
        triton.Config({"XBLOCK": 1}, num_warps=8),
        triton.Config({"XBLOCK": 1}, num_warps=16),
        triton.Config({"XBLOCK": 1}, num_warps=32),
        triton.Config({"XBLOCK": 2}, num_warps=4),
        triton.Config({"XBLOCK": 2}, num_warps=8),
        triton.Config({"XBLOCK": 2}, num_warps=16),
        triton.Config({"XBLOCK": 4}, num_warps=4),
        triton.Config({"XBLOCK": 4}, num_warps=8),
        triton.Config({"XBLOCK": 4}, num_warps=16),
        triton.Config({"XBLOCK": 8}, num_warps=4),
        triton.Config({"XBLOCK": 8}, num_warps=8),
        triton.Config({"XBLOCK": 16}, num_warps=8),
        triton.Config({"XBLOCK": 16}, num_warps=4),
        triton.Config({"XBLOCK": 32}, num_warps=8),
        triton.Config({"XBLOCK": 32}, num_warps=4),
    )


def _loop_configs():
    configs = []
    for xblock in (1, 2):
        for block_n, warps in (
            (512, 4),
            (1024, 4),
            (1024, 8),
            (2048, 8),
            (2048, 16),
            (4096, 16),
        ):
            configs.append(
                triton.Config(
                    {"XBLOCK": xblock, "BLOCK_N": block_n}, num_warps=warps
                )
            )
    return tuple(configs)


@triton.autotune(configs=list(_configs()), key=["H", "N"])
@triton.jit
def _rmsnorm_hf_row_kernel(
    X,
    W,
    Y,
    stride_x,
    stride_y,
    stride_w,
    H,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    XBLOCK: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    pid = tl.program_id(0)

    rows = pid * XBLOCK + tl.arange(0, XBLOCK)
    cols = tl.max_contiguous(
        tl.multiple_of(tl.arange(0, BLOCK_N), BLOCK_N), BLOCK_N
    )
    row_mask = rows < N
    ptrs = X + rows[:, None] * stride_x + cols[None, :]

    if EVEN_N:
        x = tl.load(
            ptrs,
            mask=row_mask[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        w = tl.load(W + cols * stride_w, eviction_policy="evict_last")
        out_mask = row_mask[:, None]
    else:
        x = tl.load(
            ptrs,
            mask=row_mask[:, None] & (cols[None, :] < H),
            other=0.0,
            eviction_policy="evict_first",
        )
        w = tl.load(
            W + cols * stride_w,
            mask=cols < H,
            other=0.0,
            eviction_policy="evict_last",
        )
        out_mask = row_mask[:, None] & (cols[None, :] < H)
    xf = x.to(tl.float32)

    mean = tl.sum(xf * xf, axis=1) / H
    rstd = tl.rsqrt(mean + eps)

    y = (xf * rstd[:, None]).to(Y.dtype.element_ty)
    tl.store(
        Y + rows[:, None] * stride_y + cols[None, :],
        y * w[None, :],
        mask=out_mask,
        eviction_policy="evict_first",
    )


@triton.autotune(configs=list(_loop_configs()), key=["H", "N"])
@triton.jit
def _rmsnorm_hf_loop_kernel(
    X,
    W,
    Y,
    stride_x,
    stride_y,
    stride_w,
    H,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * XBLOCK + tl.arange(0, XBLOCK)
    row_mask = rows < N
    cols = tl.arange(0, BLOCK_N)
    x_base = X + rows[:, None] * stride_x + cols[None, :]
    y_base = Y + rows[:, None] * stride_y + cols[None, :]

    acc = tl.zeros((XBLOCK, BLOCK_N), dtype=tl.float32)
    for off in range(0, H, BLOCK_N):
        idx = off + cols
        if off + BLOCK_N <= H:
            x = tl.load(x_base + off, mask=row_mask[:, None], other=0.0).to(
                tl.float32
            )
        else:
            x = tl.load(
                x_base + off,
                mask=row_mask[:, None] & (idx[None, :] < H),
                other=0.0,
            ).to(tl.float32)
        acc += x * x
    mean = tl.sum(acc, axis=1) / H
    rstd = tl.rsqrt(mean + eps)

    out_dtype = Y.dtype.element_ty
    for off in range(0, H, BLOCK_N):
        idx = off + cols
        if off + BLOCK_N <= H:
            x = tl.load(x_base + off, mask=row_mask[:, None], other=0.0).to(
                tl.float32
            )
            w = tl.load(W + idx * stride_w)
            y = (x * rstd[:, None]).to(out_dtype)
            tl.store(y_base + off, y * w[None, :], mask=row_mask[:, None])
        else:
            x = tl.load(
                x_base + off,
                mask=row_mask[:, None] & (idx[None, :] < H),
                other=0.0,
            ).to(tl.float32)
            w = tl.load(W + idx * stride_w, mask=idx < H, other=0.0)
            y = (x * rstd[:, None]).to(out_dtype)
            tl.store(
                y_base + off,
                y * w[None, :],
                mask=row_mask[:, None] & (idx[None, :] < H),
            )


def rmsnorm_hf(input, weight, eps):
    num_tokens, hidden = input.shape
    out = torch.empty_like(input)
    sx, sy, sw = input.stride(0), out.stride(0), weight.stride(0)

    block_n = triton.next_power_of_2(hidden)
    use_row = (
        block_n <= _ALWAYS_ROW_BLOCK_N
        or block_n > _MAX_ROW_IN_REGS
        or num_tokens < _ROW_SEQ_TOKENS
    )
    if use_row:

        def grid(meta):
            return (triton.cdiv(num_tokens, meta["XBLOCK"]),)

        _rmsnorm_hf_row_kernel[grid](
            input,
            weight,
            out,
            sx,
            sy,
            sw,
            hidden,
            num_tokens,
            eps,
            BLOCK_N=block_n,
            EVEN_N=(hidden == block_n),
        )
    else:
        _rmsnorm_hf_loop_kernel[(triton.cdiv(num_tokens, 1),)](
            input,
            weight,
            out,
            sx,
            sy,
            sw,
            hidden,
            num_tokens,
            eps,
        )
    return out


__all__ = ["rmsnorm_hf"]
