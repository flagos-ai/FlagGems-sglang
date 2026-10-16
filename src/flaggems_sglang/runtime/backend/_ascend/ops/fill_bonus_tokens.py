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
def _scalar(src, lens, out, STRIDE: tl.constexpr):
    i = tl.program_id(0)
    length = tl.load(lens + i).to(tl.int64)
    tl.store(out + i, tl.load(src + i.to(tl.int64) * STRIDE + length - 1))


@triton.jit
def _block(
    src, lens, out, STRIDE: tl.constexpr, N: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    j = tl.minimum(i, N - 1)
    length = tl.load(lens + j).to(tl.int64)
    value = tl.load(src + j.to(tl.int64) * STRIDE + length - 1)
    tl.store(out + i, value, i < N)


@triton.jit
def _persistent(src, lens, out, STRIDE: tl.constexpr, N: tl.constexpr):
    block = tl.program_id(0)
    while block < tl.cdiv(N, 8):
        i = block * 8 + tl.arange(0, 8)
        j = tl.minimum(i, N - 1)
        length = tl.load(lens + j).to(tl.int64)
        value = tl.load(src + j.to(tl.int64) * STRIDE + length - 1)
        tl.store(out + i, value, i < N)
        block += 40


@triton.jit
def _bytes(
    src,
    lens,
    out,
    STRIDE: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
):
    for block in range(tl.program_id(0), tl.cdiv(N, B), 40):
        r = tl.arange(0, B)
        row = block * B + r
        byte = tl.arange(0, 8)
        length = tl.load(lens + row, row < N, 1).to(tl.int32)
        offsets = tl.arange(0, K)
        global_offsets = block.to(tl.int64) * B * STRIDE * 8 + offsets
        values = tl.load(
            src + global_offsets,
            (offsets < B * STRIDE * 8) & (global_offsets < N * STRIDE * 8),
            0,
        )
        index = (r[:, None] * STRIDE + length[:, None] - 1) * 8 + byte[None, :]
        selected = tl.gather(values, tl.reshape(index, (B * 8,)), 0)
        w = tl.arange(0, B * 8)
        tl.store(
            out + block.to(tl.int64) * B * 8 + w,
            selected,
            block * B + w // 8 < N,
        )


@triton.jit
def _full_bytes(
    src,
    lens,
    out,
    STRIDE: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
):
    for block in range(tl.program_id(0), tl.cdiv(N, B), 40):
        r = tl.arange(0, B)
        row = block * B + r
        byte = tl.arange(0, 8)
        length = tl.load(lens + row).to(tl.int32)
        offsets = tl.arange(0, K)
        global_offsets = block.to(tl.int64) * B * STRIDE * 8 + offsets
        values = tl.load(src + global_offsets)
        index = (r[:, None] * STRIDE + length[:, None] - 1) * 8 + byte[None, :]
        selected = tl.gather(values, tl.reshape(index, (B * 8,)), 0)
        w = tl.arange(0, B * 8)
        tl.store(out + block.to(tl.int64) * B * 8 + w, selected)


@triton.jit
def _single_bytes(
    src,
    lens,
    out,
    STRIDE: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
):
    block = tl.program_id(0)
    r = tl.arange(0, B)
    row = block * B + r
    byte = tl.arange(0, 8)
    length = tl.load(lens + row).to(tl.int32)
    offsets = tl.arange(0, K)
    global_offsets = block.to(tl.int64) * B * STRIDE * 8 + offsets
    values = tl.load(src + global_offsets)
    index = (r[:, None] * STRIDE + length[:, None] - 1) * 8 + byte[None, :]
    selected = tl.gather(values, tl.reshape(index, (B * 8,)), 0)
    w = tl.arange(0, B * 8)
    tl.store(out + block.to(tl.int64) * B * 8 + w, selected)


def fill_bonus_tokens(accept_tokens, accept_lens, bonus_tokens, accept_stride):
    n = accept_lens.shape[0]
    out = (
        bonus_tokens
        if bonus_tokens.shape == (n,) and bonus_tokens.is_contiguous()
        else torch.empty(
            n, dtype=bonus_tokens.dtype, device=accept_tokens.device
        )
    )
    if n == 0:
        return out
    src = accept_tokens.contiguous()
    lens = accept_lens.contiguous()
    stride = int(accept_stride)
    use_bytes = (
        n >= 8
        and stride <= 128
        and src.dtype == torch.int64
        and out.dtype == torch.int64
    )
    if n <= 8 and not use_bytes:
        _scalar[(n,)](src, lens, out, stride, num_warps=1)
    elif use_bytes:
        block = (
            min(16, triton.next_power_of_2(n))
            if n <= 128
            else 64 if n <= 2048 else 128
        )
        if n > 2048 and stride <= 16:
            block = 256
        block = min(
            block,
            max(
                1,
                (2048 if n > 2048 and stride <= 16 else 1024)
                // triton.next_power_of_2(stride),
            ),
        )
        full = n % block == 0 and stride & (stride - 1) == 0
        kernel = (
            _single_bytes
            if full and n >= 32
            else _full_bytes if full else _bytes
        )
        grid = (
            triton.cdiv(n, block)
            if full and n >= 32
            else min(40, triton.cdiv(n, block))
        )
        kernel[(grid,)](
            src.view(torch.int8),
            lens,
            out.view(torch.int8),
            stride,
            n,
            block,
            triton.next_power_of_2(block * stride * 8),
            num_warps=1,
        )
    elif n <= 2048:
        block = 4 if n <= 128 else 8 if n <= 256 else 16
        _block[(triton.cdiv(n, block),)](
            src, lens, out, stride, n, block, num_warps=1
        )
    else:
        _persistent[(40,)](src, lens, out, stride, n, num_warps=1)
    return out


__all__ = ["fill_bonus_tokens"]
