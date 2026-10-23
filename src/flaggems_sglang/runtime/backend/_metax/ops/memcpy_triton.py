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
# implied. See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
import torch
import triton
import triton.language as tl


@triton.jit
def _copy(
    D,
    S,
    OFF,
    SZ,
    N: tl.constexpr,
    W: tl.constexpr,
    SH: tl.constexpr,
    DS: tl.constexpr,
    SS: tl.constexpr,
    FROM: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    PERSIST: tl.constexpr,
):
    pid = tl.program_id(0) + tl.program_id(1) * 32768
    offset = tl.load(OFF).to(tl.int64)
    size = tl.load(SZ).to(tl.int64)
    tile = pid
    i = (tile * B + tl.arange(0, B)).to(tl.int64)
    row = i // W
    tail = i % W
    sr = row + offset if FROM else row
    dr = row if FROM else row + offset
    sa = sr * SS[0]
    da = dr * DS[0]
    for j in tl.static_range(len(SH) - 1, 0, -1):
        c = tail % SH[j]
        tail = tail // SH[j]
        sa += c * SS[j]
        da += c * DS[j]
    valid = (i < N) & (row < size)
    value = tl.load(S + sa, valid, other=0)
    tl.store(D + da, value, valid)


@triton.jit
def _linear_fast(
    D,
    S,
    OFF,
    SZ,
    N: tl.constexpr,
    W: tl.constexpr,
    FROM: tl.constexpr,
    B: tl.constexpr,
    I32: tl.constexpr,
):
    dt: tl.constexpr = tl.int32 if I32 else tl.int64
    pid = tl.program_id(0).to(dt) + tl.program_id(1).to(dt) * 32768
    i = pid * B + tl.arange(0, B).to(dt)
    off = tl.load(OFF).to(dt) * W
    size = tl.load(SZ).to(dt) * W
    source = i + off if FROM else i
    dest = i if FROM else i + off
    value = tl.load(S + source, (i < N) & (i < size), other=0)
    tl.store(D + dest, value, (i < N) & (i < size))


def _frozen_fast(dst, src, dim, offset, sz, offset_src):
    assert dim == 0
    w = 1
    for size in dst.shape[1:]:
        w *= size
    n = min(dst.shape[0], src.shape[0]) * w
    if n:
        if dst.is_contiguous() and src.is_contiguous():
            b = 2048
            t = triton.cdiv(n, b)
            narrow = max(dst.numel(), src.numel()) + b < 2**31
            _linear_fast[(min(t, 32768), triton.cdiv(t, 32768))](
                dst,
                src,
                offset,
                sz,
                n,
                w,
                offset_src,
                b,
                narrow,
                num_warps=4,
                num_stages=1,
            )
            return dst
        b = 1024
        t = triton.cdiv(n, b)
        _copy[(min(t, 32768), triton.cdiv(t, 32768))](
            dst,
            src,
            offset,
            sz,
            n,
            w,
            tuple(dst.shape),
            tuple(dst.stride()),
            tuple(src.stride()),
            offset_src,
            b,
            t,
            False,
            num_warps=4,
            num_stages=1,
        )
    return dst


@triton.jit
def _alias_strided(
    D,
    S,
    O,
    OFF,
    SZ,
    N: tl.constexpr,
    W: tl.constexpr,
    SH: tl.constexpr,
    DS: tl.constexpr,
    SS: tl.constexpr,
    OS: tl.constexpr,
    FROM_SRC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = (
        tl.cast(tl.program_id(0), tl.int64)
        + tl.cast(tl.program_id(1), tl.int64) * 32768
    ) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    row = i // W
    tail = i % W
    off = tl.load(OFF).to(tl.int64)
    size = tl.load(SZ).to(tl.int64)
    if FROM_SRC:
        take = row < size
        sr = row + off
    else:
        take = (row >= off) & (row < off + size)
        sr = row - off
    da = row * DS[0]
    sa = sr * SS[0]
    oa = row * OS[0]
    for axis in tl.static_range(len(SH) - 1, 0, -1):
        coord = tail % SH[axis]
        tail = tail // SH[axis]
        da += coord * DS[axis]
        sa += coord * SS[axis]
        oa += coord * OS[axis]
    old = tl.load(D + da, (i < N) & ~take, other=0)
    new = tl.load(S + sa, (i < N) & take, other=0)
    tl.store(O + oa, tl.where(take, new, old), i < N)


def _shared_storage_copy(dst, src, offset, sz, offset_src):
    out = torch.empty(tuple(dst.shape), device=dst.device, dtype=dst.dtype)
    width = 1
    for size in dst.shape[1:]:
        width *= size
    if dst.numel():
        tiles = triton.cdiv(dst.numel(), 256)
        _alias_strided[(min(tiles, 32768), triton.cdiv(tiles, 32768))](
            dst,
            src,
            out,
            offset,
            sz,
            dst.numel(),
            max(width, 1),
            tuple(dst.shape),
            tuple(dst.stride()),
            tuple(src.stride()),
            tuple(out.stride()),
            offset_src,
            256,
            num_warps=4,
        )
    return out


def _scalar_dispatch(dst, src, dim, offset, sz, offset_src):
    assert dim == 0
    if (
        dst.numel()
        and src.numel()
        and dst.untyped_storage().data_ptr()
        == src.untyped_storage().data_ptr()
    ):
        return _shared_storage_copy(dst, src, offset, sz, offset_src)
    return _frozen_fast(dst, src, dim, offset, sz, offset_src)


@triton.jit
def _packed_window(
    D,
    S,
    OFF,
    SZ,
    N: tl.constexpr,
    W: tl.constexpr,
    FROM: tl.constexpr,
    B: tl.constexpr,
    GRID: tl.constexpr,
    I32: tl.constexpr,
):
    D = D.to(tl.pointer_type(tl.uint64))
    S = S.to(tl.pointer_type(tl.uint64))
    dt: tl.constexpr = tl.int32 if I32 else tl.int64
    off = tl.load(OFF).to(dt) * W
    size = tl.load(SZ).to(dt) * W
    tile = tl.program_id(0).to(dt)
    while tile * B < size:
        i = tile * B + tl.arange(0, B).to(dt)
        source = i + off if FROM else i
        dest = i if FROM else i + off
        mask = (i < N) & (i < size)
        value = tl.load(S + source, mask, other=0)
        tl.store(D + dest, value, mask)
        tile += GRID


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    assert dim == 0
    if (
        dst.numel()
        and src.numel()
        and dst.untyped_storage().data_ptr()
        == src.untyped_storage().data_ptr()
    ):
        return _shared_storage_copy(dst, src, offset, sz, offset_src)
    width = 1
    for size in dst.shape[1:]:
        width *= size
    element = dst.element_size()
    pack = 8 // element if element in (1, 2, 4) else 1
    n = min(dst.shape[0], src.shape[0]) * width
    aligned = (
        dst.dtype == src.dtype
        and pack > 1
        and width % pack == 0
        and dst.is_contiguous()
        and src.is_contiguous()
        and dst.storage_offset() % pack == 0
        and src.storage_offset() % pack == 0
    )
    if aligned and n // pack >= 1024:
        words = n // pack
        grid = min(triton.cdiv(words, 512), 256)
        narrow = max(dst.numel(), src.numel()) // pack + 512 < 2**31
        _packed_window[(grid,)](
            dst,
            src,
            offset,
            sz,
            words,
            width // pack,
            offset_src,
            512,
            grid,
            narrow,
            num_warps=4,
            num_stages=1,
        )
        return dst
    return _frozen_fast(dst, src, dim, offset, sz, offset_src)


__all__ = ["memcpy_triton"]
