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
def _strided(
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
    i = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
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


@triton.jit
def _linear(
    D,
    S,
    O,
    OFF,
    SZ,
    N: tl.constexpr,
    W: tl.constexpr,
    FROM_SRC: tl.constexpr,
    I32: tl.constexpr,
    BLOCK: tl.constexpr,
):
    dt: tl.constexpr = tl.int32 if I32 else tl.int64
    i = (
        tl.program_id(0).to(dt) + tl.program_id(1).to(dt) * 32768
    ) * BLOCK + tl.arange(0, BLOCK).to(dt)
    off = tl.load(OFF).to(dt)
    size = tl.load(SZ).to(dt)
    row = i // W
    if FROM_SRC:
        take = row < size
        source = i + off * W
    else:
        take = (row >= off) & (row < off + size)
        source = i - off * W
    first = (tl.program_id(0).to(dt) + tl.program_id(1).to(dt) * 32768) * BLOCK
    if FROM_SRC:
        active = first < size * W
    else:
        active = (first < (off + size) * W) & (first + BLOCK > off * W)
    if active:
        old = tl.load(D + i, (i < N) & ~take, other=0)
        new = tl.load(S + source, (i < N) & take, other=0)
        tl.store(O + i, tl.where(take, new, old), i < N)


def _frozen_copy(dst, src, dim, offset, sz, offset_src):
    assert dim == 0
    out = dst
    width = 1
    for size in dst.shape[1:]:
        width *= size
    if dst.is_contiguous() and src.is_contiguous():
        tiles = triton.cdiv(dst.numel(), 1024)
        narrow = max(dst.numel(), src.numel()) + 1024 < 2147483648
        _linear[
            (max(1, min(32768, tiles)), max(1, triton.cdiv(tiles, 32768)))
        ](
            dst,
            src,
            out,
            offset,
            sz,
            dst.numel(),
            max(width, 1),
            offset_src,
            narrow,
            1024,
            num_warps=4,
        )
    else:
        _strided[(max(1, triton.cdiv(dst.numel(), 256)),)](
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


@triton.jit
def _window_copy(
    D,
    S,
    OFF,
    SZ,
    DN: tl.constexpr,
    W: tl.constexpr,
    FROM_SRC: tl.constexpr,
    OWNED: tl.constexpr,
    RAW: tl.constexpr,
    I32: tl.constexpr,
    B: tl.constexpr,
):
    dt: tl.constexpr = tl.int32 if I32 else tl.int64
    offset = tl.load(OFF).to(dt) * W
    length = tl.load(SZ).to(dt) * W
    begin = 0 if FROM_SRC else offset
    delta = offset if FROM_SRC else -offset
    base = begin // B * B
    end = begin + length
    count = (end - base + B - 1) // B
    if length > 0:
        if RAW:
            bits: tl.constexpr = D.dtype.element_ty.primitive_bitwidth
            raw: tl.constexpr = (
                tl.uint8
                if bits <= 8
                else (
                    tl.uint16
                    if bits == 16
                    else tl.uint32 if bits == 32 else tl.uint64
                )
            )
            dp = D.to(tl.pointer_type(raw))
            sp = S.to(tl.pointer_type(raw))
        else:
            dp = D
            sp = S
        for tile in range(tl.program_id(0), count, tl.num_programs(0)):
            i = base + tl.cast(tile, dt) * B + tl.arange(0, B).to(dt)
            valid = i < DN
            take = (i >= begin) & (i < end) & valid
            new = tl.load(sp + i + delta, take, other=0)
            if OWNED:
                old = tl.load(dp + i, valid & ~take, other=0)
                tl.store(dp + i, tl.where(take, new, old), valid)
            else:
                tl.store(dp + i, new, take)


def _run_window(
    dst,
    src,
    offset,
    sz,
    offset_src,
    owned,
    raw,
    grid_cap=256,
    force_wide=False,
):
    width = 1
    for size in dst.shape[1:]:
        width *= size
    block = 1024
    n = dst.numel()
    narrow = (not force_wide) and max(dst.numel(), src.numel()) + block < 2**31
    _window_copy[(min(triton.cdiv(n, block), grid_cap),)](
        dst,
        src,
        offset,
        sz,
        n,
        width,
        offset_src,
        owned,
        raw,
        narrow,
        block,
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


def _fallback(dst, src, dim, offset, sz, offset_src):
    assert dim == 0
    if dst.numel() == 0 or src.numel() == 0:
        return dst
    if dst.untyped_storage().data_ptr() == src.untyped_storage().data_ptr():
        return _shared_storage_copy(dst, src, offset, sz, offset_src)
    if dst.is_contiguous() and src.is_contiguous():
        return _run_window(dst, src, offset, sz, offset_src, True, True)
    return _frozen_copy(dst, src, dim, offset, sz, offset_src)


@triton.jit
def _word_window(
    D,
    S,
    OFF,
    SZ,
    DN: tl.constexpr,
    W: tl.constexpr,
    FROM_SRC: tl.constexpr,
    OWNED: tl.constexpr,
    RAW: tl.constexpr,
    I32: tl.constexpr,
    B: tl.constexpr,
):
    dt: tl.constexpr = tl.int32 if I32 else tl.int64
    offset = tl.load(OFF).to(dt) * W
    length = tl.load(SZ).to(dt) * W
    begin = 0 if FROM_SRC else offset
    delta = offset if FROM_SRC else -offset
    base = begin // B * B
    end = begin + length
    count = (end - base + B - 1) // B
    if length > 0:
        dp = D.to(tl.pointer_type(tl.uint64))
        sp = S.to(tl.pointer_type(tl.uint64))
        for tile in range(tl.program_id(0), count, tl.num_programs(0)):
            i = base + tl.cast(tile, dt) * B + tl.arange(0, B).to(dt)
            first = base + tl.cast(tile, dt) * B
            if (first >= begin) & (first + B <= end) & (first + B <= DN):
                value = tl.load(sp + i + delta)
                tl.store(dp + i, value)
            else:
                valid = i < DN
                take = (i >= begin) & (i < end) & valid
                new = tl.load(sp + i + delta, take, other=0)
                if OWNED:
                    old = tl.load(dp + i, valid & ~take, other=0)
                    tl.store(dp + i, tl.where(take, new, old), valid)
                else:
                    tl.store(dp + i, new, take)


def memcpy_triton(dst, src, dim, offset, sz, offset_src):
    assert dim == 0
    size = dst.element_size()
    pack = 8 // size if size in (1, 2, 4) else 1
    width = 1
    for extent in dst.shape[1:]:
        width *= extent
    if (
        not dst.numel()
        or not src.numel()
        or pack == 1
        or dst.dtype != src.dtype
        or width % pack
        or not dst.is_contiguous()
        or not src.is_contiguous()
        or dst.storage_offset() % pack
        or src.storage_offset() % pack
        or dst.untyped_storage().data_ptr() == src.untyped_storage().data_ptr()
    ):
        return _fallback(dst, src, dim, offset, sz, offset_src)
    n = dst.numel() // pack
    w = width // pack
    block = 1024
    narrow = max(dst.numel(), src.numel()) // pack + block < 2**31
    _word_window[(min(triton.cdiv(n, block), 256),)](
        dst,
        src,
        offset,
        sz,
        n,
        w,
        offset_src,
        True,
        True,
        narrow,
        block,
        num_warps=4,
        num_stages=1,
    )
    return dst


__all__ = ["memcpy_triton"]
