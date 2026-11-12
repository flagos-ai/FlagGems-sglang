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
def _small_page_owner(
    X, L, Y, T: tl.constexpr, H: tl.constexpr, R: tl.constexpr, C: tl.constexpr
):
    L = L.to(tl.pointer_type(tl.uint32))
    X = X.to(tl.pointer_type(tl.int32))
    Y = Y.to(tl.pointer_type(tl.int32))
    token = tl.arange(0, R)
    head = tl.program_id(0)
    location = tl.load(L + token * 2, token < T, 0).to(tl.uint32)
    offset = (
        ((location >> 7) * H + head) * 128
        + (location & 31) * 4
        + (location >> 5 & 3)
    )
    value = tl.load(X + token * H + head, token < T, 0)
    tl.store(Y + offset, value, token < T)


@triton.jit
def _copy(X, Y, N: tl.constexpr, B: tl.constexpr):
    X = X.to(tl.pointer_type(tl.int32))
    Y = Y.to(tl.pointer_type(tl.int32))
    for start in range(tl.program_id(0), tl.cdiv(N, B), tl.num_programs(0)):
        i = start * B + tl.arange(0, B)
        v = tl.load(X + i, i < N, 0, cache_modifier=".cg")
        tl.store(Y + i, v, i < N, cache_modifier=".cs")


@triton.jit
def _scatter(X, L, Y, T: tl.constexpr, H: tl.constexpr, B: tl.constexpr):
    X = X.to(tl.pointer_type(tl.int32))
    Y = Y.to(tl.pointer_type(tl.int32))
    for start in range(
        tl.program_id(0), tl.cdiv(T * H, B), tl.num_programs(0)
    ):
        i = start * B + tl.arange(0, B)
        t = i // H
        h = i % H
        loc = tl.load(L + t, t < T, 0).to(tl.int64)
        out = ((loc // 128 * H + h) * 32 + loc % 32) * 4 + loc % 128 // 32
        v = tl.load(X + i, i < T * H, 0)
        tl.store(Y + out, v, i < T * H)


@triton.jit
def _block_owned(
    X,
    L,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
):
    X = X.to(tl.pointer_type(tl.int32))
    Y = Y.to(tl.pointer_type(tl.int32))
    base = tl.program_id(0).to(tl.int64) * BC
    j = tl.arange(0, BS)
    location = tl.load(L + j // H, j < T * H, 0).to(tl.int64)
    offset = (
        (location // 128 * H + j % H) * 32 + location % 32
    ) * 4 + location % 128 // 32
    valid = (j < T * H) & (offset >= base) & (offset < base + BC)
    value = tl.load(X + j, valid, 0)
    tl.store(Y + offset, value, valid)


@triton.jit
def _sorted_scatter(
    X, L, Y, T: tl.constexpr, H: tl.constexpr, R: tl.constexpr, C: tl.constexpr
):
    X = X.to(tl.pointer_type(tl.int32))
    Y = Y.to(tl.pointer_type(tl.int32))
    h = tl.program_id(1) * C + tl.arange(0, C)
    block = tl.program_id(0)
    t = block * R + tl.arange(0, R)
    location = tl.load(L + t, t < T, 0).to(tl.uint32)
    offset0 = (
        (location >> 7) * H * 128 + (location & 31) * 4 + (location >> 5 & 3)
    )
    token = t
    target0 = offset0
    valid = (h[:, None] < H) & (token[None, :] < T)
    value = tl.load(X + token[None, :].to(tl.int64) * H + h[:, None], valid, 0)
    tl.store(Y + target0[None, :] + h[:, None] * 128, value, valid)


@triton.jit
def _fused_owner(
    X,
    S,
    L,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
):
    L = L.to(tl.pointer_type(tl.uint32))
    X = X.to(tl.pointer_type(tl.int32))
    S = S.to(tl.pointer_type(tl.int32))
    Y = Y.to(tl.pointer_type(tl.int32))
    first = tl.program_id(0) * G
    h = tl.program_id(1)
    i = tl.arange(0, G * 128)
    page = first + i // 128
    copy_offset = (page * H + h).to(tl.uint32) * 128 + i % 128
    old = tl.load(S + copy_offset, page < P, 0)
    tl.store(Y + copy_offset, old, page < P)
    tl.debug_barrier()
    first_token = tl.arange(0, R)
    prefetched = tl.load(L + first_token * 2, first_token < T, 0).to(tl.uint32)
    for start in range(tl.cdiv(T, R)):
        token = start * R + tl.arange(0, R)
        location = prefetched
        next_token = token + R
        prefetched = tl.load(L + next_token * 2, next_token < T, 0).to(
            tl.uint32
        )
        target_page = location >> 7
        live = (token < T) & (target_page >= first) & (target_page < first + G)
        value = tl.load(X + token.to(tl.uint32) * H + h, live, 0)
        offset = (
            (target_page * H + h) * 128
            + (location & 31) * 4
            + (location >> 5 & 3)
        )
        tl.store(Y + offset, value, live)


def store_sf_interleaved(sf_in, sf_out, loc, page_size):
    if page_size != 128:
        raise ValueError("page_size must be 128")
    t, h, _ = sf_in.shape
    x = sf_in.contiguous()
    locations = loc.contiguous()
    if 512 < t <= 16384 and h <= 32 and (sf_out.numel() <= 536870912):
        src = sf_out.contiguous()
        out = torch.empty_like(sf_out, memory_format=torch.contiguous_format)
        p = sf_out.shape[0]
        if p:
            _fused_owner[triton.cdiv(p, 32), h](
                x,
                src,
                locations,
                out,
                t,
                h,
                p,
                min(triton.next_power_of_2(t), 2048),
                32,
                num_warps=4,
            )
        return out
    if 0 < t <= 512 and h <= 32:
        if sf_out.is_contiguous() and sf_out.storage_offset() % 4 == 0:
            out = sf_out
        else:
            out = sf_out.clone(memory_format=torch.contiguous_format)
        n = out.numel() // 4
        if n:
            if out.numel() <= 536870912 and out.shape[0] <= 8192:
                _small_page_owner[h,](
                    x,
                    locations,
                    out,
                    t,
                    h,
                    triton.next_power_of_2(t),
                    triton.next_power_of_2(h),
                    num_warps=4,
                )
            else:
                _block_owned[triton.cdiv(n, 2048),](
                    x,
                    locations,
                    out,
                    t,
                    h,
                    2048,
                    triton.next_power_of_2(t * h),
                    num_warps=4,
                )
        return out
    src = sf_out.contiguous()
    out = torch.empty_like(sf_out, memory_format=torch.contiguous_format)
    n = src.numel() // 4
    if n:
        _copy[min(triton.cdiv(n, 1024), 4096),](src, out, n, 1024, num_warps=4)
    if t * h:
        if out.numel() <= 536870912 and t <= 131072:
            c = min(triton.next_power_of_2(h), 32)
            _sorted_scatter[min(triton.cdiv(t, 32), 4096), triton.cdiv(h, c)](
                x, locations, out, t, h, 32, c, num_warps=4
            )
        else:
            _scatter[min(triton.cdiv(t * h, 128), 4096),](
                x, locations, out, t, h, 128, num_warps=4
            )
    return out


__all__ = ["store_sf_interleaved"]
