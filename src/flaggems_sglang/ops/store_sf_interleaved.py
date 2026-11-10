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


@triton.jit(do_not_specialize=["X", "L", "Y"])
def _scatter_bounded(
    X, L, Y, T: tl.constexpr, H: tl.constexpr, B: tl.constexpr
):
    for start in range(
        tl.program_id(0), tl.cdiv(T * H, B), tl.num_programs(0)
    ):
        i = start * B + tl.arange(0, B)
        t = i // H
        h = i % H
        if (start + 1) * B <= T * H:
            loc = tl.load(L + t).to(tl.int64)
            out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
            v = tl.load(X + i, cache_modifier=".cg")
            tl.store(Y + out, v, cache_modifier=".cs")
        else:
            loc = tl.load(L + t, t < T, 0).to(tl.int64)
            out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
            v = tl.load(X + i, i < T * H, 0, cache_modifier=".cg")
            tl.store(Y + out, v, i < T * H, cache_modifier=".cs")


def _bounded(sf_in, sf_out, loc, page_size):
    if page_size != 128:
        raise ValueError("page_size must be 128")
    x = sf_in.contiguous().view(torch.int32)
    locations = loc.contiguous()
    if sf_out.is_contiguous() and sf_out.storage_offset() % 4 == 0:
        out = sf_out
    else:
        out = sf_out.clone(memory_format=torch.contiguous_format)
    y = out.view(torch.int32)
    t, h, _ = sf_in.shape
    if t * h:
        _scatter_bounded.run(
            x,
            locations,
            y,
            t,
            h,
            128,
            num_warps=4,
            grid=(min(triton.cdiv(t * h, 128), 4096),),
            warmup=False,
        )
    return out


@triton.jit(do_not_specialize=["X", "L", "Y"])
def _scatter(X, L, Y, T: tl.constexpr, H: tl.constexpr, B: tl.constexpr):
    start = tl.program_id(0)
    i = start * B + tl.arange(0, B)
    t = i // H
    h = i % H
    if (start + 1) * B <= T * H:
        loc = tl.load(L + t).to(tl.int64)
        out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
        v = tl.load(X + i, cache_modifier=".cg")
        tl.store(Y + out, v, cache_modifier=".cs")
    else:
        loc = tl.load(L + t, t < T, 0).to(tl.int64)
        out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
        v = tl.load(X + i, i < T * H, 0, cache_modifier=".cg")
        tl.store(Y + out, v, i < T * H, cache_modifier=".cs")


@triton.jit(do_not_specialize=["X", "L", "Y"])
def _single_token(
    X: tl.pointer_type(tl.int32),
    L: tl.pointer_type(tl.int64),
    Y: tl.pointer_type(tl.int32),
    H: tl.constexpr,
):
    head = tl.arange(0, H)
    location = tl.load(L).to(tl.int64)
    offset = (
        ((location >> 7) * H + head) * 128
        + (location & 31) * 4
        + (location >> 5 & 3)
    )
    value = tl.load(X + head, cache_modifier=".cg")
    tl.store(Y + offset, value, cache_modifier=".cs")


@triton.jit(do_not_specialize=["X", "L", "Y"])
def _token_owned(
    X: tl.pointer_type(tl.int32),
    L: tl.pointer_type(tl.uint32),
    Y: tl.pointer_type(tl.int32),
    H: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.arange(0, H)
    location = tl.load(L + token * 2).to(tl.int32)
    offset = (
        ((location >> 7) * H + head) * 128
        + (location & 31) * 4
        + (location >> 5 & 3)
    )
    value = tl.load(X + token * H + head, cache_modifier=".cg")
    tl.store(Y + offset, value, cache_modifier=".cs")


@triton.jit(do_not_specialize=["X", "L", "Y"])
def _token_owned_wide(
    X: tl.pointer_type(tl.int32),
    L: tl.pointer_type(tl.int64),
    Y: tl.pointer_type(tl.int32),
    H: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.arange(0, H)
    location = tl.load(L + token).to(tl.int64)
    offset = (
        ((location >> 7) * H + head) * 128
        + (location & 31) * 4
        + (location >> 5 & 3)
    )
    value = tl.load(X + token * H + head, cache_modifier=".cg")
    tl.store(Y + offset, value, cache_modifier=".cs")


def store_sf_interleaved(sf_in, sf_out, loc, page_size):
    t, h, _ = sf_in.shape
    if t * h > 524288:
        return _bounded(sf_in, sf_out, loc, page_size)
    if page_size != 128:
        raise ValueError("page_size must be 128")
    tiny_t, tiny_h, _ = sf_in.shape
    if (
        1 < tiny_t <= 64
        and 0 < tiny_h <= 32
        and (tiny_h & tiny_h - 1 == 0)
        and (sf_in.storage_offset() % 4 == 0)
    ) and sf_out.numel() <= 8589934592:
        tiny_x = sf_in.contiguous()
        tiny_l = loc.contiguous()
        if sf_out.is_contiguous() and sf_out.storage_offset() % 4 == 0:
            tiny_out = sf_out
        else:
            tiny_out = sf_out.clone(memory_format=torch.contiguous_format)
        if tiny_t < 64:
            _token_owned.run(
                tiny_x,
                tiny_l,
                tiny_out,
                tiny_h,
                num_warps=4,
                grid=(tiny_t,),
                warmup=False,
            )
        else:
            _token_owned_wide.run(
                tiny_x,
                tiny_l,
                tiny_out,
                tiny_h,
                num_warps=1,
                grid=(tiny_t,),
                warmup=False,
            )
        return tiny_out
    x = sf_in.contiguous().view(torch.int32)
    locations = loc.contiguous()
    if sf_out.is_contiguous() and sf_out.storage_offset() % 4 == 0:
        out = sf_out
    else:
        out = sf_out.clone(memory_format=torch.contiguous_format)
    y = out.view(torch.int32)
    if t == 1 and 0 < h <= 32 and (h & h - 1 == 0):
        _single_token.run(
            x, locations, y, h, num_warps=4, grid=(1,), warmup=False
        )
        return out
    if t * h:
        _scatter.run(
            x,
            locations,
            y,
            t,
            h,
            128,
            num_warps=4,
            grid=(triton.cdiv(t * h, 128),),
            warmup=False,
        )
    return out


__all__ = ["store_sf_interleaved"]
