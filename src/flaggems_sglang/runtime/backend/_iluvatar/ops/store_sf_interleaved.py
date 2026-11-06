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
def _scatter(
    X: tl.pointer_type(tl.int32),
    L: tl.pointer_type(tl.int64),
    Y: tl.pointer_type(tl.int32),
    T: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
):
    if T * H <= B * 4096:
        start = tl.program_id(0)
        i = start * B + tl.arange(0, B)
        t = i // H
        h = i % H
        if (start + 1) * B <= T * H:
            loc = tl.load(L + t).to(tl.int64)
            out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
            v = tl.load(X + i)
            tl.store(Y + out, v)
        else:
            loc = tl.load(L + t, t < T, 0).to(tl.int64)
            out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
            v = tl.load(X + i, i < T * H, 0)
            tl.store(Y + out, v, i < T * H)
    else:
        for start in range(
            tl.program_id(0), tl.cdiv(T * H, B), tl.num_programs(0)
        ):
            i = start * B + tl.arange(0, B)
            t = i // H
            h = i % H
            if (start + 1) * B <= T * H:
                loc = tl.load(L + t).to(tl.int64)
                out = (
                    ((loc >> 7) * H + h) * 128
                    + (loc & 31) * 4
                    + (loc >> 5 & 3)
                )
                v = tl.load(X + i)
                tl.store(Y + out, v)
            else:
                loc = tl.load(L + t, t < T, 0).to(tl.int64)
                out = (
                    ((loc >> 7) * H + h) * 128
                    + (loc & 31) * 4
                    + (loc >> 5 & 3)
                )
                v = tl.load(X + i, i < T * H, 0)
                tl.store(Y + out, v, i < T * H)


@triton.jit(do_not_specialize=["X", "L", "Y"])
def _small(
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
    value = tl.load(X + token * H + head, head < H, 0)
    tl.store(Y + offset, value, head < H)


@triton.jit(do_not_specialize=["X", "L", "Y"])
def _full(
    X: tl.pointer_type(tl.int32),
    L: tl.pointer_type(tl.int64),
    Y: tl.pointer_type(tl.int32),
    H: tl.constexpr,
    B: tl.constexpr,
):
    start = tl.program_id(0)
    i = start * B + tl.arange(0, B)
    t = i // H
    h = i % H
    loc = tl.load(L + t).to(tl.int64)
    out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
    v = tl.load(X + i)
    tl.store(Y + out, v)


def store_sf_interleaved(sf_in, sf_out, loc, page_size):
    if page_size != 128:
        raise ValueError("page_size must be 128")
    x = sf_in.contiguous()
    locations = loc.contiguous()
    if sf_out.is_contiguous() and sf_out.storage_offset() % 4 == 0:
        out = sf_out
    else:
        out = sf_out.clone(memory_format=torch.contiguous_format)
    y = out
    t, h, _ = sf_in.shape
    if t * h:
        if t == 1 and 0 < h <= 32 and (h & h - 1 == 0):
            _small[1,](x, locations, y, h, num_warps=4)
        else:
            b = 512 if 2048 <= t * h <= 4096 else 128
            if t * h % b == 0 and t * h <= b * 4096:
                _full[min(triton.cdiv(t * h, b), 4096),](
                    x, locations, y, h, b, num_warps=4
                )
            else:
                _scatter[min(triton.cdiv(t * h, b), 4096),](
                    x, locations, y, t, h, b, num_warps=4
                )
    return out


__all__ = ["store_sf_interleaved"]
