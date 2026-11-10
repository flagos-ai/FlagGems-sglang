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
def _alpha_scatter(
    sf_in,
    sf_out,
    loc,
    T: tl.constexpr,
    H: tl.constexpr,
    PAGE: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
):
    pid = tl.program_id(0)
    tok = pid * BT + tl.arange(0, BT)
    tmask = tok < T
    slot = tl.load(loc + tok, mask=tmask, other=0).to(tl.int32)
    base = slot // PAGE * (H * 512) + slot % 32 * 16 + slot // 32 % 4 * 4
    ibase = tok * (H * 4)
    j = tl.arange(0, 4)
    for h0 in tl.static_range(0, H, BH):
        h = h0 + tl.arange(0, BH)
        m = tmask[:, None, None] & (h < H)[None, :, None]
        v = tl.load(
            sf_in
            + (ibase[:, None, None] + h[None, :, None] * 4 + j[None, None, :]),
            mask=m,
            other=0,
        )
        tl.store(
            sf_out
            + (
                base[:, None, None] + h[None, :, None] * 512 + j[None, None, :]
            ),
            v,
            mask=m,
        )


@triton.jit
def _scatter32(
    X: tl.pointer_type(tl.int32),
    L: tl.pointer_type(tl.int64),
    Y: tl.pointer_type(tl.int32),
    T: tl.constexpr,
    H: tl.constexpr,
):
    B: tl.constexpr = 128
    for start in range(
        tl.program_id(0), tl.cdiv(T * H, B), tl.num_programs(0)
    ):
        i = start * B + tl.arange(0, B)
        t = i // H
        h = i % H
        if (start + 1) * B <= T * H:
            loc = tl.load(L + t).to(tl.int32)
            out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
            v = tl.load(X + i)
            tl.store(Y + out, v)
        else:
            loc = tl.load(L + t, t < T, 0).to(tl.int32)
            out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
            v = tl.load(X + i, i < T * H, 0)
            tl.store(Y + out, v, i < T * H)


@triton.jit
def _scatter64(
    X: tl.pointer_type(tl.int32),
    L: tl.pointer_type(tl.int64),
    Y: tl.pointer_type(tl.int32),
    T: tl.constexpr,
    H: tl.constexpr,
):
    B: tl.constexpr = 128
    for start in range(
        tl.program_id(0), tl.cdiv(T * H, B), tl.num_programs(0)
    ):
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


@triton.jit
def _fill(M: tl.pointer_type(tl.int16), N: tl.constexpr):
    i = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(M + i, -1, i < N)


@triton.jit
def _index(
    L: tl.pointer_type(tl.uint32),
    M: tl.pointer_type(tl.int16),
    T: tl.constexpr,
):
    t = tl.program_id(0) * 64 + tl.arange(0, 64)
    if T % 64 == 0:
        location = tl.load(L + t * 2)
    else:
        location = tl.load(L + t * 2, t < T, 0)
    position = (
        (location >> 7) * 128 + (location & 31) * 4 + (location >> 5 & 3)
    )
    if T % 64 == 0:
        tl.store(M + position, t)
    else:
        tl.store(M + position, t, t < T)


@triton.jit
def _gather(
    X: tl.pointer_type(tl.int32),
    M: tl.pointer_type(tl.int16),
    Y: tl.pointer_type(tl.int32),
    H: tl.constexpr,
    C: tl.constexpr,
):
    page = tl.program_id(0)
    i = tl.arange(0, 128 * C)
    slot = i % 128
    head = tl.program_id(1) * C + i // 128
    token = tl.load(M + page * 128 + slot).to(tl.int32)
    live = token >= 0
    values = tl.load(X + token * H + head, live & (head < H), 0)
    offset = page * H * 128 + head * 128 + slot
    old = tl.load(Y + offset, head < H, 0)
    result = tl.where(live, values, old)
    tl.store(Y + offset, result, head < H)


@triton.jit(do_not_specialize=["Y"])
def _small_tokens(
    X: tl.pointer_type(tl.int32),
    L: tl.pointer_type(tl.int64),
    Y: tl.pointer_type(tl.int32),
    H: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.arange(0, H)
    location = tl.load(L + token).to(tl.int32)
    offset = (
        ((location >> 7) * H + head) * 128
        + (location & 31) * 4
        + (location >> 5 & 3)
    )
    value = tl.load(X + token * H + head)
    tl.store(Y + offset, value)


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
    if (
        512 < t < 12288
        and 0 < h <= 32
        and (h & h - 1 == 0)
        and (out.numel() <= 2147483648)
    ):
        bh = min(h, 8)
        bt = 128 if t >= 1024 else 32
        _alpha_scatter[triton.cdiv(t, bt),](
            x, out, locations, t, h, 128, bt, bh, num_warps=4
        )
        return out
    if (
        0 < t <= 512
        and 0 < h <= 32
        and (h & h - 1 == 0)
        and (sf_out.numel() <= 8589934592)
    ):
        _small_tokens[t,](x, locations, y, h, num_warps=4)
        return out
    if (
        12288 <= t <= 16384
        and 0 < h <= 32
        and (sf_out.numel() <= 536870912)
        and (t * 8 >= out.shape[0] * 128 * 3)
    ):
        pages = out.shape[0]
        mapping = torch.empty(
            (pages * 128,), dtype=torch.int16, device=sf_out.device
        )
        _fill[triton.cdiv(pages * 128, 1024),](
            mapping, pages * 128, num_warps=4
        )
        _index[triton.cdiv(t, 64),](locations, mapping, t, num_warps=4)
        c = min(triton.next_power_of_2(h), 32)
        _gather[pages, triton.cdiv(h, c)](x, mapping, out, h, c, num_warps=4)
        return out
    if t * h:
        grid = (min(triton.cdiv(t * h, 128), 4096),)
        if sf_out.numel() <= 8589934592:
            _scatter32[grid](x, locations, y, t, h, num_warps=4)
        else:
            _scatter64[grid](x, locations, y, t, h, num_warps=4)
    return out


__all__ = ["store_sf_interleaved"]
