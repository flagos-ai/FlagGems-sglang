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

import math

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "Out"])
def _embedding(T, Out, PARAMS: tl.constexpr):
    Out = tl.multiple_of(Out, 16)
    H: tl.constexpr = PARAMS[0]
    S: tl.constexpr = PARAMS[1]
    N: tl.constexpr = PARAMS[2]
    L: tl.constexpr = PARAMS[3]
    D: tl.constexpr = PARAMS[4]
    SCALE: tl.constexpr = PARAMS[5]
    FLIP: tl.constexpr = PARAMS[6]
    B: tl.constexpr = 512
    C: tl.constexpr = (H + B - 1) // B
    for p in range(tl.program_id(0), N, tl.num_programs(0)):
        row = p // C
        i = p % C * B + tl.arange(0, B)
        t = tl.load(T + row * S).to(tl.float32)
        exponent = i.to(tl.float32) * L / D
        angle = SCALE * t * tl.exp(exponent)
        sin = tl.sin(angle)
        cos = tl.cos(angle)
        if FLIP:
            tl.store(Out + row * (2 * H) + i, cos, i < H)
            tl.store(Out + row * (2 * H) + H + i, sin, i < H)
        else:
            tl.store(Out + row * (2 * H) + i, sin, i < H)
            tl.store(Out + row * (2 * H) + H + i, cos, i < H)


@triton.jit
def _padded(
    T,
    Out,
    N: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    S: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    FLIP: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    COUNT: tl.constexpr,
):
    for p in range(tl.program_id(0), COUNT, tl.num_programs(0)):
        row = p // C
        col = p % C * B + tl.arange(0, B)
        i = tl.where(col < H, col, col - H)
        t = tl.load(T + row * S).to(tl.float32)
        angle = SCALE * t * tl.exp(i.to(tl.float32) * L / D)
        sn = tl.sin(angle)
        cs = tl.cos(angle)
        if FLIP:
            value = tl.where(col < H, cs, sn)
        else:
            value = tl.where(col < H, sn, cs)
        tl.store(Out + row * P + col, value, col < P)


def _baseline(
    t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
):
    n = t.numel()
    h = dim // 2
    if n and h and h % 4:
        pitch = triton.cdiv(dim, 4) * 4
        storage = t.new_empty((n, pitch), dtype=torch.float32)
        c = triton.cdiv(pitch, 512)
        count = n * c
        _padded[min(count, 40),](
            t,
            storage,
            n,
            h,
            pitch,
            t.stride(0),
            -math.log(max_period),
            h - downscale_freq_shift,
            scale,
            flip_sin_to_cos,
            512,
            c,
            count,
            enable_fp_fusion=False,
        )
        return storage[:, :dim]
    out = t.new_empty((n, dim), dtype=torch.float32)
    if n == 0 or h == 0:
        return out
    b = 512
    c = triton.cdiv(h, b)
    count = n * c
    _embedding.run(
        t,
        out,
        (
            h,
            t.stride(0),
            n * c,
            -math.log(max_period),
            h - downscale_freq_shift,
            scale,
            flip_sin_to_cos,
        ),
        grid=(min(n * c, 40),),
        warmup=False,
        enable_fp_fusion=False,
    )
    return out


@triton.jit(do_not_specialize=["T", "Out"])
def _hoist(T, Out, PARAMS: tl.constexpr):
    Out = tl.cast(tl.multiple_of(Out, 16), tl.pointer_type(tl.float32))
    H: tl.constexpr = PARAMS[0]
    S: tl.constexpr = PARAMS[1]
    N: tl.constexpr = PARAMS[2]
    L: tl.constexpr = PARAMS[3]
    D: tl.constexpr = PARAMS[4]
    SCALE: tl.constexpr = PARAMS[5]
    FLIP: tl.constexpr = PARAMS[6]
    B: tl.constexpr = 512
    i = tl.arange(0, B)
    freq = tl.exp(i.to(tl.float32) * L / D)
    for row in range(tl.program_id(0), N, tl.num_programs(0)):
        t = tl.load(T + row * S).to(tl.float32)
        angle = SCALE * t * freq
        sn = tl.sin(angle)
        cs = tl.cos(angle)
        if FLIP:
            tl.store(Out + row * (2 * H) + i, cs, i < H)
            tl.store(Out + row * (2 * H) + H + i, sn, i < H)
        else:
            tl.store(Out + row * (2 * H) + i, sn, i < H)
            tl.store(Out + row * (2 * H) + H + i, cs, i < H)


def timestep_embedding(
    t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
):
    n, h = (t.numel(), dim // 2)
    if not n or not h or h > 512 or h % 4:
        return _baseline(
            t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
        )
    out = t.new_empty((n, dim), dtype=torch.float32)
    _hoist.run(
        t,
        out,
        (
            h,
            t.stride(0),
            n,
            -math.log(max_period),
            h - downscale_freq_shift,
            scale,
            flip_sin_to_cos,
            512,
        ),
        grid=(min(n, 40),),
        warmup=False,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["timestep_embedding"]
