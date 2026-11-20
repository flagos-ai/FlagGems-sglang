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
from triton.language.extra import libdevice


@triton.jit(do_not_specialize=["T"])
def _embedding(T, Out, PARAMS: tl.constexpr):
    H: tl.constexpr = PARAMS[0]
    S: tl.constexpr = PARAMS[1]
    L: tl.constexpr = PARAMS[2]
    D: tl.constexpr = PARAMS[3]
    SCALE: tl.constexpr = PARAMS[4]
    FLIP: tl.constexpr = PARAMS[5]
    B: tl.constexpr = 256
    C: tl.constexpr = (H + B - 1) // B
    p = tl.program_id(0)
    row = p // C
    i = p % C * B + tl.arange(0, B)
    t = tl.load(T + row * S).to(tl.float32)
    exponent = i.to(tl.float32) * L * tl.div_rn(1.0, D)
    angle = SCALE * t * libdevice.exp(exponent)
    sin = libdevice.sin(angle)
    cos = libdevice.cos(angle)
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
        angle = (
            SCALE * t * libdevice.exp(i.to(tl.float32) * L * tl.div_rn(1.0, D))
        )
        sn = libdevice.sin(angle)
        cs = libdevice.cos(angle)
        value = tl.where((col < H) != FLIP, sn, cs)
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
        _padded[min(count, 65535),](
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
            num_warps=1,
            enable_fp_fusion=False,
        )
        return storage[:, :dim]
    out = t.new_empty((n, dim), dtype=torch.float32)
    if n == 0 or h == 0:
        return out
    b = 256
    c = triton.cdiv(h, b)
    _embedding.run(
        t,
        out,
        (
            h,
            t.stride(0),
            -math.log(max_period),
            h - downscale_freq_shift,
            scale,
            flip_sin_to_cos,
        ),
        grid=(n * c,),
        warmup=False,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out


@triton.jit(do_not_specialize=["T"])
def _pair(T, Out, PARAMS: tl.constexpr):
    H: tl.constexpr = PARAMS[0]
    S: tl.constexpr = PARAMS[1]
    L: tl.constexpr = PARAMS[2]
    D: tl.constexpr = PARAMS[3]
    SCALE: tl.constexpr = PARAMS[4]
    FLIP: tl.constexpr = PARAMS[5]
    row = tl.program_id(0)
    col = tl.arange(0, H // 2)[:, None] * 4 + tl.arange(0, 4)[None, :]
    i = col % H
    want_cos = (col >= H) ^ FLIP
    times = tl.load(T + row * S).to(tl.float32) * SCALE
    exponent = i.to(tl.float32) * L * tl.div_rn(1.0, D)
    if L <= 0 and D > 0 and (tl.abs(times) <= 1024):
        shift = want_cos.to(tl.float32) * 0.5
        frequency = tl.exp(exponent)
        x = tl.fma(frequency, times * 0.3183098861837907, shift)
        rounded = x + 12582912.0
        q = rounded - 12582912.0
        r = x - q
        z = r * r
        p = tl.fma(z, -0.5571601568635588, 2.5428757555727204)
        p = tl.fma(z, p, -5.167248130747836)
        p = tl.fma(z, p, 3.1415847593922575)
        value = r * p
        sign = (rounded.to(tl.uint32, bitcast=True) & 1) << 31
        value = (value.to(tl.uint32, bitcast=True) ^ sign).to(
            tl.float32, bitcast=True
        )
        cols = tl.arange(0, 2 * H)
        tl.store(Out + row * 2 * H + cols, value.reshape(2 * H))
    else:
        frequency = libdevice.exp(exponent)
        angle = times * frequency
        sn = libdevice.sin(angle)
        cs = libdevice.cos(angle)
        value = tl.where(want_cos, cs, sn)
        cols = tl.arange(0, 2 * H)
        tl.store(Out + row * 2 * H + cols, value.reshape(2 * H))


def _paired(t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period):
    n, h = (t.numel(), dim // 2)
    if not n or h < 128 or h > 512 or h & h - 1:
        return _baseline(
            t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
        )
    out = t.new_empty((n, dim), dtype=torch.float32)
    _pair.run(
        t,
        out,
        (
            h,
            t.stride(0),
            -math.log(max_period),
            h - downscale_freq_shift,
            scale,
            flip_sin_to_cos,
        ),
        grid=(n,),
        warmup=False,
        num_warps=h // 128,
        enable_fp_fusion=False,
    )
    return out


def timestep_embedding(
    t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
):
    return _paired(
        t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
    )


__all__ = ["timestep_embedding"]
