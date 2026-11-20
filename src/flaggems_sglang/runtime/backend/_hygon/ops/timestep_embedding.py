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
    N: tl.constexpr = PARAMS[0]
    H: tl.constexpr = PARAMS[1]
    S: tl.constexpr = PARAMS[2]
    L: tl.constexpr = PARAMS[3]
    D: tl.constexpr = PARAMS[4]
    SCALE: tl.constexpr = PARAMS[5]
    FLIP: tl.constexpr = PARAMS[6]
    BR: tl.constexpr = 4
    BC: tl.constexpr = 256
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    i = tl.program_id(1) * BC + tl.arange(0, BC)
    times = tl.load(T + rows * S, rows < N, 0).to(tl.float32) * SCALE
    exponent = i.to(tl.float32) * L / D
    angle = times[:, None] * tl.exp(exponent)[None, :]
    if L <= 0 and D > 0 and (tl.max(tl.abs(times), 0) <= 4096):
        q = tl.floor(angle * 0.6366197723675814 + 0.5)
        r = tl.fma(-q, 1.570796251296997, angle)
        r = tl.fma(-q, 7.549789415861596e-08, r)
        z = r * r
        sp = tl.fma(z, 2.7557314297e-06, -0.00019841270114)
        sp = tl.fma(z, sp, 0.008333333768)
        sp = tl.fma(z, sp, -0.16666667163)
        sn0 = tl.fma(r * z, sp, r)
        cp = tl.fma(z, -2.7557314297e-07, 2.4801587642e-05)
        cp = tl.fma(z, cp, -0.0013888889225)
        cp = tl.fma(z, cp, 0.041666667908)
        cp = tl.fma(z, cp, -0.5)
        cs0 = tl.fma(z, cp, 1.0)
        k = q.to(tl.int32)
        sn = tl.where(k & 1 == 0, sn0, cs0)
        cs = tl.where(k & 1 == 0, cs0, -sn0)
        sn = tl.where(k & 2 == 0, sn, -sn)
        cs = tl.where(k & 2 == 0, cs, -cs)
    else:
        exponent = i.to(tl.float32) * L * tl.div_rn(1.0, D)
        angle = times[:, None] * libdevice.exp(exponent)[None, :]
        sn = libdevice.sin(angle)
        cs = libdevice.cos(angle)
    mask = (rows[:, None] < N) & (i[None, :] < H)
    first = cs if FLIP else sn
    second = sn if FLIP else cs
    tl.store(Out + rows[:, None] * (2 * H) + i[None, :], first, mask)
    tl.store(Out + rows[:, None] * (2 * H) + H + i[None, :], second, mask)


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


def _fallback(
    t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
):
    n = t.numel()
    h = dim // 2
    if n and h and h % 4:
        pitch = triton.cdiv(dim, 4) * 4
        storage = t.new_empty((n, pitch), dtype=torch.float32)
        c = triton.cdiv(pitch, 1024)
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
            1024,
            c,
            count,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return storage[:, :dim]
    out = t.new_empty((n, dim), dtype=torch.float32)
    if n == 0 or h == 0:
        return out
    _embedding.run(
        t,
        out,
        (
            n,
            h,
            t.stride(0),
            -math.log(max_period),
            h - downscale_freq_shift,
            scale,
            flip_sin_to_cos,
        ),
        grid=(triton.cdiv(n, 4), triton.cdiv(h, 256)),
        warmup=False,
    )
    return out


@triton.jit(do_not_specialize=["T"])
def _plain_row(T, Out, PARAMS: tl.constexpr):
    H: tl.constexpr = PARAMS[0]
    S: tl.constexpr = PARAMS[1]
    L: tl.constexpr = -math.log(PARAMS[2])
    D: tl.constexpr = H - PARAMS[3]
    SCALE: tl.constexpr = PARAMS[4]
    FLIP: tl.constexpr = PARAMS[5]
    row = tl.program_id(0)
    K: tl.constexpr = triton.next_power_of_2(max(H, 256)) // 4
    i = tl.arange(0, K)[:, None] * 4 + tl.arange(0, 4)[None, :]
    gate = tl.load(T + row * S).to(tl.float32) * SCALE
    exponent = i.to(tl.float32) * L * tl.div_rn(1.0, D)
    angle = gate * libdevice.exp(exponent)
    if L <= 0 and D > 0 and (tl.abs(gate) <= 1024):
        cycles = angle * 0.15915494309189535
        sn = tl.inline_asm_elementwise(
            "v_sin_f32 $0, $1",
            constraints="=v,v",
            args=[cycles],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        cs = tl.inline_asm_elementwise(
            "v_cos_f32 $0, $1",
            constraints="=v,v",
            args=[cycles],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        sn = libdevice.sin(angle)
        cs = libdevice.cos(angle)
    B: tl.constexpr = 4 * K
    cols = tl.arange(0, B)
    first = (cs if FLIP else sn).reshape(B)
    second = (sn if FLIP else cs).reshape(B)
    tl.store(tl.multiple_of(Out, 16) + row * 2 * H + cols, first, cols < H)
    tl.store(
        tl.multiple_of(Out, 16) + row * 2 * H + H + cols, second, cols < H
    )


def timestep_embedding(
    t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
):
    h = dim // 2
    n = t.numel()
    if not n or h < 128 or h > 512 or h % 4:
        return _fallback(
            t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
        )
    out = t.new_empty((n, dim), dtype=torch.float32)
    _plain_row.run(
        t,
        out,
        (
            h,
            t.stride(0),
            max_period,
            downscale_freq_shift,
            scale,
            flip_sin_to_cos,
        ),
        grid=(n,),
        warmup=False,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["timestep_embedding"]
