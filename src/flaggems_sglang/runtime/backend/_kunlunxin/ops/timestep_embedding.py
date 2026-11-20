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


@triton.jit(do_not_specialize=["T"])
def _rows(T, Out, PARAMS: tl.constexpr):
    N: tl.constexpr = PARAMS[0]
    H: tl.constexpr = PARAMS[1]
    S: tl.constexpr = PARAMS[2]
    L: tl.constexpr = PARAMS[3]
    D: tl.constexpr = PARAMS[4]
    SCALE: tl.constexpr = PARAMS[5]
    FLIP: tl.constexpr = PARAMS[6]
    B: tl.constexpr = 1024
    R: tl.constexpr = 22 if N >= 512 else 1
    C: tl.constexpr = (H + B - 1) // B
    COUNT: tl.constexpr = (N + R - 1) // R * C
    for p in range(tl.program_id(0), COUNT, tl.num_programs(0)):
        i = p % C * B + tl.arange(0, B)
        freq = tl.exp(i.to(tl.float32) * L / D)
        start = p // C * R
        for row in range(start, tl.minimum(start + R, N)):
            t = tl.load(T + row * S).to(tl.float32)
            angle = SCALE * t * freq
            sn = tl.sin(angle)
            cs = tl.cos(angle)
            if FLIP:
                tl.store(
                    Out + row * 2 * H + i,
                    cs,
                    i < H,
                    mem_sync_mode="async" if N >= 512 else "sync",
                )
                tl.store(Out + row * 2 * H + H + i, sn, i < H)
            else:
                tl.store(
                    Out + row * 2 * H + i,
                    sn,
                    i < H,
                    mem_sync_mode="async" if N >= 512 else "sync",
                )
                tl.store(Out + row * 2 * H + H + i, cs, i < H)


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
        value = tl.where((col < H) != FLIP, sn, cs)
        tl.store(Out + row * P + col, value, col < P)


@triton.jit(do_not_specialize=["T"])
def _pipeline(T, Out, PARAMS: tl.constexpr):
    N: tl.constexpr = PARAMS[0]
    H: tl.constexpr = PARAMS[1]
    S: tl.constexpr = PARAMS[2]
    L: tl.constexpr = PARAMS[3]
    D: tl.constexpr = PARAMS[4]
    SCALE: tl.constexpr = PARAMS[5]
    FLIP: tl.constexpr = PARAMS[6]
    B: tl.constexpr = 1024
    R: tl.constexpr = 4 if N < 128 else 20 if N < 512 else 44
    C: tl.constexpr = (H + B - 1) // B
    p = tl.program_id(0)
    i = p % C * B + tl.arange(0, B)
    frequency = tl.exp(i.to(tl.float32) * L / D)
    begin = p // C * R
    for row in range(begin, tl.minimum(begin + R, N), 4):
        times0 = tl.load(T + (row + 0) * S).to(tl.float32) * SCALE
        times1 = tl.load(T + (row + 1) * S).to(tl.float32) * SCALE
        times2 = tl.load(T + (row + 2) * S).to(tl.float32) * SCALE
        times3 = tl.load(T + (row + 3) * S).to(tl.float32) * SCALE
        angle0 = times0 * frequency
        sn0 = tl.sin(angle0)
        cs0 = tl.cos(angle0)
        angle1 = times1 * frequency
        sn1 = tl.sin(angle1)
        cs1 = tl.cos(angle1)
        angle2 = times2 * frequency
        sn2 = tl.sin(angle2)
        cs2 = tl.cos(angle2)
        angle3 = times3 * frequency
        sn3 = tl.sin(angle3)
        cs3 = tl.cos(angle3)
        first0 = cs0 if FLIP else sn0
        second0 = sn0 if FLIP else cs0
        tl.store(
            Out + (row + 0) * 2 * H + i, first0, i < H, mem_sync_mode="async"
        )
        tl.store(
            Out + (row + 0) * 2 * H + H + i,
            second0,
            i < H,
            mem_sync_mode="async",
        )
        first1 = cs1 if FLIP else sn1
        second1 = sn1 if FLIP else cs1
        tl.store(
            Out + (row + 1) * 2 * H + i, first1, i < H, mem_sync_mode="async"
        )
        tl.store(
            Out + (row + 1) * 2 * H + H + i,
            second1,
            i < H,
            mem_sync_mode="async",
        )
        first2 = cs2 if FLIP else sn2
        second2 = sn2 if FLIP else cs2
        tl.store(
            Out + (row + 2) * 2 * H + i, first2, i < H, mem_sync_mode="async"
        )
        tl.store(
            Out + (row + 2) * 2 * H + H + i,
            second2,
            i < H,
            mem_sync_mode="async",
        )
        first3 = cs3 if FLIP else sn3
        second3 = sn3 if FLIP else cs3
        tl.store(
            Out + (row + 3) * 2 * H + i, first3, i < H, mem_sync_mode="async"
        )
        tl.store(Out + (row + 3) * 2 * H + H + i, second3, i < H)


def timestep_embedding(
    t, dim, flip_sin_to_cos, downscale_freq_shift, scale, max_period
):
    n = t.numel()
    h = dim // 2
    if n and h and h % 4:
        pitch = triton.cdiv(dim, 4) * 4
        storage = t.new_empty((n, pitch), dtype=torch.float32)
        c = triton.cdiv(pitch, 128)
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
            128,
            c,
            count,
            enable_fp_fusion=False,
        )
        return storage[:, :dim]
    out = t.new_empty((n, dim), dtype=torch.float32)
    if n == 0 or h == 0:
        return out
    if n >= 64 and n % 4 == 0:
        count = triton.cdiv(
            n, 4 if n < 128 else 20 if n < 512 else 44
        ) * triton.cdiv(h, 1024)
        _pipeline.run(
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
            grid=(count,),
            warmup=False,
            enable_fp_fusion=False,
            unroll_num=8,
            budget_tiling=False,
        )
    else:
        b = 1024
        r = 22 if n >= 512 else 1
        c = triton.cdiv(h, b)
        count = triton.cdiv(n, r) * c
        _rows.run(
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
            grid=(count,),
            warmup=False,
            enable_fp_fusion=False,
            unroll_num=8,
            budget_tiling=False,
        )
    return out


__all__ = ["timestep_embedding"]
