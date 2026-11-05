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
def _stage(x, N: tl.constexpr, S: tl.constexpr):
    if S == 1:
        a, b = tl.split(tl.reshape(x, (1, N // 2, 2)))
        return tl.reshape(tl.join(a + b, a - b), (1, N))
    a, b = tl.split(
        tl.permute(tl.reshape(x, (1, N // (2 * S), 2, S)), (0, 1, 3, 2))
    )
    return tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 1, 3, 2)), (1, N))


@triton.jit
def _row(X, Y, N: tl.constexpr, SCALE: tl.constexpr):
    rows = tl.program_id(0) + tl.arange(0, 1)
    idx = rows[:, None] * N + tl.arange(0, N)[None, :]
    v = tl.load(X + idx, cache_modifier=".cg").to(tl.float32)
    for stage in tl.static_range(N.bit_length() - 1):
        v = _stage(v, N, N >> (stage + 1))
    tl.store(Y + idx, v * SCALE, cache_modifier=".cs")


@triton.jit
def _wave_stage(v, S: tl.constexpr):
    p = tl.reshape(tl.flip(tl.reshape(v, (32 // (2 * S), 2, S)), 1), (32,))
    sign = tl.where((tl.arange(0, 32) & S) != 0, -1.0, 1.0)
    return tl.fma(sign, v, p)


@triton.jit
def _wave(v):
    for stage in tl.static_range(5):
        v = _wave_stage(v, 16 >> stage)
    return v


@triton.jit
def _planes(X, Y, N: tl.constexpr, SCALE: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, 32)
    if N == 128:
        v0_0 = tl.load(X + row * N + lane + 0, cache_modifier=".cg").to(
            tl.float32
        )
        v0_1 = tl.load(X + row * N + lane + 32, cache_modifier=".cg").to(
            tl.float32
        )
        v0_2 = tl.load(X + row * N + lane + 64, cache_modifier=".cg").to(
            tl.float32
        )
        v0_3 = tl.load(X + row * N + lane + 96, cache_modifier=".cg").to(
            tl.float32
        )
        v1_0 = v0_0 + v0_2
        v1_1 = v0_1 + v0_3
        v1_2 = v0_0 - v0_2
        v1_3 = v0_1 - v0_3
        v2_0 = v1_0 + v1_1
        v2_1 = v1_0 - v1_1
        v2_2 = v1_2 + v1_3
        v2_3 = v1_2 - v1_3
        w0 = _wave(v2_0)
        tl.store(Y + row * N + lane + 0, w0 * SCALE, cache_modifier=".cs")
        w1 = _wave(v2_1)
        tl.store(Y + row * N + lane + 32, w1 * SCALE, cache_modifier=".cs")
        w2 = _wave(v2_2)
        tl.store(Y + row * N + lane + 64, w2 * SCALE, cache_modifier=".cs")
        w3 = _wave(v2_3)
        tl.store(Y + row * N + lane + 96, w3 * SCALE, cache_modifier=".cs")
    elif N == 256:
        v0_0 = tl.load(X + row * N + lane + 0, cache_modifier=".cg").to(
            tl.float32
        )
        v0_1 = tl.load(X + row * N + lane + 32, cache_modifier=".cg").to(
            tl.float32
        )
        v0_2 = tl.load(X + row * N + lane + 64, cache_modifier=".cg").to(
            tl.float32
        )
        v0_3 = tl.load(X + row * N + lane + 96, cache_modifier=".cg").to(
            tl.float32
        )
        v0_4 = tl.load(X + row * N + lane + 128, cache_modifier=".cg").to(
            tl.float32
        )
        v0_5 = tl.load(X + row * N + lane + 160, cache_modifier=".cg").to(
            tl.float32
        )
        v0_6 = tl.load(X + row * N + lane + 192, cache_modifier=".cg").to(
            tl.float32
        )
        v0_7 = tl.load(X + row * N + lane + 224, cache_modifier=".cg").to(
            tl.float32
        )
        v1_0 = v0_0 + v0_4
        v1_1 = v0_1 + v0_5
        v1_2 = v0_2 + v0_6
        v1_3 = v0_3 + v0_7
        v1_4 = v0_0 - v0_4
        v1_5 = v0_1 - v0_5
        v1_6 = v0_2 - v0_6
        v1_7 = v0_3 - v0_7
        v2_0 = v1_0 + v1_2
        v2_1 = v1_1 + v1_3
        v2_2 = v1_0 - v1_2
        v2_3 = v1_1 - v1_3
        v2_4 = v1_4 + v1_6
        v2_5 = v1_5 + v1_7
        v2_6 = v1_4 - v1_6
        v2_7 = v1_5 - v1_7
        v3_0 = v2_0 + v2_1
        v3_1 = v2_0 - v2_1
        v3_2 = v2_2 + v2_3
        v3_3 = v2_2 - v2_3
        v3_4 = v2_4 + v2_5
        v3_5 = v2_4 - v2_5
        v3_6 = v2_6 + v2_7
        v3_7 = v2_6 - v2_7
        w0 = _wave(v3_0)
        tl.store(Y + row * N + lane + 0, w0 * SCALE, cache_modifier=".cs")
        w1 = _wave(v3_1)
        tl.store(Y + row * N + lane + 32, w1 * SCALE, cache_modifier=".cs")
        w2 = _wave(v3_2)
        tl.store(Y + row * N + lane + 64, w2 * SCALE, cache_modifier=".cs")
        w3 = _wave(v3_3)
        tl.store(Y + row * N + lane + 96, w3 * SCALE, cache_modifier=".cs")
        w4 = _wave(v3_4)
        tl.store(Y + row * N + lane + 128, w4 * SCALE, cache_modifier=".cs")
        w5 = _wave(v3_5)
        tl.store(Y + row * N + lane + 160, w5 * SCALE, cache_modifier=".cs")
        w6 = _wave(v3_6)
        tl.store(Y + row * N + lane + 192, w6 * SCALE, cache_modifier=".cs")
        w7 = _wave(v3_7)
        tl.store(Y + row * N + lane + 224, w7 * SCALE, cache_modifier=".cs")
    elif N == 512:
        v0_0 = tl.load(X + row * N + lane + 0, cache_modifier=".cg").to(
            tl.float32
        )
        v0_1 = tl.load(X + row * N + lane + 32, cache_modifier=".cg").to(
            tl.float32
        )
        v0_2 = tl.load(X + row * N + lane + 64, cache_modifier=".cg").to(
            tl.float32
        )
        v0_3 = tl.load(X + row * N + lane + 96, cache_modifier=".cg").to(
            tl.float32
        )
        v0_4 = tl.load(X + row * N + lane + 128, cache_modifier=".cg").to(
            tl.float32
        )
        v0_5 = tl.load(X + row * N + lane + 160, cache_modifier=".cg").to(
            tl.float32
        )
        v0_6 = tl.load(X + row * N + lane + 192, cache_modifier=".cg").to(
            tl.float32
        )
        v0_7 = tl.load(X + row * N + lane + 224, cache_modifier=".cg").to(
            tl.float32
        )
        v0_8 = tl.load(X + row * N + lane + 256, cache_modifier=".cg").to(
            tl.float32
        )
        v0_9 = tl.load(X + row * N + lane + 288, cache_modifier=".cg").to(
            tl.float32
        )
        v0_10 = tl.load(X + row * N + lane + 320, cache_modifier=".cg").to(
            tl.float32
        )
        v0_11 = tl.load(X + row * N + lane + 352, cache_modifier=".cg").to(
            tl.float32
        )
        v0_12 = tl.load(X + row * N + lane + 384, cache_modifier=".cg").to(
            tl.float32
        )
        v0_13 = tl.load(X + row * N + lane + 416, cache_modifier=".cg").to(
            tl.float32
        )
        v0_14 = tl.load(X + row * N + lane + 448, cache_modifier=".cg").to(
            tl.float32
        )
        v0_15 = tl.load(X + row * N + lane + 480, cache_modifier=".cg").to(
            tl.float32
        )
        v1_0 = v0_0 + v0_8
        v1_1 = v0_1 + v0_9
        v1_2 = v0_2 + v0_10
        v1_3 = v0_3 + v0_11
        v1_4 = v0_4 + v0_12
        v1_5 = v0_5 + v0_13
        v1_6 = v0_6 + v0_14
        v1_7 = v0_7 + v0_15
        v1_8 = v0_0 - v0_8
        v1_9 = v0_1 - v0_9
        v1_10 = v0_2 - v0_10
        v1_11 = v0_3 - v0_11
        v1_12 = v0_4 - v0_12
        v1_13 = v0_5 - v0_13
        v1_14 = v0_6 - v0_14
        v1_15 = v0_7 - v0_15
        v2_0 = v1_0 + v1_4
        v2_1 = v1_1 + v1_5
        v2_2 = v1_2 + v1_6
        v2_3 = v1_3 + v1_7
        v2_4 = v1_0 - v1_4
        v2_5 = v1_1 - v1_5
        v2_6 = v1_2 - v1_6
        v2_7 = v1_3 - v1_7
        v2_8 = v1_8 + v1_12
        v2_9 = v1_9 + v1_13
        v2_10 = v1_10 + v1_14
        v2_11 = v1_11 + v1_15
        v2_12 = v1_8 - v1_12
        v2_13 = v1_9 - v1_13
        v2_14 = v1_10 - v1_14
        v2_15 = v1_11 - v1_15
        v3_0 = v2_0 + v2_2
        v3_1 = v2_1 + v2_3
        v3_2 = v2_0 - v2_2
        v3_3 = v2_1 - v2_3
        v3_4 = v2_4 + v2_6
        v3_5 = v2_5 + v2_7
        v3_6 = v2_4 - v2_6
        v3_7 = v2_5 - v2_7
        v3_8 = v2_8 + v2_10
        v3_9 = v2_9 + v2_11
        v3_10 = v2_8 - v2_10
        v3_11 = v2_9 - v2_11
        v3_12 = v2_12 + v2_14
        v3_13 = v2_13 + v2_15
        v3_14 = v2_12 - v2_14
        v3_15 = v2_13 - v2_15
        v4_0 = v3_0 + v3_1
        v4_1 = v3_0 - v3_1
        v4_2 = v3_2 + v3_3
        v4_3 = v3_2 - v3_3
        v4_4 = v3_4 + v3_5
        v4_5 = v3_4 - v3_5
        v4_6 = v3_6 + v3_7
        v4_7 = v3_6 - v3_7
        v4_8 = v3_8 + v3_9
        v4_9 = v3_8 - v3_9
        v4_10 = v3_10 + v3_11
        v4_11 = v3_10 - v3_11
        v4_12 = v3_12 + v3_13
        v4_13 = v3_12 - v3_13
        v4_14 = v3_14 + v3_15
        v4_15 = v3_14 - v3_15
        w0 = _wave(v4_0)
        tl.store(Y + row * N + lane + 0, w0 * SCALE, cache_modifier=".cs")
        w1 = _wave(v4_1)
        tl.store(Y + row * N + lane + 32, w1 * SCALE, cache_modifier=".cs")
        w2 = _wave(v4_2)
        tl.store(Y + row * N + lane + 64, w2 * SCALE, cache_modifier=".cs")
        w3 = _wave(v4_3)
        tl.store(Y + row * N + lane + 96, w3 * SCALE, cache_modifier=".cs")
        w4 = _wave(v4_4)
        tl.store(Y + row * N + lane + 128, w4 * SCALE, cache_modifier=".cs")
        w5 = _wave(v4_5)
        tl.store(Y + row * N + lane + 160, w5 * SCALE, cache_modifier=".cs")
        w6 = _wave(v4_6)
        tl.store(Y + row * N + lane + 192, w6 * SCALE, cache_modifier=".cs")
        w7 = _wave(v4_7)
        tl.store(Y + row * N + lane + 224, w7 * SCALE, cache_modifier=".cs")
        w8 = _wave(v4_8)
        tl.store(Y + row * N + lane + 256, w8 * SCALE, cache_modifier=".cs")
        w9 = _wave(v4_9)
        tl.store(Y + row * N + lane + 288, w9 * SCALE, cache_modifier=".cs")
        w10 = _wave(v4_10)
        tl.store(Y + row * N + lane + 320, w10 * SCALE, cache_modifier=".cs")
        w11 = _wave(v4_11)
        tl.store(Y + row * N + lane + 352, w11 * SCALE, cache_modifier=".cs")
        w12 = _wave(v4_12)
        tl.store(Y + row * N + lane + 384, w12 * SCALE, cache_modifier=".cs")
        w13 = _wave(v4_13)
        tl.store(Y + row * N + lane + 416, w13 * SCALE, cache_modifier=".cs")
        w14 = _wave(v4_14)
        tl.store(Y + row * N + lane + 448, w14 * SCALE, cache_modifier=".cs")
        w15 = _wave(v4_15)
        tl.store(Y + row * N + lane + 480, w15 * SCALE, cache_modifier=".cs")
    elif N == 1024:
        v0_0 = tl.load(X + row * N + lane + 0, cache_modifier=".cg").to(
            tl.float32
        )
        v0_1 = tl.load(X + row * N + lane + 32, cache_modifier=".cg").to(
            tl.float32
        )
        v0_2 = tl.load(X + row * N + lane + 64, cache_modifier=".cg").to(
            tl.float32
        )
        v0_3 = tl.load(X + row * N + lane + 96, cache_modifier=".cg").to(
            tl.float32
        )
        v0_4 = tl.load(X + row * N + lane + 128, cache_modifier=".cg").to(
            tl.float32
        )
        v0_5 = tl.load(X + row * N + lane + 160, cache_modifier=".cg").to(
            tl.float32
        )
        v0_6 = tl.load(X + row * N + lane + 192, cache_modifier=".cg").to(
            tl.float32
        )
        v0_7 = tl.load(X + row * N + lane + 224, cache_modifier=".cg").to(
            tl.float32
        )
        v0_8 = tl.load(X + row * N + lane + 256, cache_modifier=".cg").to(
            tl.float32
        )
        v0_9 = tl.load(X + row * N + lane + 288, cache_modifier=".cg").to(
            tl.float32
        )
        v0_10 = tl.load(X + row * N + lane + 320, cache_modifier=".cg").to(
            tl.float32
        )
        v0_11 = tl.load(X + row * N + lane + 352, cache_modifier=".cg").to(
            tl.float32
        )
        v0_12 = tl.load(X + row * N + lane + 384, cache_modifier=".cg").to(
            tl.float32
        )
        v0_13 = tl.load(X + row * N + lane + 416, cache_modifier=".cg").to(
            tl.float32
        )
        v0_14 = tl.load(X + row * N + lane + 448, cache_modifier=".cg").to(
            tl.float32
        )
        v0_15 = tl.load(X + row * N + lane + 480, cache_modifier=".cg").to(
            tl.float32
        )
        v0_16 = tl.load(X + row * N + lane + 512, cache_modifier=".cg").to(
            tl.float32
        )
        v0_17 = tl.load(X + row * N + lane + 544, cache_modifier=".cg").to(
            tl.float32
        )
        v0_18 = tl.load(X + row * N + lane + 576, cache_modifier=".cg").to(
            tl.float32
        )
        v0_19 = tl.load(X + row * N + lane + 608, cache_modifier=".cg").to(
            tl.float32
        )
        v0_20 = tl.load(X + row * N + lane + 640, cache_modifier=".cg").to(
            tl.float32
        )
        v0_21 = tl.load(X + row * N + lane + 672, cache_modifier=".cg").to(
            tl.float32
        )
        v0_22 = tl.load(X + row * N + lane + 704, cache_modifier=".cg").to(
            tl.float32
        )
        v0_23 = tl.load(X + row * N + lane + 736, cache_modifier=".cg").to(
            tl.float32
        )
        v0_24 = tl.load(X + row * N + lane + 768, cache_modifier=".cg").to(
            tl.float32
        )
        v0_25 = tl.load(X + row * N + lane + 800, cache_modifier=".cg").to(
            tl.float32
        )
        v0_26 = tl.load(X + row * N + lane + 832, cache_modifier=".cg").to(
            tl.float32
        )
        v0_27 = tl.load(X + row * N + lane + 864, cache_modifier=".cg").to(
            tl.float32
        )
        v0_28 = tl.load(X + row * N + lane + 896, cache_modifier=".cg").to(
            tl.float32
        )
        v0_29 = tl.load(X + row * N + lane + 928, cache_modifier=".cg").to(
            tl.float32
        )
        v0_30 = tl.load(X + row * N + lane + 960, cache_modifier=".cg").to(
            tl.float32
        )
        v0_31 = tl.load(X + row * N + lane + 992, cache_modifier=".cg").to(
            tl.float32
        )
        v1_0 = v0_0 + v0_16
        v1_1 = v0_1 + v0_17
        v1_2 = v0_2 + v0_18
        v1_3 = v0_3 + v0_19
        v1_4 = v0_4 + v0_20
        v1_5 = v0_5 + v0_21
        v1_6 = v0_6 + v0_22
        v1_7 = v0_7 + v0_23
        v1_8 = v0_8 + v0_24
        v1_9 = v0_9 + v0_25
        v1_10 = v0_10 + v0_26
        v1_11 = v0_11 + v0_27
        v1_12 = v0_12 + v0_28
        v1_13 = v0_13 + v0_29
        v1_14 = v0_14 + v0_30
        v1_15 = v0_15 + v0_31
        v1_16 = v0_0 - v0_16
        v1_17 = v0_1 - v0_17
        v1_18 = v0_2 - v0_18
        v1_19 = v0_3 - v0_19
        v1_20 = v0_4 - v0_20
        v1_21 = v0_5 - v0_21
        v1_22 = v0_6 - v0_22
        v1_23 = v0_7 - v0_23
        v1_24 = v0_8 - v0_24
        v1_25 = v0_9 - v0_25
        v1_26 = v0_10 - v0_26
        v1_27 = v0_11 - v0_27
        v1_28 = v0_12 - v0_28
        v1_29 = v0_13 - v0_29
        v1_30 = v0_14 - v0_30
        v1_31 = v0_15 - v0_31
        v2_0 = v1_0 + v1_8
        v2_1 = v1_1 + v1_9
        v2_2 = v1_2 + v1_10
        v2_3 = v1_3 + v1_11
        v2_4 = v1_4 + v1_12
        v2_5 = v1_5 + v1_13
        v2_6 = v1_6 + v1_14
        v2_7 = v1_7 + v1_15
        v2_8 = v1_0 - v1_8
        v2_9 = v1_1 - v1_9
        v2_10 = v1_2 - v1_10
        v2_11 = v1_3 - v1_11
        v2_12 = v1_4 - v1_12
        v2_13 = v1_5 - v1_13
        v2_14 = v1_6 - v1_14
        v2_15 = v1_7 - v1_15
        v2_16 = v1_16 + v1_24
        v2_17 = v1_17 + v1_25
        v2_18 = v1_18 + v1_26
        v2_19 = v1_19 + v1_27
        v2_20 = v1_20 + v1_28
        v2_21 = v1_21 + v1_29
        v2_22 = v1_22 + v1_30
        v2_23 = v1_23 + v1_31
        v2_24 = v1_16 - v1_24
        v2_25 = v1_17 - v1_25
        v2_26 = v1_18 - v1_26
        v2_27 = v1_19 - v1_27
        v2_28 = v1_20 - v1_28
        v2_29 = v1_21 - v1_29
        v2_30 = v1_22 - v1_30
        v2_31 = v1_23 - v1_31
        v3_0 = v2_0 + v2_4
        v3_1 = v2_1 + v2_5
        v3_2 = v2_2 + v2_6
        v3_3 = v2_3 + v2_7
        v3_4 = v2_0 - v2_4
        v3_5 = v2_1 - v2_5
        v3_6 = v2_2 - v2_6
        v3_7 = v2_3 - v2_7
        v3_8 = v2_8 + v2_12
        v3_9 = v2_9 + v2_13
        v3_10 = v2_10 + v2_14
        v3_11 = v2_11 + v2_15
        v3_12 = v2_8 - v2_12
        v3_13 = v2_9 - v2_13
        v3_14 = v2_10 - v2_14
        v3_15 = v2_11 - v2_15
        v3_16 = v2_16 + v2_20
        v3_17 = v2_17 + v2_21
        v3_18 = v2_18 + v2_22
        v3_19 = v2_19 + v2_23
        v3_20 = v2_16 - v2_20
        v3_21 = v2_17 - v2_21
        v3_22 = v2_18 - v2_22
        v3_23 = v2_19 - v2_23
        v3_24 = v2_24 + v2_28
        v3_25 = v2_25 + v2_29
        v3_26 = v2_26 + v2_30
        v3_27 = v2_27 + v2_31
        v3_28 = v2_24 - v2_28
        v3_29 = v2_25 - v2_29
        v3_30 = v2_26 - v2_30
        v3_31 = v2_27 - v2_31
        v4_0 = v3_0 + v3_2
        v4_1 = v3_1 + v3_3
        v4_2 = v3_0 - v3_2
        v4_3 = v3_1 - v3_3
        v4_4 = v3_4 + v3_6
        v4_5 = v3_5 + v3_7
        v4_6 = v3_4 - v3_6
        v4_7 = v3_5 - v3_7
        v4_8 = v3_8 + v3_10
        v4_9 = v3_9 + v3_11
        v4_10 = v3_8 - v3_10
        v4_11 = v3_9 - v3_11
        v4_12 = v3_12 + v3_14
        v4_13 = v3_13 + v3_15
        v4_14 = v3_12 - v3_14
        v4_15 = v3_13 - v3_15
        v4_16 = v3_16 + v3_18
        v4_17 = v3_17 + v3_19
        v4_18 = v3_16 - v3_18
        v4_19 = v3_17 - v3_19
        v4_20 = v3_20 + v3_22
        v4_21 = v3_21 + v3_23
        v4_22 = v3_20 - v3_22
        v4_23 = v3_21 - v3_23
        v4_24 = v3_24 + v3_26
        v4_25 = v3_25 + v3_27
        v4_26 = v3_24 - v3_26
        v4_27 = v3_25 - v3_27
        v4_28 = v3_28 + v3_30
        v4_29 = v3_29 + v3_31
        v4_30 = v3_28 - v3_30
        v4_31 = v3_29 - v3_31
        v5_0 = v4_0 + v4_1
        v5_1 = v4_0 - v4_1
        v5_2 = v4_2 + v4_3
        v5_3 = v4_2 - v4_3
        v5_4 = v4_4 + v4_5
        v5_5 = v4_4 - v4_5
        v5_6 = v4_6 + v4_7
        v5_7 = v4_6 - v4_7
        v5_8 = v4_8 + v4_9
        v5_9 = v4_8 - v4_9
        v5_10 = v4_10 + v4_11
        v5_11 = v4_10 - v4_11
        v5_12 = v4_12 + v4_13
        v5_13 = v4_12 - v4_13
        v5_14 = v4_14 + v4_15
        v5_15 = v4_14 - v4_15
        v5_16 = v4_16 + v4_17
        v5_17 = v4_16 - v4_17
        v5_18 = v4_18 + v4_19
        v5_19 = v4_18 - v4_19
        v5_20 = v4_20 + v4_21
        v5_21 = v4_20 - v4_21
        v5_22 = v4_22 + v4_23
        v5_23 = v4_22 - v4_23
        v5_24 = v4_24 + v4_25
        v5_25 = v4_24 - v4_25
        v5_26 = v4_26 + v4_27
        v5_27 = v4_26 - v4_27
        v5_28 = v4_28 + v4_29
        v5_29 = v4_28 - v4_29
        v5_30 = v4_30 + v4_31
        v5_31 = v4_30 - v4_31
        w0 = _wave(v5_0)
        tl.store(Y + row * N + lane + 0, w0 * SCALE, cache_modifier=".cs")
        w1 = _wave(v5_1)
        tl.store(Y + row * N + lane + 32, w1 * SCALE, cache_modifier=".cs")
        w2 = _wave(v5_2)
        tl.store(Y + row * N + lane + 64, w2 * SCALE, cache_modifier=".cs")
        w3 = _wave(v5_3)
        tl.store(Y + row * N + lane + 96, w3 * SCALE, cache_modifier=".cs")
        w4 = _wave(v5_4)
        tl.store(Y + row * N + lane + 128, w4 * SCALE, cache_modifier=".cs")
        w5 = _wave(v5_5)
        tl.store(Y + row * N + lane + 160, w5 * SCALE, cache_modifier=".cs")
        w6 = _wave(v5_6)
        tl.store(Y + row * N + lane + 192, w6 * SCALE, cache_modifier=".cs")
        w7 = _wave(v5_7)
        tl.store(Y + row * N + lane + 224, w7 * SCALE, cache_modifier=".cs")
        w8 = _wave(v5_8)
        tl.store(Y + row * N + lane + 256, w8 * SCALE, cache_modifier=".cs")
        w9 = _wave(v5_9)
        tl.store(Y + row * N + lane + 288, w9 * SCALE, cache_modifier=".cs")
        w10 = _wave(v5_10)
        tl.store(Y + row * N + lane + 320, w10 * SCALE, cache_modifier=".cs")
        w11 = _wave(v5_11)
        tl.store(Y + row * N + lane + 352, w11 * SCALE, cache_modifier=".cs")
        w12 = _wave(v5_12)
        tl.store(Y + row * N + lane + 384, w12 * SCALE, cache_modifier=".cs")
        w13 = _wave(v5_13)
        tl.store(Y + row * N + lane + 416, w13 * SCALE, cache_modifier=".cs")
        w14 = _wave(v5_14)
        tl.store(Y + row * N + lane + 448, w14 * SCALE, cache_modifier=".cs")
        w15 = _wave(v5_15)
        tl.store(Y + row * N + lane + 480, w15 * SCALE, cache_modifier=".cs")
        w16 = _wave(v5_16)
        tl.store(Y + row * N + lane + 512, w16 * SCALE, cache_modifier=".cs")
        w17 = _wave(v5_17)
        tl.store(Y + row * N + lane + 544, w17 * SCALE, cache_modifier=".cs")
        w18 = _wave(v5_18)
        tl.store(Y + row * N + lane + 576, w18 * SCALE, cache_modifier=".cs")
        w19 = _wave(v5_19)
        tl.store(Y + row * N + lane + 608, w19 * SCALE, cache_modifier=".cs")
        w20 = _wave(v5_20)
        tl.store(Y + row * N + lane + 640, w20 * SCALE, cache_modifier=".cs")
        w21 = _wave(v5_21)
        tl.store(Y + row * N + lane + 672, w21 * SCALE, cache_modifier=".cs")
        w22 = _wave(v5_22)
        tl.store(Y + row * N + lane + 704, w22 * SCALE, cache_modifier=".cs")
        w23 = _wave(v5_23)
        tl.store(Y + row * N + lane + 736, w23 * SCALE, cache_modifier=".cs")
        w24 = _wave(v5_24)
        tl.store(Y + row * N + lane + 768, w24 * SCALE, cache_modifier=".cs")
        w25 = _wave(v5_25)
        tl.store(Y + row * N + lane + 800, w25 * SCALE, cache_modifier=".cs")
        w26 = _wave(v5_26)
        tl.store(Y + row * N + lane + 832, w26 * SCALE, cache_modifier=".cs")
        w27 = _wave(v5_27)
        tl.store(Y + row * N + lane + 864, w27 * SCALE, cache_modifier=".cs")
        w28 = _wave(v5_28)
        tl.store(Y + row * N + lane + 896, w28 * SCALE, cache_modifier=".cs")
        w29 = _wave(v5_29)
        tl.store(Y + row * N + lane + 928, w29 * SCALE, cache_modifier=".cs")
        w30 = _wave(v5_30)
        tl.store(Y + row * N + lane + 960, w30 * SCALE, cache_modifier=".cs")
        w31 = _wave(v5_31)
        tl.store(Y + row * N + lane + 992, w31 * SCALE, cache_modifier=".cs")
    elif N == 2048:
        v0_0 = tl.load(X + row * N + lane + 0, cache_modifier=".cg").to(
            tl.float32
        )
        v0_1 = tl.load(X + row * N + lane + 32, cache_modifier=".cg").to(
            tl.float32
        )
        v0_2 = tl.load(X + row * N + lane + 64, cache_modifier=".cg").to(
            tl.float32
        )
        v0_3 = tl.load(X + row * N + lane + 96, cache_modifier=".cg").to(
            tl.float32
        )
        v0_4 = tl.load(X + row * N + lane + 128, cache_modifier=".cg").to(
            tl.float32
        )
        v0_5 = tl.load(X + row * N + lane + 160, cache_modifier=".cg").to(
            tl.float32
        )
        v0_6 = tl.load(X + row * N + lane + 192, cache_modifier=".cg").to(
            tl.float32
        )
        v0_7 = tl.load(X + row * N + lane + 224, cache_modifier=".cg").to(
            tl.float32
        )
        v0_8 = tl.load(X + row * N + lane + 256, cache_modifier=".cg").to(
            tl.float32
        )
        v0_9 = tl.load(X + row * N + lane + 288, cache_modifier=".cg").to(
            tl.float32
        )
        v0_10 = tl.load(X + row * N + lane + 320, cache_modifier=".cg").to(
            tl.float32
        )
        v0_11 = tl.load(X + row * N + lane + 352, cache_modifier=".cg").to(
            tl.float32
        )
        v0_12 = tl.load(X + row * N + lane + 384, cache_modifier=".cg").to(
            tl.float32
        )
        v0_13 = tl.load(X + row * N + lane + 416, cache_modifier=".cg").to(
            tl.float32
        )
        v0_14 = tl.load(X + row * N + lane + 448, cache_modifier=".cg").to(
            tl.float32
        )
        v0_15 = tl.load(X + row * N + lane + 480, cache_modifier=".cg").to(
            tl.float32
        )
        v0_16 = tl.load(X + row * N + lane + 512, cache_modifier=".cg").to(
            tl.float32
        )
        v0_17 = tl.load(X + row * N + lane + 544, cache_modifier=".cg").to(
            tl.float32
        )
        v0_18 = tl.load(X + row * N + lane + 576, cache_modifier=".cg").to(
            tl.float32
        )
        v0_19 = tl.load(X + row * N + lane + 608, cache_modifier=".cg").to(
            tl.float32
        )
        v0_20 = tl.load(X + row * N + lane + 640, cache_modifier=".cg").to(
            tl.float32
        )
        v0_21 = tl.load(X + row * N + lane + 672, cache_modifier=".cg").to(
            tl.float32
        )
        v0_22 = tl.load(X + row * N + lane + 704, cache_modifier=".cg").to(
            tl.float32
        )
        v0_23 = tl.load(X + row * N + lane + 736, cache_modifier=".cg").to(
            tl.float32
        )
        v0_24 = tl.load(X + row * N + lane + 768, cache_modifier=".cg").to(
            tl.float32
        )
        v0_25 = tl.load(X + row * N + lane + 800, cache_modifier=".cg").to(
            tl.float32
        )
        v0_26 = tl.load(X + row * N + lane + 832, cache_modifier=".cg").to(
            tl.float32
        )
        v0_27 = tl.load(X + row * N + lane + 864, cache_modifier=".cg").to(
            tl.float32
        )
        v0_28 = tl.load(X + row * N + lane + 896, cache_modifier=".cg").to(
            tl.float32
        )
        v0_29 = tl.load(X + row * N + lane + 928, cache_modifier=".cg").to(
            tl.float32
        )
        v0_30 = tl.load(X + row * N + lane + 960, cache_modifier=".cg").to(
            tl.float32
        )
        v0_31 = tl.load(X + row * N + lane + 992, cache_modifier=".cg").to(
            tl.float32
        )
        v0_32 = tl.load(X + row * N + lane + 1024, cache_modifier=".cg").to(
            tl.float32
        )
        v0_33 = tl.load(X + row * N + lane + 1056, cache_modifier=".cg").to(
            tl.float32
        )
        v0_34 = tl.load(X + row * N + lane + 1088, cache_modifier=".cg").to(
            tl.float32
        )
        v0_35 = tl.load(X + row * N + lane + 1120, cache_modifier=".cg").to(
            tl.float32
        )
        v0_36 = tl.load(X + row * N + lane + 1152, cache_modifier=".cg").to(
            tl.float32
        )
        v0_37 = tl.load(X + row * N + lane + 1184, cache_modifier=".cg").to(
            tl.float32
        )
        v0_38 = tl.load(X + row * N + lane + 1216, cache_modifier=".cg").to(
            tl.float32
        )
        v0_39 = tl.load(X + row * N + lane + 1248, cache_modifier=".cg").to(
            tl.float32
        )
        v0_40 = tl.load(X + row * N + lane + 1280, cache_modifier=".cg").to(
            tl.float32
        )
        v0_41 = tl.load(X + row * N + lane + 1312, cache_modifier=".cg").to(
            tl.float32
        )
        v0_42 = tl.load(X + row * N + lane + 1344, cache_modifier=".cg").to(
            tl.float32
        )
        v0_43 = tl.load(X + row * N + lane + 1376, cache_modifier=".cg").to(
            tl.float32
        )
        v0_44 = tl.load(X + row * N + lane + 1408, cache_modifier=".cg").to(
            tl.float32
        )
        v0_45 = tl.load(X + row * N + lane + 1440, cache_modifier=".cg").to(
            tl.float32
        )
        v0_46 = tl.load(X + row * N + lane + 1472, cache_modifier=".cg").to(
            tl.float32
        )
        v0_47 = tl.load(X + row * N + lane + 1504, cache_modifier=".cg").to(
            tl.float32
        )
        v0_48 = tl.load(X + row * N + lane + 1536, cache_modifier=".cg").to(
            tl.float32
        )
        v0_49 = tl.load(X + row * N + lane + 1568, cache_modifier=".cg").to(
            tl.float32
        )
        v0_50 = tl.load(X + row * N + lane + 1600, cache_modifier=".cg").to(
            tl.float32
        )
        v0_51 = tl.load(X + row * N + lane + 1632, cache_modifier=".cg").to(
            tl.float32
        )
        v0_52 = tl.load(X + row * N + lane + 1664, cache_modifier=".cg").to(
            tl.float32
        )
        v0_53 = tl.load(X + row * N + lane + 1696, cache_modifier=".cg").to(
            tl.float32
        )
        v0_54 = tl.load(X + row * N + lane + 1728, cache_modifier=".cg").to(
            tl.float32
        )
        v0_55 = tl.load(X + row * N + lane + 1760, cache_modifier=".cg").to(
            tl.float32
        )
        v0_56 = tl.load(X + row * N + lane + 1792, cache_modifier=".cg").to(
            tl.float32
        )
        v0_57 = tl.load(X + row * N + lane + 1824, cache_modifier=".cg").to(
            tl.float32
        )
        v0_58 = tl.load(X + row * N + lane + 1856, cache_modifier=".cg").to(
            tl.float32
        )
        v0_59 = tl.load(X + row * N + lane + 1888, cache_modifier=".cg").to(
            tl.float32
        )
        v0_60 = tl.load(X + row * N + lane + 1920, cache_modifier=".cg").to(
            tl.float32
        )
        v0_61 = tl.load(X + row * N + lane + 1952, cache_modifier=".cg").to(
            tl.float32
        )
        v0_62 = tl.load(X + row * N + lane + 1984, cache_modifier=".cg").to(
            tl.float32
        )
        v0_63 = tl.load(X + row * N + lane + 2016, cache_modifier=".cg").to(
            tl.float32
        )
        v1_0 = v0_0 + v0_32
        v1_1 = v0_1 + v0_33
        v1_2 = v0_2 + v0_34
        v1_3 = v0_3 + v0_35
        v1_4 = v0_4 + v0_36
        v1_5 = v0_5 + v0_37
        v1_6 = v0_6 + v0_38
        v1_7 = v0_7 + v0_39
        v1_8 = v0_8 + v0_40
        v1_9 = v0_9 + v0_41
        v1_10 = v0_10 + v0_42
        v1_11 = v0_11 + v0_43
        v1_12 = v0_12 + v0_44
        v1_13 = v0_13 + v0_45
        v1_14 = v0_14 + v0_46
        v1_15 = v0_15 + v0_47
        v1_16 = v0_16 + v0_48
        v1_17 = v0_17 + v0_49
        v1_18 = v0_18 + v0_50
        v1_19 = v0_19 + v0_51
        v1_20 = v0_20 + v0_52
        v1_21 = v0_21 + v0_53
        v1_22 = v0_22 + v0_54
        v1_23 = v0_23 + v0_55
        v1_24 = v0_24 + v0_56
        v1_25 = v0_25 + v0_57
        v1_26 = v0_26 + v0_58
        v1_27 = v0_27 + v0_59
        v1_28 = v0_28 + v0_60
        v1_29 = v0_29 + v0_61
        v1_30 = v0_30 + v0_62
        v1_31 = v0_31 + v0_63
        v1_32 = v0_0 - v0_32
        v1_33 = v0_1 - v0_33
        v1_34 = v0_2 - v0_34
        v1_35 = v0_3 - v0_35
        v1_36 = v0_4 - v0_36
        v1_37 = v0_5 - v0_37
        v1_38 = v0_6 - v0_38
        v1_39 = v0_7 - v0_39
        v1_40 = v0_8 - v0_40
        v1_41 = v0_9 - v0_41
        v1_42 = v0_10 - v0_42
        v1_43 = v0_11 - v0_43
        v1_44 = v0_12 - v0_44
        v1_45 = v0_13 - v0_45
        v1_46 = v0_14 - v0_46
        v1_47 = v0_15 - v0_47
        v1_48 = v0_16 - v0_48
        v1_49 = v0_17 - v0_49
        v1_50 = v0_18 - v0_50
        v1_51 = v0_19 - v0_51
        v1_52 = v0_20 - v0_52
        v1_53 = v0_21 - v0_53
        v1_54 = v0_22 - v0_54
        v1_55 = v0_23 - v0_55
        v1_56 = v0_24 - v0_56
        v1_57 = v0_25 - v0_57
        v1_58 = v0_26 - v0_58
        v1_59 = v0_27 - v0_59
        v1_60 = v0_28 - v0_60
        v1_61 = v0_29 - v0_61
        v1_62 = v0_30 - v0_62
        v1_63 = v0_31 - v0_63
        v2_0 = v1_0 + v1_16
        v2_1 = v1_1 + v1_17
        v2_2 = v1_2 + v1_18
        v2_3 = v1_3 + v1_19
        v2_4 = v1_4 + v1_20
        v2_5 = v1_5 + v1_21
        v2_6 = v1_6 + v1_22
        v2_7 = v1_7 + v1_23
        v2_8 = v1_8 + v1_24
        v2_9 = v1_9 + v1_25
        v2_10 = v1_10 + v1_26
        v2_11 = v1_11 + v1_27
        v2_12 = v1_12 + v1_28
        v2_13 = v1_13 + v1_29
        v2_14 = v1_14 + v1_30
        v2_15 = v1_15 + v1_31
        v2_16 = v1_0 - v1_16
        v2_17 = v1_1 - v1_17
        v2_18 = v1_2 - v1_18
        v2_19 = v1_3 - v1_19
        v2_20 = v1_4 - v1_20
        v2_21 = v1_5 - v1_21
        v2_22 = v1_6 - v1_22
        v2_23 = v1_7 - v1_23
        v2_24 = v1_8 - v1_24
        v2_25 = v1_9 - v1_25
        v2_26 = v1_10 - v1_26
        v2_27 = v1_11 - v1_27
        v2_28 = v1_12 - v1_28
        v2_29 = v1_13 - v1_29
        v2_30 = v1_14 - v1_30
        v2_31 = v1_15 - v1_31
        v2_32 = v1_32 + v1_48
        v2_33 = v1_33 + v1_49
        v2_34 = v1_34 + v1_50
        v2_35 = v1_35 + v1_51
        v2_36 = v1_36 + v1_52
        v2_37 = v1_37 + v1_53
        v2_38 = v1_38 + v1_54
        v2_39 = v1_39 + v1_55
        v2_40 = v1_40 + v1_56
        v2_41 = v1_41 + v1_57
        v2_42 = v1_42 + v1_58
        v2_43 = v1_43 + v1_59
        v2_44 = v1_44 + v1_60
        v2_45 = v1_45 + v1_61
        v2_46 = v1_46 + v1_62
        v2_47 = v1_47 + v1_63
        v2_48 = v1_32 - v1_48
        v2_49 = v1_33 - v1_49
        v2_50 = v1_34 - v1_50
        v2_51 = v1_35 - v1_51
        v2_52 = v1_36 - v1_52
        v2_53 = v1_37 - v1_53
        v2_54 = v1_38 - v1_54
        v2_55 = v1_39 - v1_55
        v2_56 = v1_40 - v1_56
        v2_57 = v1_41 - v1_57
        v2_58 = v1_42 - v1_58
        v2_59 = v1_43 - v1_59
        v2_60 = v1_44 - v1_60
        v2_61 = v1_45 - v1_61
        v2_62 = v1_46 - v1_62
        v2_63 = v1_47 - v1_63
        v3_0 = v2_0 + v2_8
        v3_1 = v2_1 + v2_9
        v3_2 = v2_2 + v2_10
        v3_3 = v2_3 + v2_11
        v3_4 = v2_4 + v2_12
        v3_5 = v2_5 + v2_13
        v3_6 = v2_6 + v2_14
        v3_7 = v2_7 + v2_15
        v3_8 = v2_0 - v2_8
        v3_9 = v2_1 - v2_9
        v3_10 = v2_2 - v2_10
        v3_11 = v2_3 - v2_11
        v3_12 = v2_4 - v2_12
        v3_13 = v2_5 - v2_13
        v3_14 = v2_6 - v2_14
        v3_15 = v2_7 - v2_15
        v3_16 = v2_16 + v2_24
        v3_17 = v2_17 + v2_25
        v3_18 = v2_18 + v2_26
        v3_19 = v2_19 + v2_27
        v3_20 = v2_20 + v2_28
        v3_21 = v2_21 + v2_29
        v3_22 = v2_22 + v2_30
        v3_23 = v2_23 + v2_31
        v3_24 = v2_16 - v2_24
        v3_25 = v2_17 - v2_25
        v3_26 = v2_18 - v2_26
        v3_27 = v2_19 - v2_27
        v3_28 = v2_20 - v2_28
        v3_29 = v2_21 - v2_29
        v3_30 = v2_22 - v2_30
        v3_31 = v2_23 - v2_31
        v3_32 = v2_32 + v2_40
        v3_33 = v2_33 + v2_41
        v3_34 = v2_34 + v2_42
        v3_35 = v2_35 + v2_43
        v3_36 = v2_36 + v2_44
        v3_37 = v2_37 + v2_45
        v3_38 = v2_38 + v2_46
        v3_39 = v2_39 + v2_47
        v3_40 = v2_32 - v2_40
        v3_41 = v2_33 - v2_41
        v3_42 = v2_34 - v2_42
        v3_43 = v2_35 - v2_43
        v3_44 = v2_36 - v2_44
        v3_45 = v2_37 - v2_45
        v3_46 = v2_38 - v2_46
        v3_47 = v2_39 - v2_47
        v3_48 = v2_48 + v2_56
        v3_49 = v2_49 + v2_57
        v3_50 = v2_50 + v2_58
        v3_51 = v2_51 + v2_59
        v3_52 = v2_52 + v2_60
        v3_53 = v2_53 + v2_61
        v3_54 = v2_54 + v2_62
        v3_55 = v2_55 + v2_63
        v3_56 = v2_48 - v2_56
        v3_57 = v2_49 - v2_57
        v3_58 = v2_50 - v2_58
        v3_59 = v2_51 - v2_59
        v3_60 = v2_52 - v2_60
        v3_61 = v2_53 - v2_61
        v3_62 = v2_54 - v2_62
        v3_63 = v2_55 - v2_63
        v4_0 = v3_0 + v3_4
        v4_1 = v3_1 + v3_5
        v4_2 = v3_2 + v3_6
        v4_3 = v3_3 + v3_7
        v4_4 = v3_0 - v3_4
        v4_5 = v3_1 - v3_5
        v4_6 = v3_2 - v3_6
        v4_7 = v3_3 - v3_7
        v4_8 = v3_8 + v3_12
        v4_9 = v3_9 + v3_13
        v4_10 = v3_10 + v3_14
        v4_11 = v3_11 + v3_15
        v4_12 = v3_8 - v3_12
        v4_13 = v3_9 - v3_13
        v4_14 = v3_10 - v3_14
        v4_15 = v3_11 - v3_15
        v4_16 = v3_16 + v3_20
        v4_17 = v3_17 + v3_21
        v4_18 = v3_18 + v3_22
        v4_19 = v3_19 + v3_23
        v4_20 = v3_16 - v3_20
        v4_21 = v3_17 - v3_21
        v4_22 = v3_18 - v3_22
        v4_23 = v3_19 - v3_23
        v4_24 = v3_24 + v3_28
        v4_25 = v3_25 + v3_29
        v4_26 = v3_26 + v3_30
        v4_27 = v3_27 + v3_31
        v4_28 = v3_24 - v3_28
        v4_29 = v3_25 - v3_29
        v4_30 = v3_26 - v3_30
        v4_31 = v3_27 - v3_31
        v4_32 = v3_32 + v3_36
        v4_33 = v3_33 + v3_37
        v4_34 = v3_34 + v3_38
        v4_35 = v3_35 + v3_39
        v4_36 = v3_32 - v3_36
        v4_37 = v3_33 - v3_37
        v4_38 = v3_34 - v3_38
        v4_39 = v3_35 - v3_39
        v4_40 = v3_40 + v3_44
        v4_41 = v3_41 + v3_45
        v4_42 = v3_42 + v3_46
        v4_43 = v3_43 + v3_47
        v4_44 = v3_40 - v3_44
        v4_45 = v3_41 - v3_45
        v4_46 = v3_42 - v3_46
        v4_47 = v3_43 - v3_47
        v4_48 = v3_48 + v3_52
        v4_49 = v3_49 + v3_53
        v4_50 = v3_50 + v3_54
        v4_51 = v3_51 + v3_55
        v4_52 = v3_48 - v3_52
        v4_53 = v3_49 - v3_53
        v4_54 = v3_50 - v3_54
        v4_55 = v3_51 - v3_55
        v4_56 = v3_56 + v3_60
        v4_57 = v3_57 + v3_61
        v4_58 = v3_58 + v3_62
        v4_59 = v3_59 + v3_63
        v4_60 = v3_56 - v3_60
        v4_61 = v3_57 - v3_61
        v4_62 = v3_58 - v3_62
        v4_63 = v3_59 - v3_63
        v5_0 = v4_0 + v4_2
        v5_1 = v4_1 + v4_3
        v5_2 = v4_0 - v4_2
        v5_3 = v4_1 - v4_3
        v5_4 = v4_4 + v4_6
        v5_5 = v4_5 + v4_7
        v5_6 = v4_4 - v4_6
        v5_7 = v4_5 - v4_7
        v5_8 = v4_8 + v4_10
        v5_9 = v4_9 + v4_11
        v5_10 = v4_8 - v4_10
        v5_11 = v4_9 - v4_11
        v5_12 = v4_12 + v4_14
        v5_13 = v4_13 + v4_15
        v5_14 = v4_12 - v4_14
        v5_15 = v4_13 - v4_15
        v5_16 = v4_16 + v4_18
        v5_17 = v4_17 + v4_19
        v5_18 = v4_16 - v4_18
        v5_19 = v4_17 - v4_19
        v5_20 = v4_20 + v4_22
        v5_21 = v4_21 + v4_23
        v5_22 = v4_20 - v4_22
        v5_23 = v4_21 - v4_23
        v5_24 = v4_24 + v4_26
        v5_25 = v4_25 + v4_27
        v5_26 = v4_24 - v4_26
        v5_27 = v4_25 - v4_27
        v5_28 = v4_28 + v4_30
        v5_29 = v4_29 + v4_31
        v5_30 = v4_28 - v4_30
        v5_31 = v4_29 - v4_31
        v5_32 = v4_32 + v4_34
        v5_33 = v4_33 + v4_35
        v5_34 = v4_32 - v4_34
        v5_35 = v4_33 - v4_35
        v5_36 = v4_36 + v4_38
        v5_37 = v4_37 + v4_39
        v5_38 = v4_36 - v4_38
        v5_39 = v4_37 - v4_39
        v5_40 = v4_40 + v4_42
        v5_41 = v4_41 + v4_43
        v5_42 = v4_40 - v4_42
        v5_43 = v4_41 - v4_43
        v5_44 = v4_44 + v4_46
        v5_45 = v4_45 + v4_47
        v5_46 = v4_44 - v4_46
        v5_47 = v4_45 - v4_47
        v5_48 = v4_48 + v4_50
        v5_49 = v4_49 + v4_51
        v5_50 = v4_48 - v4_50
        v5_51 = v4_49 - v4_51
        v5_52 = v4_52 + v4_54
        v5_53 = v4_53 + v4_55
        v5_54 = v4_52 - v4_54
        v5_55 = v4_53 - v4_55
        v5_56 = v4_56 + v4_58
        v5_57 = v4_57 + v4_59
        v5_58 = v4_56 - v4_58
        v5_59 = v4_57 - v4_59
        v5_60 = v4_60 + v4_62
        v5_61 = v4_61 + v4_63
        v5_62 = v4_60 - v4_62
        v5_63 = v4_61 - v4_63
        v6_0 = v5_0 + v5_1
        v6_1 = v5_0 - v5_1
        v6_2 = v5_2 + v5_3
        v6_3 = v5_2 - v5_3
        v6_4 = v5_4 + v5_5
        v6_5 = v5_4 - v5_5
        v6_6 = v5_6 + v5_7
        v6_7 = v5_6 - v5_7
        v6_8 = v5_8 + v5_9
        v6_9 = v5_8 - v5_9
        v6_10 = v5_10 + v5_11
        v6_11 = v5_10 - v5_11
        v6_12 = v5_12 + v5_13
        v6_13 = v5_12 - v5_13
        v6_14 = v5_14 + v5_15
        v6_15 = v5_14 - v5_15
        v6_16 = v5_16 + v5_17
        v6_17 = v5_16 - v5_17
        v6_18 = v5_18 + v5_19
        v6_19 = v5_18 - v5_19
        v6_20 = v5_20 + v5_21
        v6_21 = v5_20 - v5_21
        v6_22 = v5_22 + v5_23
        v6_23 = v5_22 - v5_23
        v6_24 = v5_24 + v5_25
        v6_25 = v5_24 - v5_25
        v6_26 = v5_26 + v5_27
        v6_27 = v5_26 - v5_27
        v6_28 = v5_28 + v5_29
        v6_29 = v5_28 - v5_29
        v6_30 = v5_30 + v5_31
        v6_31 = v5_30 - v5_31
        v6_32 = v5_32 + v5_33
        v6_33 = v5_32 - v5_33
        v6_34 = v5_34 + v5_35
        v6_35 = v5_34 - v5_35
        v6_36 = v5_36 + v5_37
        v6_37 = v5_36 - v5_37
        v6_38 = v5_38 + v5_39
        v6_39 = v5_38 - v5_39
        v6_40 = v5_40 + v5_41
        v6_41 = v5_40 - v5_41
        v6_42 = v5_42 + v5_43
        v6_43 = v5_42 - v5_43
        v6_44 = v5_44 + v5_45
        v6_45 = v5_44 - v5_45
        v6_46 = v5_46 + v5_47
        v6_47 = v5_46 - v5_47
        v6_48 = v5_48 + v5_49
        v6_49 = v5_48 - v5_49
        v6_50 = v5_50 + v5_51
        v6_51 = v5_50 - v5_51
        v6_52 = v5_52 + v5_53
        v6_53 = v5_52 - v5_53
        v6_54 = v5_54 + v5_55
        v6_55 = v5_54 - v5_55
        v6_56 = v5_56 + v5_57
        v6_57 = v5_56 - v5_57
        v6_58 = v5_58 + v5_59
        v6_59 = v5_58 - v5_59
        v6_60 = v5_60 + v5_61
        v6_61 = v5_60 - v5_61
        v6_62 = v5_62 + v5_63
        v6_63 = v5_62 - v5_63
        w0 = _wave(v6_0)
        tl.store(Y + row * N + lane + 0, w0 * SCALE, cache_modifier=".cs")
        w1 = _wave(v6_1)
        tl.store(Y + row * N + lane + 32, w1 * SCALE, cache_modifier=".cs")
        w2 = _wave(v6_2)
        tl.store(Y + row * N + lane + 64, w2 * SCALE, cache_modifier=".cs")
        w3 = _wave(v6_3)
        tl.store(Y + row * N + lane + 96, w3 * SCALE, cache_modifier=".cs")
        w4 = _wave(v6_4)
        tl.store(Y + row * N + lane + 128, w4 * SCALE, cache_modifier=".cs")
        w5 = _wave(v6_5)
        tl.store(Y + row * N + lane + 160, w5 * SCALE, cache_modifier=".cs")
        w6 = _wave(v6_6)
        tl.store(Y + row * N + lane + 192, w6 * SCALE, cache_modifier=".cs")
        w7 = _wave(v6_7)
        tl.store(Y + row * N + lane + 224, w7 * SCALE, cache_modifier=".cs")
        w8 = _wave(v6_8)
        tl.store(Y + row * N + lane + 256, w8 * SCALE, cache_modifier=".cs")
        w9 = _wave(v6_9)
        tl.store(Y + row * N + lane + 288, w9 * SCALE, cache_modifier=".cs")
        w10 = _wave(v6_10)
        tl.store(Y + row * N + lane + 320, w10 * SCALE, cache_modifier=".cs")
        w11 = _wave(v6_11)
        tl.store(Y + row * N + lane + 352, w11 * SCALE, cache_modifier=".cs")
        w12 = _wave(v6_12)
        tl.store(Y + row * N + lane + 384, w12 * SCALE, cache_modifier=".cs")
        w13 = _wave(v6_13)
        tl.store(Y + row * N + lane + 416, w13 * SCALE, cache_modifier=".cs")
        w14 = _wave(v6_14)
        tl.store(Y + row * N + lane + 448, w14 * SCALE, cache_modifier=".cs")
        w15 = _wave(v6_15)
        tl.store(Y + row * N + lane + 480, w15 * SCALE, cache_modifier=".cs")
        w16 = _wave(v6_16)
        tl.store(Y + row * N + lane + 512, w16 * SCALE, cache_modifier=".cs")
        w17 = _wave(v6_17)
        tl.store(Y + row * N + lane + 544, w17 * SCALE, cache_modifier=".cs")
        w18 = _wave(v6_18)
        tl.store(Y + row * N + lane + 576, w18 * SCALE, cache_modifier=".cs")
        w19 = _wave(v6_19)
        tl.store(Y + row * N + lane + 608, w19 * SCALE, cache_modifier=".cs")
        w20 = _wave(v6_20)
        tl.store(Y + row * N + lane + 640, w20 * SCALE, cache_modifier=".cs")
        w21 = _wave(v6_21)
        tl.store(Y + row * N + lane + 672, w21 * SCALE, cache_modifier=".cs")
        w22 = _wave(v6_22)
        tl.store(Y + row * N + lane + 704, w22 * SCALE, cache_modifier=".cs")
        w23 = _wave(v6_23)
        tl.store(Y + row * N + lane + 736, w23 * SCALE, cache_modifier=".cs")
        w24 = _wave(v6_24)
        tl.store(Y + row * N + lane + 768, w24 * SCALE, cache_modifier=".cs")
        w25 = _wave(v6_25)
        tl.store(Y + row * N + lane + 800, w25 * SCALE, cache_modifier=".cs")
        w26 = _wave(v6_26)
        tl.store(Y + row * N + lane + 832, w26 * SCALE, cache_modifier=".cs")
        w27 = _wave(v6_27)
        tl.store(Y + row * N + lane + 864, w27 * SCALE, cache_modifier=".cs")
        w28 = _wave(v6_28)
        tl.store(Y + row * N + lane + 896, w28 * SCALE, cache_modifier=".cs")
        w29 = _wave(v6_29)
        tl.store(Y + row * N + lane + 928, w29 * SCALE, cache_modifier=".cs")
        w30 = _wave(v6_30)
        tl.store(Y + row * N + lane + 960, w30 * SCALE, cache_modifier=".cs")
        w31 = _wave(v6_31)
        tl.store(Y + row * N + lane + 992, w31 * SCALE, cache_modifier=".cs")
        w32 = _wave(v6_32)
        tl.store(Y + row * N + lane + 1024, w32 * SCALE, cache_modifier=".cs")
        w33 = _wave(v6_33)
        tl.store(Y + row * N + lane + 1056, w33 * SCALE, cache_modifier=".cs")
        w34 = _wave(v6_34)
        tl.store(Y + row * N + lane + 1088, w34 * SCALE, cache_modifier=".cs")
        w35 = _wave(v6_35)
        tl.store(Y + row * N + lane + 1120, w35 * SCALE, cache_modifier=".cs")
        w36 = _wave(v6_36)
        tl.store(Y + row * N + lane + 1152, w36 * SCALE, cache_modifier=".cs")
        w37 = _wave(v6_37)
        tl.store(Y + row * N + lane + 1184, w37 * SCALE, cache_modifier=".cs")
        w38 = _wave(v6_38)
        tl.store(Y + row * N + lane + 1216, w38 * SCALE, cache_modifier=".cs")
        w39 = _wave(v6_39)
        tl.store(Y + row * N + lane + 1248, w39 * SCALE, cache_modifier=".cs")
        w40 = _wave(v6_40)
        tl.store(Y + row * N + lane + 1280, w40 * SCALE, cache_modifier=".cs")
        w41 = _wave(v6_41)
        tl.store(Y + row * N + lane + 1312, w41 * SCALE, cache_modifier=".cs")
        w42 = _wave(v6_42)
        tl.store(Y + row * N + lane + 1344, w42 * SCALE, cache_modifier=".cs")
        w43 = _wave(v6_43)
        tl.store(Y + row * N + lane + 1376, w43 * SCALE, cache_modifier=".cs")
        w44 = _wave(v6_44)
        tl.store(Y + row * N + lane + 1408, w44 * SCALE, cache_modifier=".cs")
        w45 = _wave(v6_45)
        tl.store(Y + row * N + lane + 1440, w45 * SCALE, cache_modifier=".cs")
        w46 = _wave(v6_46)
        tl.store(Y + row * N + lane + 1472, w46 * SCALE, cache_modifier=".cs")
        w47 = _wave(v6_47)
        tl.store(Y + row * N + lane + 1504, w47 * SCALE, cache_modifier=".cs")
        w48 = _wave(v6_48)
        tl.store(Y + row * N + lane + 1536, w48 * SCALE, cache_modifier=".cs")
        w49 = _wave(v6_49)
        tl.store(Y + row * N + lane + 1568, w49 * SCALE, cache_modifier=".cs")
        w50 = _wave(v6_50)
        tl.store(Y + row * N + lane + 1600, w50 * SCALE, cache_modifier=".cs")
        w51 = _wave(v6_51)
        tl.store(Y + row * N + lane + 1632, w51 * SCALE, cache_modifier=".cs")
        w52 = _wave(v6_52)
        tl.store(Y + row * N + lane + 1664, w52 * SCALE, cache_modifier=".cs")
        w53 = _wave(v6_53)
        tl.store(Y + row * N + lane + 1696, w53 * SCALE, cache_modifier=".cs")
        w54 = _wave(v6_54)
        tl.store(Y + row * N + lane + 1728, w54 * SCALE, cache_modifier=".cs")
        w55 = _wave(v6_55)
        tl.store(Y + row * N + lane + 1760, w55 * SCALE, cache_modifier=".cs")
        w56 = _wave(v6_56)
        tl.store(Y + row * N + lane + 1792, w56 * SCALE, cache_modifier=".cs")
        w57 = _wave(v6_57)
        tl.store(Y + row * N + lane + 1824, w57 * SCALE, cache_modifier=".cs")
        w58 = _wave(v6_58)
        tl.store(Y + row * N + lane + 1856, w58 * SCALE, cache_modifier=".cs")
        w59 = _wave(v6_59)
        tl.store(Y + row * N + lane + 1888, w59 * SCALE, cache_modifier=".cs")
        w60 = _wave(v6_60)
        tl.store(Y + row * N + lane + 1920, w60 * SCALE, cache_modifier=".cs")
        w61 = _wave(v6_61)
        tl.store(Y + row * N + lane + 1952, w61 * SCALE, cache_modifier=".cs")
        w62 = _wave(v6_62)
        tl.store(Y + row * N + lane + 1984, w62 * SCALE, cache_modifier=".cs")
        w63 = _wave(v6_63)
        tl.store(Y + row * N + lane + 2016, w63 * SCALE, cache_modifier=".cs")


def _rocm_warps(m, n):
    if n == 128:
        return 2 if m == 1 or m == 8 else 1
    if n == 256:
        return 4
    if n == 512:
        if m <= 64:
            return 8
        return 1
    if n == 1024:
        return 8
    if n == 2048:
        if m >= 4096:
            return 1
        return 4 if m == 512 else 8
    return 4


@triton.jit
def _tuple_step(values, STEP: tl.constexpr):
    result = ()
    for plane in tl.static_range(len(values)):
        if plane & STEP:
            result += (values[plane & ~STEP] - values[plane | STEP],)
        else:
            result += (values[plane & ~STEP] + values[plane | STEP],)
    return result


@triton.jit
def _compact_wave_step(v, W: tl.constexpr, S: tl.constexpr):
    p = tl.reshape(tl.flip(tl.reshape(v, (W // (2 * S), 2, S)), 1), (W,))
    sign = tl.where((tl.arange(0, W) & S) != 0, -1.0, 1.0)
    return tl.fma(sign, v, p)


@triton.jit
def _compact_planes(
    X, Y, N: tl.constexpr, SCALE: tl.constexpr, W: tl.constexpr
):
    row = tl.program_id(0)
    lane = tl.arange(0, W)
    values = ()
    for plane in tl.static_range(N // W):
        values += (
            tl.load(X + row * N + plane * W + lane, cache_modifier=".cg").to(
                tl.float32
            ),
        )
    for stage in tl.static_range((N // W).bit_length() - 1):
        values = _tuple_step(values, (N // W) >> (stage + 1))
    for plane in tl.static_range(N // W):
        v = values[plane]
        for stage in tl.static_range(W.bit_length() - 1):
            v = _compact_wave_step(v, W, W >> (stage + 1))
        tl.store(
            Y + row * N + plane * W + lane, v * SCALE, cache_modifier=".cs"
        )


@triton.jit
def _ordered_planes(
    X,
    Y,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    W: tl.constexpr,
    LOW_FIRST: tl.constexpr,
    ASCENDING: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, W)
    values = ()
    for plane in tl.static_range(N // W):
        v = tl.load(X + row * N + plane * W + lane, cache_modifier=".cg").to(
            tl.float32
        )
        if LOW_FIRST:
            for stage in tl.static_range(W.bit_length() - 1):
                v = _compact_wave_step(
                    v, W, (1 << stage) if ASCENDING else (W >> (stage + 1))
                )
        values += (v,)
    for stage in tl.static_range((N // W).bit_length() - 1):
        values = _tuple_step(
            values, (1 << stage) if ASCENDING else ((N // W) >> (stage + 1))
        )
    for plane in tl.static_range(N // W):
        v = values[plane]
        if not LOW_FIRST:
            for stage in tl.static_range(W.bit_length() - 1):
                v = _compact_wave_step(
                    v, W, (1 << stage) if ASCENDING else (W >> (stage + 1))
                )
        tl.store(
            Y + row * N + plane * W + lane, v * SCALE, cache_modifier=".cs"
        )


@triton.jit
def _packed_row(X, Y, CONFIG: tl.constexpr):
    _row(X, Y, CONFIG[0], CONFIG[1])


@triton.jit
def _packed_planes(X, Y, CONFIG: tl.constexpr):
    _planes(X, Y, CONFIG[0], CONFIG[1])


@triton.jit
def _packed_compact(X, Y, CONFIG: tl.constexpr):
    _compact_planes(X, Y, CONFIG[0], CONFIG[1], CONFIG[2])


@triton.jit
def _packed_ordered(X, Y, CONFIG: tl.constexpr):
    _ordered_planes(X, Y, CONFIG[0], CONFIG[1], 128, True, True)


def hadamard_transform(x, scale):
    n = x.shape[-1]
    size = x.numel()
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    if size:
        m = size // n
        if torch.version.hip is not None:
            if n == 2048 and m >= 4096:
                _packed_ordered.run(
                    x, out, (n, scale), grid=(m,), warmup=False, num_warps=1
                )
                return out
            if n == 128 and m >= 64:
                _packed_compact.run(
                    x,
                    out,
                    (n, scale, 128),
                    grid=(m,),
                    warmup=False,
                    num_warps=1,
                )
                return out
            if m >= 512 and n in (512, 1024, 2048):
                if n == 2048 or n == 512 and m >= 4096:
                    width = 128 if n == 2048 and m >= 4096 else 256
                    _packed_compact.run(
                        x,
                        out,
                        (n, scale, width),
                        grid=(m,),
                        warmup=False,
                        num_warps=1,
                    )
                else:
                    _packed_planes.run(
                        x,
                        out,
                        (n, scale),
                        grid=(m,),
                        warmup=False,
                        num_warps=1,
                    )
                return out
            warps = _rocm_warps(m, n)
            _packed_row.run(
                x, out, (n, scale), grid=(m,), warmup=False, num_warps=warps
            )
            return out
        else:
            warps = 1 if n <= 128 or size >= 1048576 else 2 if n <= 512 else 4
        _row.run(
            x,
            out,
            n,
            scale,
            grid=(m,),
            warmup=False,
            num_warps=warps,
            num_stages=1,
        )
    return out


__all__ = ["hadamard_transform"]
