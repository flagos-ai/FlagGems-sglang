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

import triton
import triton.experimental.tle.language as tle
import triton.language as tl


@triton.jit
def _shared_grid(
    X,
    Y,
    K: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    Q: tl.constexpr,
    PAD: tl.constexpr,
    GROUP: tl.constexpr,
    CP: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (D // BD * GROUP) * GROUP + pid % GROUP
    block = pid // GROUP % (D // BD)
    kr = tl.arange(0, K)
    j = tl.arange(0, BD * 2)
    c = block * BD + j % BD
    side = j // BD
    x = tl.load(
        X
        + b * K * 2 * D
        + kr[:, None] * 2 * D
        + side[None, :] * D
        + c[None, :],
        cache_modifier=CP,
    )
    buf = tle.gpu.alloc(
        [K, BD * 2 + PAD],
        dtype=X.dtype.element_ty,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    wp = tle.gpu.local_ptr(
        buf,
        (
            tl.broadcast_to(kr[:, None], (K, BD * 2)),
            tl.broadcast_to(j[None, :], (K, BD * 2)),
        ),
    )
    tl.store(wp, x)
    tl.debug_barrier()
    lanes = tl.arange(0, Q)
    c = tl.arange(0, BD)
    m = tl.full((Q, BD), float("-inf"), tl.float32)
    for i in tl.static_range(K // Q):
        rows = tl.broadcast_to((i * Q + lanes)[:, None], (Q, BD))
        cols = tl.broadcast_to(c[None, :], (Q, BD))
        s = tl.load(tle.gpu.local_ptr(buf, (rows, cols + BD))).to(tl.float32)
        m = tl.maximum(m, s)
    mx = tl.max(m, 0)
    a = tl.zeros((Q, BD), tl.float32)
    z = tl.zeros((Q, BD), tl.float32)
    for i in tl.static_range(K // Q):
        rows = tl.broadcast_to((i * Q + lanes)[:, None], (Q, BD))
        cols = tl.broadcast_to(c[None, :], (Q, BD))
        s = tl.load(tle.gpu.local_ptr(buf, (rows, cols + BD))).to(tl.float32)
        v = tl.load(tle.gpu.local_ptr(buf, (rows, cols))).to(tl.float32)
        w = tl.exp(s - mx[None, :])
        a += v * w
        z += w
    tl.store(Y + b * D + block * BD + c, tl.sum(a, 0) / tl.sum(z, 0))


@triton.jit(do_not_specialize=["JOBS"])
def _small(
    X,
    Y,
    K: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TILES: tl.constexpr,
    JOBS,
):
    offs = tl.arange(0, BLOCK)
    stride_k = D * 2
    for job in tl.range(tl.program_id(0), JOBS, tl.num_programs(0)):
        b = job // TILES
        c = (job - b * TILES) * BLOCK + offs
        mask = c < D
        base = X + b * K * stride_k
        m = tl.full((BLOCK,), float("-inf"), tl.float32)
        for k in tl.static_range(K):
            s = tl.load(
                base + k * stride_k + D + c, mask=mask, other=float("-inf")
            ).to(tl.float32)
            m = tl.maximum(m, s)
        m = tl.where(m == float("-inf"), 0.0, m)
        acc = tl.zeros((BLOCK,), tl.float32)
        den = tl.zeros((BLOCK,), tl.float32)
        for k in tl.static_range(K):
            s = tl.load(
                base + k * stride_k + D + c, mask=mask, other=float("-inf")
            ).to(tl.float32)
            v = tl.load(base + k * stride_k + c, mask=mask, other=0.0).to(
                tl.float32
            )
            w = tl.exp(s - m)
            acc += v * w
            den += w
        tl.store(Y + b * D + c, acc / den, mask=mask)


@triton.jit(do_not_specialize=["K", "JOBS"])
def _large(
    X, Y, K, D: tl.constexpr, BLOCK: tl.constexpr, TILES: tl.constexpr, JOBS
):
    offs = tl.arange(0, BLOCK)
    stride_k = D * 2
    for job in tl.range(tl.program_id(0), JOBS, tl.num_programs(0)):
        b = job // TILES
        c = (job - b * TILES) * BLOCK + offs
        mask = c < D
        base = X + b * K * stride_k
        m = tl.full((BLOCK,), float("-inf"), tl.float32)
        acc = tl.zeros((BLOCK,), tl.float32)
        den = tl.zeros((BLOCK,), tl.float32)
        for k in range(K):
            s = tl.load(
                base + k * stride_k + D + c, mask=mask, other=float("-inf")
            ).to(tl.float32)
            v = tl.load(base + k * stride_k + c, mask=mask, other=0.0).to(
                tl.float32
            )
            m2 = tl.maximum(m, s)
            alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - m2))
            w = tl.where(s == float("-inf"), 0.0, tl.exp(s - m2))
            acc = acc * alpha + v * w
            den = den * alpha + w
            m = m2
        tl.store(Y + b * D + c, acc / den, mask=mask)


@triton.jit(do_not_specialize=["JOBS"])
def _plain(X, Y, K: tl.constexpr, D: tl.constexpr, JOBS):
    offs = tl.arange(0, D)
    stride_k = D * 2
    for b in tl.range(tl.program_id(0), JOBS, tl.num_programs(0)):
        base = X + b * K * stride_k
        m = tl.full((D,), float("-inf"), tl.float32)
        for k in tl.static_range(K):
            s = tl.load(base + k * stride_k + D + offs).to(tl.float32)
            m = tl.maximum(m, s)
        acc = tl.zeros((D,), tl.float32)
        den = tl.zeros((D,), tl.float32)
        for k in tl.static_range(K):
            s = tl.load(base + k * stride_k + D + offs).to(tl.float32)
            v = tl.load(base + k * stride_k + offs).to(tl.float32)
            w = tl.exp(s - m)
            acc += v * w
            den += w
        tl.store(Y + b * D + offs, acc / den)


@triton.jit
def _ilp(X, Y, K: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    b = tl.program_id(0)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    lanes = tl.arange(0, 2)
    base = X + b * K * 2 * D + c[None, :]
    m = tl.full((2, BD), float("-inf"), tl.float32)
    for i in tl.static_range(triton.cdiv(K, 2)):
        k = i * 2 + lanes
        mask = (k[:, None] < K) & (c[None, :] < D)
        s = tl.load(base + k[:, None] * 2 * D + D, mask, float("-inf")).to(
            tl.float32
        )
        m = tl.maximum(m, s)
    mx = tl.max(m, axis=0)
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    den = tl.zeros((2, BD), tl.float32)
    acc = tl.zeros((2, BD), tl.float32)
    for i in tl.static_range(triton.cdiv(K, 2)):
        k = i * 2 + lanes
        mask = (k[:, None] < K) & (c[None, :] < D)
        s = tl.load(base + k[:, None] * 2 * D + D, mask, float("-inf")).to(
            tl.float32
        )
        v = tl.load(base + k[:, None] * 2 * D, mask, 0.0).to(tl.float32)
        w = tl.exp(s - mx[None, :])
        acc += v * w
        den += w
    result = tl.sum(acc, axis=0) / tl.sum(den, axis=0)
    tl.store(Y + b * D + c, result, c < D)


def _block(k, d):
    if d <= 1:
        return 1
    if k <= 16 and d >= 512:
        return 512
    if k <= 16 and d >= 256:
        return 256
    if d >= 128:
        return 128
    return triton.next_power_of_2(d)


@triton.jit
def _entry4(X, Y, N):
    _plain(X, Y, 4, 128, N)


@triton.jit
def _entry16(X, Y, N):
    _plain(X, Y, 16, 128, N)


@triton.jit
def _entry64(X, Y):
    _ilp(X, Y, 64, 128, 128)


@triton.jit(do_not_specialize=["N"])
def _tree4(X, Y, N, D: tl.constexpr):
    c = tl.arange(0, D)
    for b in tl.range(tl.program_id(0), N, tl.num_programs(0)):
        base = X + b * 4 * 2 * D + c
        s0 = tl.load(base + 1 * D).to(tl.float32)
        s1 = tl.load(base + 3 * D).to(tl.float32)
        s2 = tl.load(base + 5 * D).to(tl.float32)
        s3 = tl.load(base + 7 * D).to(tl.float32)
        m0_0 = tl.maximum(s0, s1)
        m0_1 = tl.maximum(s2, s3)
        m1_0 = tl.maximum(m0_0, m0_1)
        m = tl.where(m1_0 == float("-inf"), 0.0, m1_0)
        w0 = tl.exp(s0 - m)
        a0 = tl.load(base + 0 * D).to(tl.float32) * w0
        w1 = tl.exp(s1 - m)
        a1 = tl.load(base + 2 * D).to(tl.float32) * w1
        w2 = tl.exp(s2 - m)
        a2 = tl.load(base + 4 * D).to(tl.float32) * w2
        w3 = tl.exp(s3 - m)
        a3 = tl.load(base + 6 * D).to(tl.float32) * w3
        w0_0 = w0 + w1
        w0_1 = w2 + w3
        w1_0 = w0_0 + w0_1
        wsum = w1_0
        a0_0 = a0 + a1
        a0_1 = a2 + a3
        a1_0 = a0_0 + a0_1
        asum = a1_0
        tl.store(Y + b * D + c, asum / wsum)


@triton.jit(do_not_specialize=["N"])
def _tree16(X, Y, N, D: tl.constexpr):
    c = tl.arange(0, D)
    for b in tl.range(tl.program_id(0), N, tl.num_programs(0)):
        base = X + b * 16 * 2 * D + c
        s0 = tl.load(base + 1 * D).to(tl.float32)
        s1 = tl.load(base + 3 * D).to(tl.float32)
        s2 = tl.load(base + 5 * D).to(tl.float32)
        s3 = tl.load(base + 7 * D).to(tl.float32)
        s4 = tl.load(base + 9 * D).to(tl.float32)
        s5 = tl.load(base + 11 * D).to(tl.float32)
        s6 = tl.load(base + 13 * D).to(tl.float32)
        s7 = tl.load(base + 15 * D).to(tl.float32)
        s8 = tl.load(base + 17 * D).to(tl.float32)
        s9 = tl.load(base + 19 * D).to(tl.float32)
        s10 = tl.load(base + 21 * D).to(tl.float32)
        s11 = tl.load(base + 23 * D).to(tl.float32)
        s12 = tl.load(base + 25 * D).to(tl.float32)
        s13 = tl.load(base + 27 * D).to(tl.float32)
        s14 = tl.load(base + 29 * D).to(tl.float32)
        s15 = tl.load(base + 31 * D).to(tl.float32)
        m0_0 = tl.maximum(s0, s1)
        m0_1 = tl.maximum(s2, s3)
        m0_2 = tl.maximum(s4, s5)
        m0_3 = tl.maximum(s6, s7)
        m0_4 = tl.maximum(s8, s9)
        m0_5 = tl.maximum(s10, s11)
        m0_6 = tl.maximum(s12, s13)
        m0_7 = tl.maximum(s14, s15)
        m1_0 = tl.maximum(m0_0, m0_1)
        m1_1 = tl.maximum(m0_2, m0_3)
        m1_2 = tl.maximum(m0_4, m0_5)
        m1_3 = tl.maximum(m0_6, m0_7)
        m2_0 = tl.maximum(m1_0, m1_1)
        m2_1 = tl.maximum(m1_2, m1_3)
        m3_0 = tl.maximum(m2_0, m2_1)
        m = tl.where(m3_0 == float("-inf"), 0.0, m3_0)
        w0 = tl.exp(s0 - m)
        a0 = tl.load(base + 0 * D).to(tl.float32) * w0
        w1 = tl.exp(s1 - m)
        a1 = tl.load(base + 2 * D).to(tl.float32) * w1
        w2 = tl.exp(s2 - m)
        a2 = tl.load(base + 4 * D).to(tl.float32) * w2
        w3 = tl.exp(s3 - m)
        a3 = tl.load(base + 6 * D).to(tl.float32) * w3
        w4 = tl.exp(s4 - m)
        a4 = tl.load(base + 8 * D).to(tl.float32) * w4
        w5 = tl.exp(s5 - m)
        a5 = tl.load(base + 10 * D).to(tl.float32) * w5
        w6 = tl.exp(s6 - m)
        a6 = tl.load(base + 12 * D).to(tl.float32) * w6
        w7 = tl.exp(s7 - m)
        a7 = tl.load(base + 14 * D).to(tl.float32) * w7
        w8 = tl.exp(s8 - m)
        a8 = tl.load(base + 16 * D).to(tl.float32) * w8
        w9 = tl.exp(s9 - m)
        a9 = tl.load(base + 18 * D).to(tl.float32) * w9
        w10 = tl.exp(s10 - m)
        a10 = tl.load(base + 20 * D).to(tl.float32) * w10
        w11 = tl.exp(s11 - m)
        a11 = tl.load(base + 22 * D).to(tl.float32) * w11
        w12 = tl.exp(s12 - m)
        a12 = tl.load(base + 24 * D).to(tl.float32) * w12
        w13 = tl.exp(s13 - m)
        a13 = tl.load(base + 26 * D).to(tl.float32) * w13
        w14 = tl.exp(s14 - m)
        a14 = tl.load(base + 28 * D).to(tl.float32) * w14
        w15 = tl.exp(s15 - m)
        a15 = tl.load(base + 30 * D).to(tl.float32) * w15
        w0_0 = w0 + w1
        w0_1 = w2 + w3
        w0_2 = w4 + w5
        w0_3 = w6 + w7
        w0_4 = w8 + w9
        w0_5 = w10 + w11
        w0_6 = w12 + w13
        w0_7 = w14 + w15
        w1_0 = w0_0 + w0_1
        w1_1 = w0_2 + w0_3
        w1_2 = w0_4 + w0_5
        w1_3 = w0_6 + w0_7
        w2_0 = w1_0 + w1_1
        w2_1 = w1_2 + w1_3
        w3_0 = w2_0 + w2_1
        wsum = w3_0
        a0_0 = a0 + a1
        a0_1 = a2 + a3
        a0_2 = a4 + a5
        a0_3 = a6 + a7
        a0_4 = a8 + a9
        a0_5 = a10 + a11
        a0_6 = a12 + a13
        a0_7 = a14 + a15
        a1_0 = a0_0 + a0_1
        a1_1 = a0_2 + a0_3
        a1_2 = a0_4 + a0_5
        a1_3 = a0_6 + a0_7
        a2_0 = a1_0 + a1_1
        a2_1 = a1_2 + a1_3
        a3_0 = a2_0 + a2_1
        asum = a3_0
        tl.store(Y + b * D + c, asum / wsum)


@triton.jit(do_not_specialize=["N"])
def _tree64(X, Y, N, D: tl.constexpr):
    c = tl.arange(0, D)
    for b in tl.range(tl.program_id(0), N, tl.num_programs(0)):
        base = X + b * 64 * 2 * D + c
        s0 = tl.load(base + 1 * D).to(tl.float32)
        s1 = tl.load(base + 3 * D).to(tl.float32)
        s2 = tl.load(base + 5 * D).to(tl.float32)
        s3 = tl.load(base + 7 * D).to(tl.float32)
        s4 = tl.load(base + 9 * D).to(tl.float32)
        s5 = tl.load(base + 11 * D).to(tl.float32)
        s6 = tl.load(base + 13 * D).to(tl.float32)
        s7 = tl.load(base + 15 * D).to(tl.float32)
        s8 = tl.load(base + 17 * D).to(tl.float32)
        s9 = tl.load(base + 19 * D).to(tl.float32)
        s10 = tl.load(base + 21 * D).to(tl.float32)
        s11 = tl.load(base + 23 * D).to(tl.float32)
        s12 = tl.load(base + 25 * D).to(tl.float32)
        s13 = tl.load(base + 27 * D).to(tl.float32)
        s14 = tl.load(base + 29 * D).to(tl.float32)
        s15 = tl.load(base + 31 * D).to(tl.float32)
        s16 = tl.load(base + 33 * D).to(tl.float32)
        s17 = tl.load(base + 35 * D).to(tl.float32)
        s18 = tl.load(base + 37 * D).to(tl.float32)
        s19 = tl.load(base + 39 * D).to(tl.float32)
        s20 = tl.load(base + 41 * D).to(tl.float32)
        s21 = tl.load(base + 43 * D).to(tl.float32)
        s22 = tl.load(base + 45 * D).to(tl.float32)
        s23 = tl.load(base + 47 * D).to(tl.float32)
        s24 = tl.load(base + 49 * D).to(tl.float32)
        s25 = tl.load(base + 51 * D).to(tl.float32)
        s26 = tl.load(base + 53 * D).to(tl.float32)
        s27 = tl.load(base + 55 * D).to(tl.float32)
        s28 = tl.load(base + 57 * D).to(tl.float32)
        s29 = tl.load(base + 59 * D).to(tl.float32)
        s30 = tl.load(base + 61 * D).to(tl.float32)
        s31 = tl.load(base + 63 * D).to(tl.float32)
        s32 = tl.load(base + 65 * D).to(tl.float32)
        s33 = tl.load(base + 67 * D).to(tl.float32)
        s34 = tl.load(base + 69 * D).to(tl.float32)
        s35 = tl.load(base + 71 * D).to(tl.float32)
        s36 = tl.load(base + 73 * D).to(tl.float32)
        s37 = tl.load(base + 75 * D).to(tl.float32)
        s38 = tl.load(base + 77 * D).to(tl.float32)
        s39 = tl.load(base + 79 * D).to(tl.float32)
        s40 = tl.load(base + 81 * D).to(tl.float32)
        s41 = tl.load(base + 83 * D).to(tl.float32)
        s42 = tl.load(base + 85 * D).to(tl.float32)
        s43 = tl.load(base + 87 * D).to(tl.float32)
        s44 = tl.load(base + 89 * D).to(tl.float32)
        s45 = tl.load(base + 91 * D).to(tl.float32)
        s46 = tl.load(base + 93 * D).to(tl.float32)
        s47 = tl.load(base + 95 * D).to(tl.float32)
        s48 = tl.load(base + 97 * D).to(tl.float32)
        s49 = tl.load(base + 99 * D).to(tl.float32)
        s50 = tl.load(base + 101 * D).to(tl.float32)
        s51 = tl.load(base + 103 * D).to(tl.float32)
        s52 = tl.load(base + 105 * D).to(tl.float32)
        s53 = tl.load(base + 107 * D).to(tl.float32)
        s54 = tl.load(base + 109 * D).to(tl.float32)
        s55 = tl.load(base + 111 * D).to(tl.float32)
        s56 = tl.load(base + 113 * D).to(tl.float32)
        s57 = tl.load(base + 115 * D).to(tl.float32)
        s58 = tl.load(base + 117 * D).to(tl.float32)
        s59 = tl.load(base + 119 * D).to(tl.float32)
        s60 = tl.load(base + 121 * D).to(tl.float32)
        s61 = tl.load(base + 123 * D).to(tl.float32)
        s62 = tl.load(base + 125 * D).to(tl.float32)
        s63 = tl.load(base + 127 * D).to(tl.float32)
        m0_0 = tl.maximum(s0, s1)
        m0_1 = tl.maximum(s2, s3)
        m0_2 = tl.maximum(s4, s5)
        m0_3 = tl.maximum(s6, s7)
        m0_4 = tl.maximum(s8, s9)
        m0_5 = tl.maximum(s10, s11)
        m0_6 = tl.maximum(s12, s13)
        m0_7 = tl.maximum(s14, s15)
        m0_8 = tl.maximum(s16, s17)
        m0_9 = tl.maximum(s18, s19)
        m0_10 = tl.maximum(s20, s21)
        m0_11 = tl.maximum(s22, s23)
        m0_12 = tl.maximum(s24, s25)
        m0_13 = tl.maximum(s26, s27)
        m0_14 = tl.maximum(s28, s29)
        m0_15 = tl.maximum(s30, s31)
        m0_16 = tl.maximum(s32, s33)
        m0_17 = tl.maximum(s34, s35)
        m0_18 = tl.maximum(s36, s37)
        m0_19 = tl.maximum(s38, s39)
        m0_20 = tl.maximum(s40, s41)
        m0_21 = tl.maximum(s42, s43)
        m0_22 = tl.maximum(s44, s45)
        m0_23 = tl.maximum(s46, s47)
        m0_24 = tl.maximum(s48, s49)
        m0_25 = tl.maximum(s50, s51)
        m0_26 = tl.maximum(s52, s53)
        m0_27 = tl.maximum(s54, s55)
        m0_28 = tl.maximum(s56, s57)
        m0_29 = tl.maximum(s58, s59)
        m0_30 = tl.maximum(s60, s61)
        m0_31 = tl.maximum(s62, s63)
        m1_0 = tl.maximum(m0_0, m0_1)
        m1_1 = tl.maximum(m0_2, m0_3)
        m1_2 = tl.maximum(m0_4, m0_5)
        m1_3 = tl.maximum(m0_6, m0_7)
        m1_4 = tl.maximum(m0_8, m0_9)
        m1_5 = tl.maximum(m0_10, m0_11)
        m1_6 = tl.maximum(m0_12, m0_13)
        m1_7 = tl.maximum(m0_14, m0_15)
        m1_8 = tl.maximum(m0_16, m0_17)
        m1_9 = tl.maximum(m0_18, m0_19)
        m1_10 = tl.maximum(m0_20, m0_21)
        m1_11 = tl.maximum(m0_22, m0_23)
        m1_12 = tl.maximum(m0_24, m0_25)
        m1_13 = tl.maximum(m0_26, m0_27)
        m1_14 = tl.maximum(m0_28, m0_29)
        m1_15 = tl.maximum(m0_30, m0_31)
        m2_0 = tl.maximum(m1_0, m1_1)
        m2_1 = tl.maximum(m1_2, m1_3)
        m2_2 = tl.maximum(m1_4, m1_5)
        m2_3 = tl.maximum(m1_6, m1_7)
        m2_4 = tl.maximum(m1_8, m1_9)
        m2_5 = tl.maximum(m1_10, m1_11)
        m2_6 = tl.maximum(m1_12, m1_13)
        m2_7 = tl.maximum(m1_14, m1_15)
        m3_0 = tl.maximum(m2_0, m2_1)
        m3_1 = tl.maximum(m2_2, m2_3)
        m3_2 = tl.maximum(m2_4, m2_5)
        m3_3 = tl.maximum(m2_6, m2_7)
        m4_0 = tl.maximum(m3_0, m3_1)
        m4_1 = tl.maximum(m3_2, m3_3)
        m5_0 = tl.maximum(m4_0, m4_1)
        m = tl.where(m5_0 == float("-inf"), 0.0, m5_0)
        w0 = tl.exp(s0 - m)
        a0 = tl.load(base + 0 * D).to(tl.float32) * w0
        w1 = tl.exp(s1 - m)
        a1 = tl.load(base + 2 * D).to(tl.float32) * w1
        w2 = tl.exp(s2 - m)
        a2 = tl.load(base + 4 * D).to(tl.float32) * w2
        w3 = tl.exp(s3 - m)
        a3 = tl.load(base + 6 * D).to(tl.float32) * w3
        w4 = tl.exp(s4 - m)
        a4 = tl.load(base + 8 * D).to(tl.float32) * w4
        w5 = tl.exp(s5 - m)
        a5 = tl.load(base + 10 * D).to(tl.float32) * w5
        w6 = tl.exp(s6 - m)
        a6 = tl.load(base + 12 * D).to(tl.float32) * w6
        w7 = tl.exp(s7 - m)
        a7 = tl.load(base + 14 * D).to(tl.float32) * w7
        w8 = tl.exp(s8 - m)
        a8 = tl.load(base + 16 * D).to(tl.float32) * w8
        w9 = tl.exp(s9 - m)
        a9 = tl.load(base + 18 * D).to(tl.float32) * w9
        w10 = tl.exp(s10 - m)
        a10 = tl.load(base + 20 * D).to(tl.float32) * w10
        w11 = tl.exp(s11 - m)
        a11 = tl.load(base + 22 * D).to(tl.float32) * w11
        w12 = tl.exp(s12 - m)
        a12 = tl.load(base + 24 * D).to(tl.float32) * w12
        w13 = tl.exp(s13 - m)
        a13 = tl.load(base + 26 * D).to(tl.float32) * w13
        w14 = tl.exp(s14 - m)
        a14 = tl.load(base + 28 * D).to(tl.float32) * w14
        w15 = tl.exp(s15 - m)
        a15 = tl.load(base + 30 * D).to(tl.float32) * w15
        w16 = tl.exp(s16 - m)
        a16 = tl.load(base + 32 * D).to(tl.float32) * w16
        w17 = tl.exp(s17 - m)
        a17 = tl.load(base + 34 * D).to(tl.float32) * w17
        w18 = tl.exp(s18 - m)
        a18 = tl.load(base + 36 * D).to(tl.float32) * w18
        w19 = tl.exp(s19 - m)
        a19 = tl.load(base + 38 * D).to(tl.float32) * w19
        w20 = tl.exp(s20 - m)
        a20 = tl.load(base + 40 * D).to(tl.float32) * w20
        w21 = tl.exp(s21 - m)
        a21 = tl.load(base + 42 * D).to(tl.float32) * w21
        w22 = tl.exp(s22 - m)
        a22 = tl.load(base + 44 * D).to(tl.float32) * w22
        w23 = tl.exp(s23 - m)
        a23 = tl.load(base + 46 * D).to(tl.float32) * w23
        w24 = tl.exp(s24 - m)
        a24 = tl.load(base + 48 * D).to(tl.float32) * w24
        w25 = tl.exp(s25 - m)
        a25 = tl.load(base + 50 * D).to(tl.float32) * w25
        w26 = tl.exp(s26 - m)
        a26 = tl.load(base + 52 * D).to(tl.float32) * w26
        w27 = tl.exp(s27 - m)
        a27 = tl.load(base + 54 * D).to(tl.float32) * w27
        w28 = tl.exp(s28 - m)
        a28 = tl.load(base + 56 * D).to(tl.float32) * w28
        w29 = tl.exp(s29 - m)
        a29 = tl.load(base + 58 * D).to(tl.float32) * w29
        w30 = tl.exp(s30 - m)
        a30 = tl.load(base + 60 * D).to(tl.float32) * w30
        w31 = tl.exp(s31 - m)
        a31 = tl.load(base + 62 * D).to(tl.float32) * w31
        w32 = tl.exp(s32 - m)
        a32 = tl.load(base + 64 * D).to(tl.float32) * w32
        w33 = tl.exp(s33 - m)
        a33 = tl.load(base + 66 * D).to(tl.float32) * w33
        w34 = tl.exp(s34 - m)
        a34 = tl.load(base + 68 * D).to(tl.float32) * w34
        w35 = tl.exp(s35 - m)
        a35 = tl.load(base + 70 * D).to(tl.float32) * w35
        w36 = tl.exp(s36 - m)
        a36 = tl.load(base + 72 * D).to(tl.float32) * w36
        w37 = tl.exp(s37 - m)
        a37 = tl.load(base + 74 * D).to(tl.float32) * w37
        w38 = tl.exp(s38 - m)
        a38 = tl.load(base + 76 * D).to(tl.float32) * w38
        w39 = tl.exp(s39 - m)
        a39 = tl.load(base + 78 * D).to(tl.float32) * w39
        w40 = tl.exp(s40 - m)
        a40 = tl.load(base + 80 * D).to(tl.float32) * w40
        w41 = tl.exp(s41 - m)
        a41 = tl.load(base + 82 * D).to(tl.float32) * w41
        w42 = tl.exp(s42 - m)
        a42 = tl.load(base + 84 * D).to(tl.float32) * w42
        w43 = tl.exp(s43 - m)
        a43 = tl.load(base + 86 * D).to(tl.float32) * w43
        w44 = tl.exp(s44 - m)
        a44 = tl.load(base + 88 * D).to(tl.float32) * w44
        w45 = tl.exp(s45 - m)
        a45 = tl.load(base + 90 * D).to(tl.float32) * w45
        w46 = tl.exp(s46 - m)
        a46 = tl.load(base + 92 * D).to(tl.float32) * w46
        w47 = tl.exp(s47 - m)
        a47 = tl.load(base + 94 * D).to(tl.float32) * w47
        w48 = tl.exp(s48 - m)
        a48 = tl.load(base + 96 * D).to(tl.float32) * w48
        w49 = tl.exp(s49 - m)
        a49 = tl.load(base + 98 * D).to(tl.float32) * w49
        w50 = tl.exp(s50 - m)
        a50 = tl.load(base + 100 * D).to(tl.float32) * w50
        w51 = tl.exp(s51 - m)
        a51 = tl.load(base + 102 * D).to(tl.float32) * w51
        w52 = tl.exp(s52 - m)
        a52 = tl.load(base + 104 * D).to(tl.float32) * w52
        w53 = tl.exp(s53 - m)
        a53 = tl.load(base + 106 * D).to(tl.float32) * w53
        w54 = tl.exp(s54 - m)
        a54 = tl.load(base + 108 * D).to(tl.float32) * w54
        w55 = tl.exp(s55 - m)
        a55 = tl.load(base + 110 * D).to(tl.float32) * w55
        w56 = tl.exp(s56 - m)
        a56 = tl.load(base + 112 * D).to(tl.float32) * w56
        w57 = tl.exp(s57 - m)
        a57 = tl.load(base + 114 * D).to(tl.float32) * w57
        w58 = tl.exp(s58 - m)
        a58 = tl.load(base + 116 * D).to(tl.float32) * w58
        w59 = tl.exp(s59 - m)
        a59 = tl.load(base + 118 * D).to(tl.float32) * w59
        w60 = tl.exp(s60 - m)
        a60 = tl.load(base + 120 * D).to(tl.float32) * w60
        w61 = tl.exp(s61 - m)
        a61 = tl.load(base + 122 * D).to(tl.float32) * w61
        w62 = tl.exp(s62 - m)
        a62 = tl.load(base + 124 * D).to(tl.float32) * w62
        w63 = tl.exp(s63 - m)
        a63 = tl.load(base + 126 * D).to(tl.float32) * w63
        w0_0 = w0 + w1
        w0_1 = w2 + w3
        w0_2 = w4 + w5
        w0_3 = w6 + w7
        w0_4 = w8 + w9
        w0_5 = w10 + w11
        w0_6 = w12 + w13
        w0_7 = w14 + w15
        w0_8 = w16 + w17
        w0_9 = w18 + w19
        w0_10 = w20 + w21
        w0_11 = w22 + w23
        w0_12 = w24 + w25
        w0_13 = w26 + w27
        w0_14 = w28 + w29
        w0_15 = w30 + w31
        w0_16 = w32 + w33
        w0_17 = w34 + w35
        w0_18 = w36 + w37
        w0_19 = w38 + w39
        w0_20 = w40 + w41
        w0_21 = w42 + w43
        w0_22 = w44 + w45
        w0_23 = w46 + w47
        w0_24 = w48 + w49
        w0_25 = w50 + w51
        w0_26 = w52 + w53
        w0_27 = w54 + w55
        w0_28 = w56 + w57
        w0_29 = w58 + w59
        w0_30 = w60 + w61
        w0_31 = w62 + w63
        w1_0 = w0_0 + w0_1
        w1_1 = w0_2 + w0_3
        w1_2 = w0_4 + w0_5
        w1_3 = w0_6 + w0_7
        w1_4 = w0_8 + w0_9
        w1_5 = w0_10 + w0_11
        w1_6 = w0_12 + w0_13
        w1_7 = w0_14 + w0_15
        w1_8 = w0_16 + w0_17
        w1_9 = w0_18 + w0_19
        w1_10 = w0_20 + w0_21
        w1_11 = w0_22 + w0_23
        w1_12 = w0_24 + w0_25
        w1_13 = w0_26 + w0_27
        w1_14 = w0_28 + w0_29
        w1_15 = w0_30 + w0_31
        w2_0 = w1_0 + w1_1
        w2_1 = w1_2 + w1_3
        w2_2 = w1_4 + w1_5
        w2_3 = w1_6 + w1_7
        w2_4 = w1_8 + w1_9
        w2_5 = w1_10 + w1_11
        w2_6 = w1_12 + w1_13
        w2_7 = w1_14 + w1_15
        w3_0 = w2_0 + w2_1
        w3_1 = w2_2 + w2_3
        w3_2 = w2_4 + w2_5
        w3_3 = w2_6 + w2_7
        w4_0 = w3_0 + w3_1
        w4_1 = w3_2 + w3_3
        w5_0 = w4_0 + w4_1
        wsum = w5_0
        a0_0 = a0 + a1
        a0_1 = a2 + a3
        a0_2 = a4 + a5
        a0_3 = a6 + a7
        a0_4 = a8 + a9
        a0_5 = a10 + a11
        a0_6 = a12 + a13
        a0_7 = a14 + a15
        a0_8 = a16 + a17
        a0_9 = a18 + a19
        a0_10 = a20 + a21
        a0_11 = a22 + a23
        a0_12 = a24 + a25
        a0_13 = a26 + a27
        a0_14 = a28 + a29
        a0_15 = a30 + a31
        a0_16 = a32 + a33
        a0_17 = a34 + a35
        a0_18 = a36 + a37
        a0_19 = a38 + a39
        a0_20 = a40 + a41
        a0_21 = a42 + a43
        a0_22 = a44 + a45
        a0_23 = a46 + a47
        a0_24 = a48 + a49
        a0_25 = a50 + a51
        a0_26 = a52 + a53
        a0_27 = a54 + a55
        a0_28 = a56 + a57
        a0_29 = a58 + a59
        a0_30 = a60 + a61
        a0_31 = a62 + a63
        a1_0 = a0_0 + a0_1
        a1_1 = a0_2 + a0_3
        a1_2 = a0_4 + a0_5
        a1_3 = a0_6 + a0_7
        a1_4 = a0_8 + a0_9
        a1_5 = a0_10 + a0_11
        a1_6 = a0_12 + a0_13
        a1_7 = a0_14 + a0_15
        a1_8 = a0_16 + a0_17
        a1_9 = a0_18 + a0_19
        a1_10 = a0_20 + a0_21
        a1_11 = a0_22 + a0_23
        a1_12 = a0_24 + a0_25
        a1_13 = a0_26 + a0_27
        a1_14 = a0_28 + a0_29
        a1_15 = a0_30 + a0_31
        a2_0 = a1_0 + a1_1
        a2_1 = a1_2 + a1_3
        a2_2 = a1_4 + a1_5
        a2_3 = a1_6 + a1_7
        a2_4 = a1_8 + a1_9
        a2_5 = a1_10 + a1_11
        a2_6 = a1_12 + a1_13
        a2_7 = a1_14 + a1_15
        a3_0 = a2_0 + a2_1
        a3_1 = a2_2 + a2_3
        a3_2 = a2_4 + a2_5
        a3_3 = a2_6 + a2_7
        a4_0 = a3_0 + a3_1
        a4_1 = a3_2 + a3_3
        a5_0 = a4_0 + a4_1
        asum = a5_0
        tl.store(Y + b * D + c, asum / wsum)


@triton.jit
def _vector(
    X,
    Y,
    BS: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    TRANS: tl.constexpr,
    GROUP: tl.constexpr,
):
    b = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    k = tl.arange(0, BK)
    if TRANS:
        pos = (
            b[:, None, None] * K * 2 * D
            + c[None, :, None]
            + k[None, None, :] * 2 * D
        )
        mask = (
            (b[:, None, None] < BS)
            & (c[None, :, None] < D)
            & (k[None, None, :] < K)
        )
        s = tl.load(X + pos + D, mask, float("-inf")).to(tl.float32)
        v = tl.load(X + pos, mask, 0.0).to(tl.float32)
        m = tl.max(s, 2)
        m = tl.where(m == float("-inf"), 0.0, m)
        w = tl.exp(s - m[:, :, None])
        out = tl.sum(v * w, 2) / tl.sum(w, 2)
    else:
        pos = (
            b[:, None, None] * K * 2 * D
            + k[None, :, None] * 2 * D
            + c[None, None, :]
        )
        mask = (
            (b[:, None, None] < BS)
            & (k[None, :, None] < K)
            & (c[None, None, :] < D)
        )
        s = tl.load(X + pos + D, mask, float("-inf")).to(tl.float32)
        v = tl.load(X + pos, mask, 0.0).to(tl.float32)
        m = tl.max(s, 1)
        m = tl.where(m == float("-inf"), 0.0, m)
        w = tl.exp(s - m[:, None, :])
        out = tl.sum(v * w, 1) / tl.sum(w, 1)
    tl.store(
        Y + b[:, None] * D + c[None, :],
        out,
        (b[:, None] < BS) & (c[None, :] < D),
    )


@triton.jit
def _flat(
    X, Y, N, K: tl.constexpr, D: tl.constexpr, B: tl.constexpr, Q: tl.constexpr
):
    j = tl.program_id(0) * B + tl.arange(0, B)
    b = j // D
    c = j % D
    valid = j < N * D
    if Q == 1:
        base = X + b * K * 2 * D + c
        m = tl.full((B,), float("-inf"), tl.float32)
        for k in tl.static_range(K):
            s = tl.load(base + k * 2 * D + D, valid, float("-inf")).to(
                tl.float32
            )
            m = tl.maximum(m, s)
        m = tl.where(m == float("-inf"), 0.0, m)
        a = tl.full((B,), 0.0, tl.float32)
        z = tl.full((B,), 0.0, tl.float32)
        for k in tl.static_range(K):
            s = tl.load(base + k * 2 * D + D, valid, float("-inf")).to(
                tl.float32
            )
            v = tl.load(base + k * 2 * D, valid, 0.0).to(tl.float32)
            w = tl.exp(s - m)
            a += v * w
            z += w
        tl.store(Y + j, a / z, valid)
    else:
        q = tl.arange(0, Q)
        base = X + b[None, :] * K * 2 * D + c[None, :]
        m = tl.full((Q, B), float("-inf"), tl.float32)
        for i in tl.static_range((K + Q - 1) // Q):
            k = i * Q + q
            mask = (k[:, None] < K) & valid[None, :]
            s = tl.load(base + k[:, None] * 2 * D + D, mask, float("-inf")).to(
                tl.float32
            )
            m = tl.maximum(m, s)
        mx = tl.max(m, 0)
        mx = tl.where(mx == float("-inf"), 0.0, mx)
        a = tl.full((Q, B), 0.0, tl.float32)
        z = tl.full((Q, B), 0.0, tl.float32)
        for i in tl.static_range((K + Q - 1) // Q):
            k = i * Q + q
            mask = (k[:, None] < K) & valid[None, :]
            s = tl.load(base + k[:, None] * 2 * D + D, mask, float("-inf")).to(
                tl.float32
            )
            v = tl.load(base + k[:, None] * 2 * D, mask, 0.0).to(tl.float32)
            w = tl.exp(s - mx[None, :])
            a += v * w
            z += w
        tl.store(Y + j, tl.sum(a, 0) / tl.sum(z, 0), valid)


@triton.jit
def _zero(Y, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(Y + i, 0.0, i < N)


def fused_softmax_pool(kv_score, head_dim):
    bs, k, _ = kv_score.shape
    out = kv_score.new_empty((bs, head_dim))
    if k == 0 and bs > 0 and (head_dim > 0):
        _zero[triton.cdiv(bs * head_dim, 256),](out, bs * head_dim)
        return out
    if bs == 0 or head_dim == 0 or k == 0:
        return out
    if head_dim == 128 and k == 64 and (bs > 64):
        group = 8 if bs % 8 == 0 else 1
        _shared_grid[bs * 2,](
            kv_score,
            out,
            k,
            128,
            64,
            4,
            0,
            group,
            "",
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out
    if head_dim == 128 and k == 4 and (bs >= 1024):
        _flat[triton.cdiv(bs, 4),](
            kv_score,
            out,
            bs,
            k,
            head_dim,
            512,
            1,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out
    if head_dim == 128 and bs <= 4 and (k == 64):
        bd = 4 if k == 4 else 2
        _vector[bs, 128 // bd](
            kv_score,
            out,
            bs,
            k,
            128,
            k,
            bd,
            False,
            1,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return out
    if head_dim == 128 and bs <= 64 and (k in (16, 64)):
        kernel = _tree16 if k == 16 else _tree64
        kernel[bs,](
            kv_score, out, bs, 128, num_warps=2, enable_fp_fusion=False
        )
        return out
    if head_dim == 128:
        if k == 64:
            _entry64[bs, 1](kv_score, out, num_warps=2, enable_fp_fusion=False)
            return out
        if k == 4 or k == 16:
            entry = _entry4 if k == 4 else _entry16
            entry[bs,](kv_score, out, bs, num_warps=2, enable_fp_fusion=False)
            return out
    if 32 <= k <= 128:
        _ilp[bs, triton.cdiv(head_dim, 128)](
            kv_score,
            out,
            k,
            head_dim,
            128,
            num_warps=2,
            enable_fp_fusion=False,
        )
        return out
    if head_dim == 64 and k <= 64 or (head_dim == 128 and k <= 16):
        _plain[bs,](
            kv_score, out, k, head_dim, bs, num_warps=2, enable_fp_fusion=False
        )
        return out
    block = _block(k, head_dim)
    tiles = triton.cdiv(head_dim, block)
    jobs = bs * tiles
    warps = 4 if block >= 128 else 2 if block >= 64 else 1
    kernel = _small if k <= 64 else _large
    kernel[jobs,](
        kv_score,
        out,
        k,
        head_dim,
        block,
        tiles,
        jobs,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["fused_softmax_pool"]
