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
import triton.language as tl


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


@triton.jit
def _flat4(X, Y, N):
    _flat(X, Y, N, 4, 128, 256, 1)


@triton.jit
def _flat16(X, Y, N):
    _flat(X, Y, N, 16, 128, 256, 1)


@triton.jit
def _flat64(X, Y, N):
    _flat(X, Y, N, 64, 128, 256, 1)


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
    if head_dim == 128 and bs <= 64 and (k in (4, 16, 64)):
        if bs <= 4:
            bd = 1 if k == 64 else 4
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
        elif k == 64:
            _vector[bs, 4](
                kv_score,
                out,
                bs,
                k,
                128,
                k,
                32,
                False,
                1,
                num_warps=1,
                enable_fp_fusion=False,
            )
        else:
            _flat[triton.cdiv(bs, 2),](
                kv_score,
                out,
                bs,
                k,
                128,
                256,
                1,
                num_warps=2,
                enable_fp_fusion=False,
            )
        return out
    if head_dim == 128 and k in (4, 16, 64):
        if bs >= 1024:
            kernel = _flat4 if k == 4 else _flat16 if k == 16 else _flat64
            kernel[triton.cdiv(bs, 2),](
                kv_score, out, bs, num_warps=2, enable_fp_fusion=False
            )
            return out
        if k == 4:
            _tree4[bs,](
                kv_score, out, bs, 128, num_warps=2, enable_fp_fusion=False
            )
            return out
    if head_dim == 128 and bs >= 1024 and (k <= 64):
        _flat[triton.cdiv(bs * head_dim, 256),](
            kv_score,
            out,
            bs,
            k,
            head_dim,
            256,
            1,
            num_warps=2,
            enable_fp_fusion=False,
        )
        return out
    block = _block(k, head_dim)
    tiles = triton.cdiv(head_dim, block)
    jobs = bs * tiles
    warps = 2
    kernel = _small if k <= 64 else _large
    kernel[min(jobs, 4096),](
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
