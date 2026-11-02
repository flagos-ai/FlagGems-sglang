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
def _direct(X, Y, K: tl.constexpr, D: tl.constexpr, CP: tl.constexpr):
    offs = tl.arange(0, D)
    stride_k = D * 2
    b = tl.program_id(0)
    base = X + b * K * stride_k
    m = tl.full((D,), float("-inf"), tl.float32)
    for k in tl.static_range(K):
        s = tl.load(base + k * stride_k + D + offs, cache_modifier=CP).to(
            tl.float32
        )
        m = tl.maximum(m, s)
    acc = tl.zeros((D,), tl.float32)
    den = tl.zeros((D,), tl.float32)
    for k in tl.static_range(K):
        s = tl.load(base + k * stride_k + D + offs, cache_modifier=CP).to(
            tl.float32
        )
        v = tl.load(base + k * stride_k + offs, cache_modifier=CP).to(
            tl.float32
        )
        w = tl.exp(s - m)
        acc += v * w
        den += w
    tl.store(Y + b * D + offs, acc / den)


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
    lanes = tl.arange(0, 4)
    base = X + b * K * 2 * D + c[None, :]
    m = tl.full((4, BD), float("-inf"), tl.float32)
    for i in tl.static_range(triton.cdiv(K, 4)):
        k = i * 4 + lanes
        mask = (k[:, None] < K) & (c[None, :] < D)
        s = tl.load(base + k[:, None] * 2 * D + D, mask, float("-inf")).to(
            tl.float32
        )
        m = tl.maximum(m, s)
    mx = tl.max(m, axis=0)
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    den = tl.zeros((4, BD), tl.float32)
    acc = tl.zeros((4, BD), tl.float32)
    for i in tl.static_range(triton.cdiv(K, 4)):
        k = i * 4 + lanes
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
def _entry4(X, Y):
    _plain(X, Y, 4, 128, tl.num_programs(0))


@triton.jit
def _entry16(X, Y):
    _plain(X, Y, 16, 128, tl.num_programs(0))


@triton.jit
def _entry64(X, Y):
    _ilp(X, Y, 64, 128, 128)


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
    if (
        head_dim == 128
        and bs <= 64
        and (k == 16)
        or (head_dim == 128 and bs <= 4 and (k == 64))
    ):
        bd = 2 if k == 64 else 32 if bs <= 4 else 8
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
    if head_dim == 128 and k in (4, 16):
        _direct[bs,](
            kv_score, out, k, 128, ".cg", num_warps=1, enable_fp_fusion=False
        )
        return out
    if head_dim == 128 and k in (4, 16, 64):
        entry = _entry4 if k == 4 else _entry16 if k == 16 else _entry64
        entry[bs,](
            kv_score,
            out,
            num_warps=4 if k == 64 else 2,
            enable_fp_fusion=False,
        )
        return out
    if 32 <= k <= 128:
        _ilp[bs, triton.cdiv(head_dim, 128)](
            kv_score,
            out,
            k,
            head_dim,
            128,
            num_warps=4,
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
