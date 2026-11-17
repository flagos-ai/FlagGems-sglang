# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl


@triton.jit
def _direct_cg(X, Y, K: tl.constexpr, D: tl.constexpr):
    offs = tl.arange(0, D)
    stride_k = D * 2
    b = tl.program_id(0)
    base = X + b * K * stride_k
    m = tl.full((D,), float("-inf"), tl.float32)
    for k in tl.static_range(K):
        s = tl.load(base + k * stride_k + D + offs, cache_modifier=".cg").to(
            tl.float32
        )
        m = tl.maximum(m, s)
    acc = tl.zeros((D,), tl.float32)
    den = tl.zeros((D,), tl.float32)
    for k in tl.static_range(K):
        s = tl.load(base + k * stride_k + D + offs, cache_modifier=".cg").to(
            tl.float32
        )
        v = tl.load(base + k * stride_k + offs, cache_modifier=".cg").to(
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
def _vector(
    X,
    Y,
    BS: tl.constexpr,
    K: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    GROUP: tl.constexpr,
):
    b = tl.program_id(0) * GROUP + tl.arange(0, GROUP)
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    k = tl.arange(0, BK)
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
def _merge(m1, a1, z1, m2, a2, z2):
    left = m1 >= m2
    m = tl.maximum(m1, m2)
    delta = tl.where(m == float("-inf"), 0.0, -tl.abs(m1 - m2))
    w = tl.exp(delta)
    a = tl.where(left, a1 + a2 * w, a2 + a1 * w)
    z = tl.where(left, z1 + z2 * w, z2 + z1 * w)
    return (m, a, z)


@triton.jit
def _online(
    X,
    Y,
    N,
    K: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    Q: tl.constexpr,
):
    c = tl.program_id(1) * BD + tl.arange(0, BD)
    r = tl.arange(0, Q)
    for b in tl.range(tl.program_id(0), N, tl.num_programs(0)):
        pos = b * K * 2 * D + r[:, None] * 2 * D + c[None, :]
        valid = (r[:, None] < K) & (c[None, :] < D)
        m = tl.load(
            X + pos + D, valid, float("-inf"), cache_modifier=".cg"
        ).to(tl.float32)
        a = tl.load(X + pos, valid, 0.0, cache_modifier=".cg").to(tl.float32)
        z = tl.where(r[:, None] < K, 1.0, 0.0) + tl.zeros((Q, BD), tl.float32)
        for i in tl.range(1, (K + Q - 1) // Q, loop_unroll_factor=1):
            k = i * Q + r
            pos = b * K * 2 * D + k[:, None] * 2 * D + c[None, :]
            valid = (k[:, None] < K) & (c[None, :] < D)
            s = tl.load(
                X + pos + D, valid, float("-inf"), cache_modifier=".cg"
            ).to(tl.float32)
            v = tl.load(X + pos, valid, 0.0, cache_modifier=".cg").to(
                tl.float32
            )
            z2 = tl.where(k[:, None] < K, 1.0, 0.0) + tl.zeros(
                (Q, BD), tl.float32
            )
            m, a, z = _merge(m, a, z, s, v, z2)
        m, a, z = tl.reduce((m, a, z), 0, _merge)
        tl.store(Y + b * D + c, a / z, c < D)


@triton.jit
def _v16(X, Y, N: tl.constexpr):
    _vector(X, Y, N, 16, 128, 16, 4, 1)


@triton.jit
def _v64(X, Y, N: tl.constexpr):
    _vector(X, Y, N, 64, 128, 64, 1, 1)


@triton.jit
def _o16(X, Y, N):
    _online(X, Y, N, 16, 128, 128, 2)


@triton.jit
def _o64(X, Y, N):
    _online(X, Y, N, 64, 128, 128, 2)


@triton.jit
def _o64_narrow(X, Y, N):
    _online(X, Y, N, 64, 128, 32, 4)


@triton.jit
def _cg4(X, Y):
    _direct_cg(X, Y, 4, 128)


@triton.jit
def _cg16(X, Y):
    _direct_cg(X, Y, 16, 128)


@triton.jit
def _zero(Y, N: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(Y + i, 0.0, i < N)


def fused_softmax_pool(kv_score, head_dim):
    bs, k, _ = kv_score.shape
    out = kv_score.new_empty((bs, head_dim))
    if k == 0 and bs > 0 and head_dim > 0:
        _zero[(triton.cdiv(bs * head_dim, 256),)](out, bs * head_dim)
        return out
    if bs == 0 or head_dim == 0 or k == 0:
        return out
    if head_dim == 128 and bs <= 4 and k in (16, 64):
        bd = 4 if k == 16 else 1
        kernel = _v16 if k == 16 else _v64
        kernel[bs, 128 // bd](
            kv_score, out, bs, num_warps=1, enable_fp_fusion=False
        )
        return out
    # gfx1100: one program per row in the two-pass kernel is latency bound
    # for k=64 (about 42 us at any row count up to 64); the online merge
    # over 32-column slices takes 11-15 us there, and its 128-column form
    # is about 2x faster from 128 rows on. It also wins for k=16 from 2048
    # rows on (1.1-1.3x).
    if head_dim == 128 and k == 64:
        if bs < 128:
            _o64_narrow[bs, 4](
                kv_score, out, bs, num_warps=4, enable_fp_fusion=False
            )
        else:
            _o64[bs, 1](kv_score, out, bs, num_warps=2, enable_fp_fusion=False)
        return out
    if head_dim == 128 and k == 16 and bs >= 2048:
        _o16[min(bs, 4096), 1](
            kv_score, out, bs, num_warps=4, enable_fp_fusion=False
        )
        return out
    if head_dim == 128 and k in (4, 16):
        kernel = _cg4 if k == 4 else _cg16
        nw = 2 if bs >= 4096 else 4
        kernel[bs,](kv_score, out, num_warps=nw, enable_fp_fusion=True)
        return out
    block = _block(k, head_dim)
    tiles = triton.cdiv(head_dim, block)
    jobs = bs * tiles
    warps = 4 if block >= 128 else 2 if block >= 64 else 1
    kernel = _small if k <= 64 else _large
    kernel[(min(jobs, 4096),)](
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
