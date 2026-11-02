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
def _pool(
    X,
    Y,
    K: tl.constexpr,
    D: tl.constexpr,
    BK: tl.constexpr,
    BD: tl.constexpr,
    NT: tl.constexpr,
    JOBS: tl.constexpr,
):
    cs = tl.arange(0, BD)
    ks = tl.arange(0, BK)
    for job in range(tl.program_id(0), JOBS, tl.num_programs(0)):
        b = job // NT
        c = job % NT * BD + cs
        offsets = b * K * 2 * D + ks[:, None] * 2 * D + c[None, :]
        mask = (ks[:, None] < K) & (c[None, :] < D)
        s = tl.load(X + offsets + D, mask=mask, other=float("-inf")).to(
            tl.float32
        )
        v = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        m = tl.max(s, axis=0)
        m = tl.where(m == float("-inf"), 0.0, m)
        w = tl.exp(s - m[None, :])
        result = tl.sum(v * w, axis=0) / tl.sum(w, axis=0)
        tl.store(Y + b * D + c, result, mask=c < D)


@triton.jit
def _stream(
    X,
    Y,
    K: tl.constexpr,
    D: tl.constexpr,
    NT: tl.constexpr,
    JOBS: tl.constexpr,
):
    cs = tl.arange(0, 128)
    for job in range(tl.program_id(0), JOBS, tl.num_programs(0)):
        b = job // NT
        c = job % NT * 128 + cs
        base = b * K * 2 * D + c
        m = tl.full((128,), float("-inf"), tl.float32)
        for k in range(K):
            s = tl.load(X + base + k * 2 * D + D, c < D, float("-inf")).to(
                tl.float32
            )
            m = tl.maximum(m, s)
        m = tl.where(m == float("-inf"), 0.0, m)
        den = tl.zeros((128,), tl.float32)
        acc = tl.zeros((128,), tl.float32)
        for k in range(K):
            s = tl.load(X + base + k * 2 * D + D, c < D, float("-inf")).to(
                tl.float32
            )
            v = tl.load(X + base + k * 2 * D, c < D, 0.0).to(tl.float32)
            w = tl.exp(s - m)
            den += w
            acc += w * v
        tl.store(Y + b * D + c, acc / den, c < D)


@triton.jit
def _entry4(X, Y, N: tl.constexpr):
    _pool(X, Y, 4, 128, 4, 128, 1, N)


@triton.jit
def _entry16(X, Y, N: tl.constexpr):
    _pool(X, Y, 16, 128, 16, 128, 1, N)


@triton.jit
def _entry64(X, Y, N: tl.constexpr):
    _pool(X, Y, 64, 128, 64, 128, 1, N)


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
    if bs == 0 or head_dim == 0:
        return out
    if head_dim == 128 and k in (4, 16, 64):
        entry = _entry4 if k == 4 else _entry16 if k == 16 else _entry64
        entry[min(bs, 40),](kv_score, out, bs, enable_fp_fusion=False)
        return out
    nt = triton.cdiv(head_dim, 128)
    jobs = bs * nt
    if k <= 256:
        _pool[min(jobs, 40),](
            kv_score,
            out,
            k,
            head_dim,
            triton.next_power_of_2(max(k, 1)),
            128,
            nt,
            jobs,
            enable_fp_fusion=False,
        )
    else:
        _stream[min(jobs, 40),](
            kv_score, out, k, head_dim, nt, jobs, enable_fp_fusion=False
        )
    return out


__all__ = ["fused_softmax_pool"]
