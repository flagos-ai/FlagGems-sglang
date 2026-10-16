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
def _pool_stream(
    X,
    Y,
    T,
    K: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
):
    i = tl.program_id(0) // P * B + tl.arange(0, B)
    p = tl.program_id(0) % P
    base = i // D * S0 + i % D * S2
    m = tl.full((B,), -float("inf"), tl.float32)
    for step in tl.static_range(R):
        k = p * R + step
        s = tl.load(
            X + base + tl.minimum(k, K - 1) * S1 + D * S2,
            (i < N) & (k < K),
            other=-float("inf"),
        ).to(tl.float32)
        s = tl.where(k < K, s, -float("inf"))
        m = tl.maximum(m, s)
    m = tl.where(m == -float("inf"), 0.0, m)
    a = tl.full((B,), 0.0, tl.float32)
    z = tl.full((B,), 0.0, tl.float32)
    for step in tl.static_range(R):
        k = p * R + step
        s = tl.load(
            X + base + tl.minimum(k, K - 1) * S1 + D * S2,
            (i < N) & (k < K),
            other=-float("inf"),
        ).to(tl.float32)
        v = tl.load(
            X + base + tl.minimum(k, K - 1) * S1, (i < N) & (k < K), other=0
        ).to(tl.float32)
        s = tl.where(k < K, s, -float("inf"))
        v = tl.where(k < K, v, 0.0)
        w = tl.exp(s - m)
        a += v * w
        z += w
    if P == 1:
        tl.store(Y + i, a / tl.where(z > 0, z, 1.0), i < N)
    else:
        tl.store(T + p * 3 * N + i, m, i < N)
        tl.store(T + (p * 3 + 1) * N + i, z, i < N)
        tl.store(T + (p * 3 + 2) * N + i, a, i < N)


@triton.jit
def _merge_stream(T, Y, N: tl.constexpr, P: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = tl.full((B,), -float("inf"), tl.float32)
    for p in tl.static_range(P):
        local_m = tl.load(T + p * 3 * N + i, i < N, other=-float("inf"))
        m = tl.maximum(m, local_m)
    m = tl.where(m == -float("inf"), 0.0, m)
    a = tl.full((B,), 0.0, tl.float32)
    z = tl.full((B,), 0.0, tl.float32)
    for p in tl.static_range(P):
        local_m = tl.load(T + p * 3 * N + i, i < N, other=-float("inf"))
        local_z = tl.load(T + (p * 3 + 1) * N + i, i < N, other=0)
        local_a = tl.load(T + (p * 3 + 2) * N + i, i < N, other=0)
        w = tl.exp(local_m - m)
        a += local_a * w
        z += local_z * w
    tl.store(Y + i, a / tl.where(z > 0, z, 1.0), i < N)


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
    total = bs * head_dim
    if total:
        r = min(16, max(k, 1))
        parts = max(1, triton.cdiv(k, r))
        temp = (
            torch.empty(
                (parts, 3, total), dtype=torch.float32, device=kv_score.device
            )
            if parts > 1
            else out
        )
        _pool_stream[triton.cdiv(total, 512) * parts,](
            kv_score,
            out,
            temp,
            k,
            head_dim,
            total,
            *kv_score.stride(),
            512,
            parts,
            r,
            num_warps=1,
            enable_fp_fusion=False,
        )
        if parts > 1:
            _merge_stream[triton.cdiv(total, 512),](
                temp,
                out,
                total,
                parts,
                512,
                num_warps=1,
                enable_fp_fusion=False,
            )
    return out


__all__ = ["fused_softmax_pool"]
