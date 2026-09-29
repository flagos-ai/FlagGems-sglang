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
def _small(
    X,
    W,
    B,
    Y,
    C: tl.constexpr,
    S: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    WS: tl.constexpr,
    BS: tl.constexpr,
    E: tl.constexpr,
    T: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, T)
    x = tl.load(X + row * K + i, i < K, 0).to(tl.float32)
    mean = tl.sum(x, 0) / K
    d = tl.where(i < K, x - mean, 0.0)
    var = tl.sum(d * d, 0) / K
    r = tl.rsqrt(var + E)
    c = row % G * (C // G) + i // S
    w = tl.load(W + c * WS, i < K, 0).to(tl.float32)
    b = tl.load(B + c * BS, i < K, 0).to(tl.float32)
    y = (x - mean) * r * w + b
    tl.store(Y + row * K + i, y / (1.0 + tl.exp(-y)), i < K)


@triton.jit
def _stats(X, M, R, K: tl.constexpr, E: tl.constexpr, T: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, T)
    shift = tl.load(X + row * K).to(tl.float32)
    a = tl.full((T,), 0, tl.float32)
    b = tl.full((T,), 0, tl.float32)
    for block in range(triton.cdiv(K, T)):
        j = block * T + i
        x = tl.load(X + row * K + tl.minimum(j, K - 1)).to(tl.float32)
        x = tl.where(j < K, x - shift, 0.0)
        a += x
        b += x * x
    total = tl.sum(a, 0)
    sq = tl.sum(b, 0)
    mean = total / K
    var = tl.maximum(sq / K - mean * mean, 0.0)
    tl.store(M + row, mean + shift)
    tl.store(R + row, tl.rsqrt(var + E))


@triton.jit
def _apply(
    X,
    W,
    B,
    M,
    R,
    Y,
    C: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    Q: tl.constexpr,
    WS: tl.constexpr,
    BS: tl.constexpr,
    T: tl.constexpr,
):
    p = tl.program_id(0)
    channel = p // Q
    i = p % Q * T + tl.arange(0, T)
    mean = tl.load(M + channel // D)
    r = tl.load(R + channel // D)
    w = tl.load(W + channel % C * WS).to(tl.float32)
    b = tl.load(B + channel % C * BS).to(tl.float32)
    x = tl.load(X + channel * S + tl.minimum(i, S - 1)).to(tl.float32)
    y = (x - mean) * r * w + b
    tl.store(Y + channel * S + i, y / (1.0 + tl.exp(-y)), i < S)


def group_norm_silu(x, weight, bias, num_groups, eps):
    x = x.contiguous()
    n, c = x.shape[:2]
    s = x.numel() // (n * c)
    k = c // num_groups * s
    out = torch.empty_like(x)
    if k <= 16384 and s & (s - 1) == 0:
        _small[(n * num_groups,)](
            x,
            weight,
            bias,
            out,
            c,
            s,
            num_groups,
            k,
            weight.stride(0),
            bias.stride(0),
            eps,
            triton.next_power_of_2(k),
            num_warps=8,
            enable_fp_fusion=False,
        )
        return out
    m = torch.empty((n * num_groups,), dtype=torch.float32, device=x.device)
    r = torch.empty_like(m)
    _stats[(n * num_groups,)](
        x,
        m,
        r,
        k,
        eps,
        min(4096, triton.next_power_of_2(k)),
        num_warps=4,
        enable_fp_fusion=False,
    )
    t = min(16384, triton.next_power_of_2(s))
    q = triton.cdiv(s, t)
    _apply[(n * c * q,)](
        x,
        weight,
        bias,
        m,
        r,
        out,
        c,
        s,
        c // num_groups,
        q,
        weight.stride(0),
        bias.stride(0),
        t,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["group_norm_silu"]
