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


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["META"],
)
@triton.jit
def _whole(X, W, B, Y, META: tl.constexpr):
    S: tl.constexpr = META[0]
    D: tl.constexpr = META[1]
    G: tl.constexpr = META[2]
    E: tl.constexpr = META[3]
    DP: tl.constexpr = META[4]
    SP: tl.constexpr = META[5]
    row = tl.program_id(0)
    K = D * S
    base = row * K
    d = tl.arange(0, DP)
    s = tl.arange(0, SP)
    inside = (d[:, None] < D) & (s[None, :] < S)
    offsets = base + d[:, None] * S + s[None, :]
    values = tl.load(X + offsets, inside, 0).to(tl.float32)
    mean = tl.sum(tl.sum(values, 1), 0) / K
    v = tl.where(inside, values - mean, 0.0)
    variance = tl.sum(tl.sum(v * v, 1), 0) / K
    r = tl.rsqrt(variance + E)
    ch = row % G * D + d
    w = tl.load(W + ch, d < D, 0).to(tl.float32)
    b = tl.load(B + ch, d < D, 0).to(tl.float32)
    y = v * (r * w)[:, None] + b[:, None]
    tl.store(Y + offsets, y / (1.0 + tl.exp(-y)), inside)


@triton.jit
def _partial(X, P, META: tl.constexpr):
    K: tl.constexpr = META[0]
    Q: tl.constexpr = META[1]
    T: tl.constexpr = META[2]
    pid = tl.program_id(0)
    base = (pid // Q) * K
    i = (pid % Q) * T + tl.arange(0, T)
    inside = i < K
    shift = tl.load(X + base).to(tl.float32)
    v = tl.where(
        inside, tl.load(X + base + i, inside, 0.0).to(tl.float32) - shift, 0.0
    )
    tl.store(P + pid * 2, tl.sum(v, 0))
    tl.store(P + pid * 2 + 1, tl.sum(v * v, 0))


@triton.jit
def _apply(X, W, B, P, Y, META: tl.constexpr):
    S: tl.constexpr = META[0]
    D: tl.constexpr = META[1]
    G: tl.constexpr = META[2]
    Q: tl.constexpr = META[3]
    QP: tl.constexpr = META[4]
    E: tl.constexpr = META[5]
    T: tl.constexpr = META[6]
    pid = tl.program_id(0)
    row = pid // Q
    K = D * S
    base = row * K
    q = tl.arange(0, QP)
    ok = q < Q
    mu = tl.sum(tl.load(P + (row * Q + q) * 2, ok, 0.0), 0) / K
    sq = tl.sum(tl.load(P + (row * Q + q) * 2 + 1, ok, 0.0), 0) / K
    rstd = tl.rsqrt(tl.maximum(sq - mu * mu, 0.0) + E)
    i = (pid % Q) * T + tl.arange(0, T)
    inside = i < K
    shift = tl.load(X + base).to(tl.float32) + mu
    c = (row % G) * D + i // S
    w = tl.load(W + c, inside, 0.0).to(tl.float32)
    b = tl.load(B + c, inside, 0.0).to(tl.float32)
    y = (
        tl.load(X + base + i, inside, 0.0).to(tl.float32) - shift
    ) * rstd * w + b
    tl.store(Y + base + i, y / (1.0 + tl.exp(-y)), inside)


@triton.jit
def _copy_strided(
    X,
    Y,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remainder = off
    source = tl.full((BLOCK,), 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        source += (remainder % SHAPE[axis]) * STRIDES[axis]
        remainder = remainder // SHAPE[axis]
    value = tl.load(X + source, off < N, 0)
    tl.store(Y + off, value, off < N)


def _contiguous(x):
    if x.is_contiguous():
        return x
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n = x.numel()
    if n:
        _copy_strided[(triton.cdiv(n, 1024),)](
            x, out, tuple(x.shape), x.stride(), n, 1024
        )
    return out


def _pow2(value):
    return 1 << (value - 1).bit_length()


def group_norm_silu(x, weight, bias, num_groups, eps):
    x = _contiguous(x)
    weight = _contiguous(weight)
    bias = _contiguous(bias)
    shape = x.shape
    channels = shape[1]
    spatial = x.numel() // (shape[0] * channels)
    depth = channels // num_groups
    rows = shape[0] * num_groups
    span = depth * spatial
    out = torch.empty_like(x)
    if _pow2(depth) * _pow2(spatial) <= 16384:
        _whole[(rows,)](
            x,
            weight,
            bias,
            out,
            (
                spatial,
                depth,
                num_groups,
                eps,
                _pow2(depth),
                _pow2(spatial),
                rows,
            ),
        )
    else:
        chunks = -(-span // 16384)
        partial = torch.empty(
            rows * chunks * 2, dtype=torch.float32, device=out.device
        )
        _partial[(rows * chunks,)](
            x, partial, (span, chunks, 16384), num_warps=8
        )
        _apply[(rows * chunks,)](
            x,
            weight,
            bias,
            partial,
            out,
            (spatial, depth, num_groups, chunks, _pow2(chunks), eps, 16384),
            num_warps=8,
        )
    return out


__all__ = ["group_norm_silu"]
