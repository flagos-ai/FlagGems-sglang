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
from triton.language.extra import libdevice


@triton.jit
def _pair_add(a, b, c, d):
    return a + c, b + d


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["META"],
)
@triton.jit
def _whole(X, W, B, Y, META: tl.constexpr):
    S: tl.constexpr = META[0]
    D: tl.constexpr = META[1]
    G: tl.constexpr = META[2]
    E: tl.constexpr = META[3]
    T: tl.constexpr = META[4]
    row = tl.program_id(0)
    K = D * S
    base = row * K
    i = tl.arange(0, T)
    inside = i < K
    shift = tl.load(X + base).to(tl.float32)
    v = tl.where(
        inside, tl.load(X + base + i, inside, 0.0).to(tl.float32) - shift, 0.0
    )
    total, totalq = tl.reduce((v, v * v), 0, _pair_add)
    mu = total / K
    rstd = tl.rsqrt(tl.maximum(totalq / K - mu * mu, 0.0) + E)
    c = (row % G) * D + i // S
    w = tl.load(W + c, inside, 0.0).to(tl.float32)
    b = tl.load(B + c, inside, 0.0).to(tl.float32)
    scale = rstd * w
    offset = b - mu * scale
    y = v * scale + offset
    tl.store(Y + base + i, libdevice.fast_dividef(y, 1.0 + tl.exp(-y)), inside)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["META"],
)
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
    total, totalq = tl.reduce((v, v * v), 0, _pair_add)
    tl.store(P + pid * 2, total)
    tl.store(P + pid * 2 + 1, totalq)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["META"],
)
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
    p = tl.load(P + (row * Q + q) * 2, ok, 0.0)
    pq = tl.load(P + (row * Q + q) * 2 + 1, ok, 0.0)
    total, totalq = tl.reduce((p, pq), 0, _pair_add)
    mu = total / K
    sq = totalq / K
    rstd = tl.rsqrt(tl.maximum(sq - mu * mu, 0.0) + E)
    i = (pid % Q) * T + tl.arange(0, T)
    inside = i < K
    shift = tl.load(X + base).to(tl.float32) + mu
    c = (row % G) * D + i // S
    w = tl.load(W + c, inside, 0.0).to(tl.float32)
    b = tl.load(B + c, inside, 0.0).to(tl.float32)
    scale = rstd * w
    offset = b - shift * scale
    y = tl.load(X + base + i, inside, 0.0).to(tl.float32) * scale + offset
    tl.store(Y + base + i, libdevice.fast_dividef(y, 1.0 + tl.exp(-y)), inside)


def _prune_replicate(configs, named_args, **kwargs):
    tile = named_args["META"][4]
    return [
        config
        for config in configs
        if config.num_warps >= 4 or tile <= config.num_warps * 8192
    ]


@triton.autotune(
    configs=[
        triton.Config({"SPLIT": 1}, num_warps=1),
        triton.Config({"SPLIT": 2}, num_warps=1),
        triton.Config({"SPLIT": 4}, num_warps=1),
        triton.Config({"SPLIT": 1}, num_warps=2),
        triton.Config({"SPLIT": 2}, num_warps=2),
        triton.Config({"SPLIT": 4}, num_warps=2),
        triton.Config({"SPLIT": 8}, num_warps=2),
        triton.Config({"SPLIT": 1}, num_warps=4),
        triton.Config({"SPLIT": 1}, num_warps=8),
        triton.Config({"SPLIT": 1}, num_warps=16),
        triton.Config({"SPLIT": 2}, num_warps=4),
        triton.Config({"SPLIT": 2}, num_warps=8),
        triton.Config({"SPLIT": 4}, num_warps=4),
        triton.Config({"SPLIT": 4}, num_warps=8),
        triton.Config({"SPLIT": 8}, num_warps=4),
        triton.Config({"SPLIT": 8}, num_warps=8),
        triton.Config({"SPLIT": 16}, num_warps=4),
    ],
    key=["META"],
    prune_configs_by={"early_config_prune": _prune_replicate},
)
@triton.jit
def _replicate(X, W, B, Y, META: tl.constexpr, SPLIT: tl.constexpr):
    S: tl.constexpr = META[0]
    D: tl.constexpr = META[1]
    G: tl.constexpr = META[2]
    E: tl.constexpr = META[3]
    T: tl.constexpr = META[4]
    row = tl.program_id(0) // SPLIT
    part = tl.program_id(0) % SPLIT
    K = D * S
    i = tl.arange(0, T)
    shift = tl.load(X + row * K).to(tl.float32)
    x = tl.load(X + row * K + i, i < K, 0).to(tl.float32)
    v = tl.where(i < K, x - shift, 0.0)
    total, totalq = tl.reduce((v, v * v), 0, _pair_add)
    mean = total / K
    var = tl.maximum(totalq / K - mean * mean, 0.0)
    r = tl.rsqrt(var + E)
    if S & (S - 1) == 0:
        BS: tl.constexpr = min(S, T // SPLIT)
        BD: tl.constexpr = T // SPLIT // BS
        first = part * (T // SPLIT)
        d = first // S + tl.arange(0, BD)
        spatial = first % S + tl.arange(0, BS)
        j = d[:, None] * S + spatial[None, :]
        valid = (d[:, None] < D) & (spatial[None, :] < S)
        c = row % G * D + d
        w = tl.load(W + c, d < D, 0).to(tl.float32)
        b = tl.load(B + c, d < D, 0).to(tl.float32)
        z = tl.load(X + row * K + j, valid, 0).to(tl.float32) - shift
        scale = r * w
        y = z * scale[:, None] + (b - mean * scale)[:, None]
        tl.store(
            Y + row * K + j, libdevice.fast_dividef(y, 1.0 + tl.exp(-y)), valid
        )
    else:
        j = part * (T // SPLIT) + tl.arange(0, T // SPLIT)
        valid = j < K
        c = row % G * D + j // S
        w = tl.load(W + c, valid, 0).to(tl.float32)
        b = tl.load(B + c, valid, 0).to(tl.float32)
        z = tl.load(X + row * K + j, valid, 0).to(tl.float32) - shift
        scale = r * w
        y = z * scale + (b - mean * scale)
        tl.store(
            Y + row * K + j, libdevice.fast_dividef(y, 1.0 + tl.exp(-y)), valid
        )


def _pow2(value):
    return 1 << (value - 1).bit_length()


def group_norm_silu(x, weight, bias, num_groups, eps):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    shape = x.shape
    channels = shape[1]
    spatial = x.numel() // (shape[0] * channels)
    depth = channels // num_groups
    rows = shape[0] * num_groups
    span = depth * spatial
    out = torch.empty_like(x)
    if 512 <= span <= 65536:
        _replicate[lambda cfg: (rows * cfg["SPLIT"],)](
            x,
            weight,
            bias,
            out,
            (spatial, depth, num_groups, eps, _pow2(span), rows),
        )
    elif span < 512:
        _whole[(rows,)](
            x,
            weight,
            bias,
            out,
            (spatial, depth, num_groups, eps, _pow2(span), rows),
        )
    else:
        chunks = -(-span // 16384)
        partial = torch.empty(
            rows * chunks * 2, dtype=torch.float32, device=out.device
        )
        _partial[(rows * chunks,)](x, partial, (span, chunks, 16384, rows))
        _apply[(rows * chunks,)](
            x,
            weight,
            bias,
            partial,
            out,
            (
                spatial,
                depth,
                num_groups,
                chunks,
                _pow2(chunks),
                eps,
                16384,
                rows,
            ),
        )
    return out


__all__ = ["group_norm_silu"]
