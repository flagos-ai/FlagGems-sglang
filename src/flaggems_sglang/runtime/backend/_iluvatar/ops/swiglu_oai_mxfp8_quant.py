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
def _encode(v):
    bits = v.to(tl.uint32, bitcast=True)
    sign = bits >> 24 & 128
    mag = bits & 2147483647
    rounded = mag + 524287 + (mag >> 20 & 1)
    normal = tl.minimum((rounded >> 20) - 960, 126)
    biased = tl.abs(v) + 16384.0
    subnormal = biased.to(tl.uint32, bitcast=True) - 1182793728
    code = tl.where(tl.abs(v) < 0.015625, subnormal, normal)
    code = tl.where(v != v, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _tile(
    X,
    Q,
    S,
    N: tl.constexpr,
    G: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    R: tl.constexpr,
    tile,
):
    col = tl.arange(0, 32)
    group = tile * R + tl.arange(0, R)
    row = group // (N // 32)
    c = (group % (N // 32))[:, None] * 32 + col[None, :]
    address = row[:, None] * SX + c * CX
    if G % R == 0:
        gate = tl.load(X + address).to(tl.float32)
        up = tl.load(X + address + N * CX).to(tl.float32)
    else:
        gate = tl.load(X + address, group[:, None] < G, 0).to(tl.float32)
        up = tl.load(X + address + N * CX, group[:, None] < G, 0).to(
            tl.float32
        )
    if LIMIT is not None:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    sig = 1.0 / (1.0 + tl.exp(-(ALPHA * gate)))
    activated = gate * sig * (up + BETA)
    maximum = tl.maximum(tl.max(tl.abs(activated), 1), 1e-30)
    exponent = tl.minimum(
        tl.maximum(
            tl.ceil(tl.log2(maximum * 0.002232142857142857)) + 127.0, 0.0
        ),
        254.0,
    )
    if G > 8192:
        inverse_bits = tl.where(
            exponent == 254, 4194304, (254 - exponent).to(tl.int32) << 23
        )
        inverse = inverse_bits.to(tl.int32).to(tl.float32, bitcast=True)
        v = activated * inverse[:, None]
    else:
        scale = (exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        scale = tl.where(exponent == 0, 5.877471754111438e-39, scale)
        v = activated / scale[:, None]
    out = group[:, None] * 32 + col[None, :]
    tl.store(Q + out, _encode(v), group[:, None] < G)
    tl.store(S + group, exponent.to(tl.uint8), group < G)


@triton.jit
def _kernel(
    X,
    Q,
    S,
    N: tl.constexpr,
    G: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    R: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
):
    if SINGLE_TILE:
        tile = tl.program_id(0)
        _tile(X, Q, S, N, G, SX, CX, ALPHA, BETA, LIMIT, R, tile)
    else:
        for tile in range(tl.program_id(0), tl.cdiv(G, R), tl.num_programs(0)):
            _tile(X, Q, S, N, G, SX, CX, ALPHA, BETA, LIMIT, R, tile)


def swiglu_oai_mxfp8_quant(gate_up, alpha, beta, limit):
    shape = gate_up.shape
    width = shape[-1]
    n = width // 2
    if n % 32:
        raise ValueError("half of the last dimension must be divisible by 32")
    x = gate_up if gate_up.ndim == 2 else gate_up.reshape(-1, width)
    groups = x.shape[0] * (n // 32)
    q = torch.empty((*shape[:-1], n), dtype=torch.uint8, device=x.device)
    scales = torch.empty(
        (*shape[:-1], n // 32), dtype=torch.uint8, device=x.device
    )
    if groups:
        rows = (
            32
            if groups > 8192 and groups % 32 == 0
            else 8 if groups % 8 == 0 else 4
        )
        grid = min(triton.cdiv(groups, rows), 4096)
        _kernel.run(
            x,
            q,
            scales,
            n,
            groups,
            x.stride(0),
            x.stride(1),
            alpha,
            beta,
            limit,
            rows,
            grid == triton.cdiv(groups, rows),
            num_warps=(
                1
                if groups >= 2048 and groups <= 8192
                else 2 if groups <= 8192 else 4
            ),
            enable_fp_fusion=False,
            grid=(grid,),
            warmup=False,
        )
    return (q.view(torch.float8_e4m3fn), scales)


__all__ = ["swiglu_oai_mxfp8_quant"]
