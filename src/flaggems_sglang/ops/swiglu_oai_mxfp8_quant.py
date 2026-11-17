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


@triton.jit(do_not_specialize=["Q", "S"])
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
    Q_ALIGNED: tl.constexpr,
    tile,
    X_ALIGNED: tl.constexpr,
):
    if Q_ALIGNED:
        Q = tl.multiple_of(Q, 16)
    col = tl.arange(0, 32)
    group = tile * R + tl.arange(0, R)
    row = group // (N // 32)
    c = (group % (N // 32))[:, None] * 32 + col[None, :]
    address = row[:, None] * SX + c * CX
    if X_ALIGNED:
        base_address = (
            row[:, None] * SX + (group % (N // 32))[:, None] * 32 * CX
        )
        gate_pointer = (
            tl.multiple_of(X + base_address, [16, 16]) + col[None, :] * CX
        )
        up_pointer = (
            tl.multiple_of(X + base_address + N * CX, [16, 16])
            + col[None, :] * CX
        )
    else:
        gate_pointer = X + address
        up_pointer = X + address + N * CX
    if G % R == 0:
        gate = tl.load(
            gate_pointer, cache_modifier=".cg" if G >= 8192 else ""
        ).to(tl.float32)
        up = tl.load(up_pointer, cache_modifier=".cg" if G >= 8192 else "").to(
            tl.float32
        )
    else:
        gate = tl.load(
            gate_pointer,
            group[:, None] < G,
            0,
            cache_modifier=".cg" if G >= 8192 else "",
        ).to(tl.float32)
        up = tl.load(
            up_pointer,
            group[:, None] < G,
            0,
            cache_modifier=".cg" if G >= 8192 else "",
        ).to(tl.float32)
    if LIMIT is not None:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    sig = tl.div_rn(1.0, 1.0 + libdevice.exp(-(ALPHA * gate)))
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
    tl.store(
        tl.multiple_of(Q + group[:, None] * 32, [16, 16]) + col[None, :],
        _encode(v),
        group[:, None] < G,
        cache_modifier=".cs" if G >= 131072 else "",
    )
    tl.store(
        tl.multiple_of(S + tile * R, 16 if R >= 16 else R) + tl.arange(0, R),
        exponent.to(tl.uint8),
        group < G,
        cache_modifier=".cs" if G >= 131072 else "",
    )


@triton.jit(do_not_specialize=["Q", "S"])
def _kernel(X, Q, S, META: tl.constexpr):
    N: tl.constexpr = META[0]
    G: tl.constexpr = META[1]
    SX: tl.constexpr = META[2]
    CX: tl.constexpr = META[3]
    ALPHA: tl.constexpr = META[4]
    BETA: tl.constexpr = META[5]
    LIMIT: tl.constexpr = META[6]
    R: tl.constexpr = META[7]
    Q_ALIGNED: tl.constexpr = META[8]
    SINGLE_TILE: tl.constexpr = META[9]
    if G >= 131072:
        for chunk in tl.range(0, 2, loop_unroll_factor=2):
            tile = tl.program_id(0) * 2 + chunk
            if tl.cdiv(G, R) % 2 == 0:
                _tile(
                    X,
                    Q,
                    S,
                    N,
                    G,
                    SX,
                    CX,
                    ALPHA,
                    BETA,
                    LIMIT,
                    R,
                    Q_ALIGNED,
                    tile,
                    False,
                )
            elif tile < tl.cdiv(G, R):
                _tile(
                    X,
                    Q,
                    S,
                    N,
                    G,
                    SX,
                    CX,
                    ALPHA,
                    BETA,
                    LIMIT,
                    R,
                    Q_ALIGNED,
                    tile,
                    False,
                )
    elif SINGLE_TILE:
        tile = tl.program_id(0)
        _tile(
            X,
            Q,
            S,
            N,
            G,
            SX,
            CX,
            ALPHA,
            BETA,
            LIMIT,
            R,
            Q_ALIGNED,
            tile,
            False,
        )
    else:
        for tile in range(tl.program_id(0), tl.cdiv(G, R), tl.num_programs(0)):
            _tile(
                X,
                Q,
                S,
                N,
                G,
                SX,
                CX,
                ALPHA,
                BETA,
                LIMIT,
                R,
                Q_ALIGNED,
                tile,
                False,
            )


@triton.jit(do_not_specialize=["X", "Q", "S"])
def _kernel_bf16(X, Q, S, META: tl.constexpr):
    if META[10]:
        X = tl.multiple_of(X, 16)
    N: tl.constexpr = META[0]
    G: tl.constexpr = META[1]
    SX: tl.constexpr = META[2]
    CX: tl.constexpr = META[3]
    ALPHA: tl.constexpr = META[4]
    BETA: tl.constexpr = META[5]
    LIMIT: tl.constexpr = META[6]
    R: tl.constexpr = META[7]
    Q_ALIGNED: tl.constexpr = META[8]
    SINGLE_TILE: tl.constexpr = META[9]
    if G >= 131072:
        for chunk in tl.range(0, 2, loop_unroll_factor=2):
            tile = tl.program_id(0) * 2 + chunk
            if tl.cdiv(G, R) % 2 == 0:
                _tile(
                    X,
                    Q,
                    S,
                    N,
                    G,
                    SX,
                    CX,
                    ALPHA,
                    BETA,
                    LIMIT,
                    R,
                    Q_ALIGNED,
                    tile,
                    META[10],
                )
            elif tile < tl.cdiv(G, R):
                _tile(
                    X,
                    Q,
                    S,
                    N,
                    G,
                    SX,
                    CX,
                    ALPHA,
                    BETA,
                    LIMIT,
                    R,
                    Q_ALIGNED,
                    tile,
                    META[10],
                )
    elif SINGLE_TILE:
        tile = tl.program_id(0)
        _tile(
            X,
            Q,
            S,
            N,
            G,
            SX,
            CX,
            ALPHA,
            BETA,
            LIMIT,
            R,
            Q_ALIGNED,
            tile,
            META[10],
        )
    else:
        for tile in range(tl.program_id(0), tl.cdiv(G, R), tl.num_programs(0)):
            _tile(
                X,
                Q,
                S,
                N,
                G,
                SX,
                CX,
                ALPHA,
                BETA,
                LIMIT,
                R,
                Q_ALIGNED,
                tile,
                META[10],
            )


def swiglu_oai_mxfp8_quant(gate_up, alpha, beta, limit):
    shape = gate_up.shape
    width = shape[-1]
    n = width // 2
    blocks_per_row = n // 32
    if n % 32:
        raise ValueError("half of the last dimension must be divisible by 32")
    x = gate_up if gate_up.ndim == 2 else gate_up.reshape(-1, width)
    groups = x.shape[0] * blocks_per_row
    q = x.new_empty((*shape[:-1], n), dtype=torch.float8_e4m3fn)
    scales = x.new_empty((*shape[:-1], blocks_per_row), dtype=torch.uint8)
    if groups:
        q_bytes = q.view(torch.uint8)
        sx, cx = x.stride()
        rows = (
            32
            if groups >= 131072 and groups % 64 == 0
            else 16 if groups > 8192 and groups % 16 == 0 else 4
        )
        tiles = (groups + rows - 1) // rows
        grid = (
            (tiles + 1) // 2
            if groups >= 131072
            else min(tiles, 4096) if rows == 4 else tiles
        )
        if x.dtype == torch.bfloat16:
            _kernel_bf16.run(
                x,
                q_bytes,
                scales,
                (
                    n,
                    groups,
                    sx,
                    cx,
                    alpha,
                    beta,
                    limit,
                    rows,
                    False,
                    grid == tiles,
                    x.storage_offset() % 8 == 0 and sx % 8 == 0,
                ),
                num_warps=4 if groups >= 131072 else 4,
                enable_fp_fusion=True,
                grid=(grid,),
                warmup=False,
            )
        else:
            _kernel.run(
                x,
                q_bytes,
                scales,
                (
                    n,
                    groups,
                    sx,
                    cx,
                    alpha,
                    beta,
                    limit,
                    rows,
                    False,
                    grid == tiles,
                ),
                num_warps=4 if groups >= 131072 else 4,
                enable_fp_fusion=True,
                grid=(grid,),
                warmup=False,
            )
    return (q, scales)


__all__ = ["swiglu_oai_mxfp8_quant"]
