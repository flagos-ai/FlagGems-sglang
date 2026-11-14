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
    magnitude = bits & 2147483647
    exponent = tl.minimum(tl.maximum(magnitude >> 23, 121), 135)
    bias_bits = exponent + 20 << 23
    biased = magnitude.to(tl.float32, bitcast=True) + bias_bits.to(
        tl.float32, bitcast=True
    )
    units = biased.to(tl.uint32, bitcast=True) - bias_bits
    code = tl.minimum(units + (exponent - 121 << 3), 126)
    code = tl.where(v != v, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _activate(
    gate, up, ALPHA: tl.constexpr, BETA: tl.constexpr, LIMIT: tl.constexpr
):
    if LIMIT is not None:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    return gate * (1.0 / (1.0 + tl.exp(-(ALPHA * gate)))) * (up + BETA)


@triton.jit
def _quantize(activated):
    maximum = tl.maximum(tl.max(tl.abs(activated), 1), 1e-30)
    exponent = tl.minimum(
        tl.maximum(
            tl.ceil(tl.log2(maximum * 0.002232142857142857)) + 127.0, 0.0
        ),
        254.0,
    )
    inverse_bits = tl.where(
        exponent == 254, 4194304, (254 - exponent).to(tl.int32) << 23
    )
    inverse = inverse_bits.to(tl.int32).to(tl.float32, bitcast=True)
    v = (activated * inverse[:, None]).to(tl.float32)
    return (_encode(v), exponent.to(tl.uint8))


@triton.jit(do_not_specialize=["Q", "S"])
def _kernel_tile(
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
    X_ALIGNED: tl.constexpr,
):
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
        gate = tl.load(gate_pointer).to(tl.float32)
        up = tl.load(up_pointer).to(tl.float32)
    else:
        gate = tl.load(gate_pointer, group[:, None] < G, 0).to(tl.float32)
        up = tl.load(up_pointer, group[:, None] < G, 0).to(tl.float32)
    activated = _activate(gate, up, ALPHA, BETA, LIMIT)
    encoded, exponent = _quantize(activated)
    if G % R == 0:
        tl.store(
            tl.multiple_of(Q + group[:, None] * 32, [16, 16]) + col[None, :],
            encoded,
        )
        tl.store(
            tl.multiple_of(S + tile * R, 16 if R >= 16 else R)
            + tl.arange(0, R),
            exponent,
        )
    else:
        tl.store(
            tl.multiple_of(Q + group[:, None] * 32, [16, 16]) + col[None, :],
            encoded,
            group[:, None] < G,
        )
        tl.store(
            tl.multiple_of(S + tile * R, 16 if R >= 16 else R)
            + tl.arange(0, R),
            exponent,
            group < G,
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
    tile = tl.program_id(0)
    _kernel_tile(X, Q, S, N, G, SX, CX, ALPHA, BETA, LIMIT, R, tile, False)


@triton.jit(do_not_specialize=["X", "Q", "S"])
def _kernel_bf16(X, Q, S, META: tl.constexpr):
    N: tl.constexpr = META[0]
    G: tl.constexpr = META[1]
    SX: tl.constexpr = META[2]
    CX: tl.constexpr = META[3]
    ALPHA: tl.constexpr = META[4]
    BETA: tl.constexpr = META[5]
    LIMIT: tl.constexpr = META[6]
    R: tl.constexpr = META[7]
    tile = tl.program_id(0)
    _kernel_tile(X, Q, S, N, G, SX, CX, ALPHA, BETA, LIMIT, R, tile, META[8])


@triton.jit(do_not_specialize=["Q", "S"])
def _contiguous_tile(
    X,
    Q,
    S,
    N: tl.constexpr,
    SX: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    R: tl.constexpr,
    tile,
    X_ALIGNED: tl.constexpr,
):
    offset = tl.arange(0, R * 32)
    per_row = N // (R * 32)
    row = tile // per_row
    column = tile % per_row * R * 32
    address = row * SX + column + offset
    if X_ALIGNED:
        gate_pointer = tl.multiple_of(X + row * SX + column, 16) + offset
        up_pointer = tl.multiple_of(X + row * SX + column + N, 16) + offset
    else:
        gate_pointer = X + address
        up_pointer = X + address + N
    gate = tl.load(gate_pointer).to(tl.float32)
    up = tl.load(up_pointer).to(tl.float32)
    activated = _activate(gate, up, ALPHA, BETA, LIMIT).reshape((R, 32))
    encoded, exponent = _quantize(activated)
    tl.store(
        tl.multiple_of(Q + tile * R * 32, 16) + offset,
        encoded.reshape((R * 32,)),
    )
    tl.store(
        tl.multiple_of(S + tile * R, 16 if R >= 16 else R) + tl.arange(0, R),
        exponent,
    )


@triton.jit(do_not_specialize=["Q", "S"])
def _contiguous(X, Q, S, META: tl.constexpr):
    N: tl.constexpr = META[0]
    SX: tl.constexpr = META[1]
    ALPHA: tl.constexpr = META[2]
    BETA: tl.constexpr = META[3]
    LIMIT: tl.constexpr = META[4]
    R: tl.constexpr = META[5]
    tile = tl.program_id(0)
    _contiguous_tile(X, Q, S, N, SX, ALPHA, BETA, LIMIT, R, tile, False)


@triton.jit(do_not_specialize=["X", "Q", "S"])
def _contiguous_bf16(X, Q, S, META: tl.constexpr):
    N: tl.constexpr = META[0]
    SX: tl.constexpr = META[1]
    ALPHA: tl.constexpr = META[2]
    BETA: tl.constexpr = META[3]
    LIMIT: tl.constexpr = META[4]
    R: tl.constexpr = META[5]
    tile = tl.program_id(0)
    _contiguous_tile(X, Q, S, N, SX, ALPHA, BETA, LIMIT, R, tile, META[6])


def swiglu_oai_mxfp8_quant(gate_up, alpha, beta, limit):
    shape = gate_up.shape
    width = shape[-1]
    n = width // 2
    if n % 32:
        raise ValueError("half of the last dimension must be divisible by 32")
    x = gate_up if gate_up.ndim == 2 else gate_up.reshape(-1, width)
    groups = x.shape[0] * (n // 32)
    sx, cx = x.stride()
    q = x.new_empty((*shape[:-1], n), dtype=torch.float8_e4m3fn)
    scales = x.new_empty((*shape[:-1], n // 32), dtype=torch.uint8)
    if groups:
        q_bytes = q.view(torch.uint8)
        rows = (
            64
            if groups >= 131072 and groups % 64 == 0
            else 16 if groups > 8192 and groups % 16 == 0 else 4
        )
        grid = triton.cdiv(groups, rows)
        warps = 2 if rows == 64 else 1
        if cx == 1 and n // 32 % rows == 0:
            if x.dtype == torch.bfloat16:
                _contiguous_bf16.run(
                    x,
                    q_bytes,
                    scales,
                    (
                        n,
                        sx,
                        alpha,
                        beta,
                        limit,
                        rows,
                        x.storage_offset() % 8 == 0 and sx % 8 == 0,
                    ),
                    num_warps=warps,
                    enable_fp_fusion=False,
                    grid=(grid,),
                    warmup=False,
                )
            else:
                _contiguous.run(
                    x,
                    q_bytes,
                    scales,
                    (n, sx, alpha, beta, limit, rows),
                    num_warps=warps,
                    enable_fp_fusion=False,
                    grid=(grid,),
                    warmup=False,
                )
        elif x.dtype == torch.bfloat16:
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
                    x.storage_offset() % 8 == 0 and sx % 8 == 0,
                ),
                num_warps=warps,
                enable_fp_fusion=False,
                grid=(grid,),
                warmup=False,
            )
        else:
            _kernel.run(
                x,
                q_bytes,
                scales,
                (n, groups, sx, cx, alpha, beta, limit, rows),
                num_warps=warps,
                enable_fp_fusion=False,
                grid=(grid,),
                warmup=False,
            )
    return (q, scales)


__all__ = ["swiglu_oai_mxfp8_quant"]
