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
def _encode_stage_value(v):
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
            tl.ceil(tl.log(maximum / 448.0) * 1.4426950408889634) + 127.0, 0.0
        ),
        254.0,
    )
    inverse_bits = tl.where(
        exponent == 254, 4194304, (254 - exponent).to(tl.int32) << 23
    )
    inverse = inverse_bits.to(tl.int32).to(tl.float32, bitcast=True)
    v = (activated * inverse[:, None]).to(tl.float32)
    return (_encode(v), exponent.to(tl.uint8))


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
    Q_ALIGNED: tl.constexpr,
):
    if Q_ALIGNED:
        Q = tl.multiple_of(Q, 16)
    col = tl.arange(0, 32)
    for tile in range(tl.program_id(0), tl.cdiv(G, R), tl.num_programs(0)):
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
        activated = _activate(gate, up, ALPHA, BETA, LIMIT)
        encoded, exponent = _quantize(activated)
        out = group[:, None] * 32 + col[None, :]
        if G % R == 0:
            tl.store(Q + out, encoded)
            tl.store(S + group, exponent)
        else:
            tl.store(Q + out, encoded, group[:, None] < G)
            tl.store(S + group, exponent, group < G)


@triton.jit
def _contiguous(
    X,
    Q,
    S,
    N: tl.constexpr,
    G: tl.constexpr,
    SX: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    R: tl.constexpr,
    Q_ALIGNED: tl.constexpr,
):
    if Q_ALIGNED:
        Q = tl.multiple_of(Q, 16)
    offset = tl.arange(0, R * 32)
    per_row = N // (R * 32)
    for tile in range(tl.program_id(0), G // R, tl.num_programs(0)):
        row = tile // per_row
        column = tile % per_row * R * 32
        address = row * SX + column + offset
        gate = tl.load(X + address).to(tl.float32)
        up = tl.load(X + address + N).to(tl.float32)
        activated = _activate(gate, up, ALPHA, BETA, LIMIT).reshape((R, 32))
        encoded, exponent = _quantize(activated)
        tl.store(Q + tile * R * 32 + offset, encoded.reshape((R * 32,)))
        tl.store(S + tile * R + tl.arange(0, R), exponent)


@triton.jit
def _encode_stage(
    A,
    Q,
    S,
    G: tl.constexpr,
    B: tl.constexpr,
    NORMALIZED: tl.constexpr,
    Q_ALIGNED: tl.constexpr,
):
    if Q_ALIGNED:
        Q = tl.multiple_of(Q, 16)
    offset = tl.arange(0, B)
    for tile in range(
        tl.program_id(0), tl.cdiv(G * 32, B), tl.num_programs(0)
    ):
        index = tile * B + offset
        if G * 32 % B == 0:
            value = tl.load(A + index)
        else:
            value = tl.load(A + index, index < G * 32, 0)
        if G * 32 % B == 0:
            if not NORMALIZED:
                exponent = tl.load(S + index // 32).to(tl.int32)
                inverse_bits = tl.where(
                    exponent == 254, 4194304, 254 - exponent << 23
                )
                value = value * inverse_bits.to(tl.float32, bitcast=True)
        elif not NORMALIZED:
            exponent = tl.load(S + index // 32, index < G * 32, 127).to(
                tl.int32
            )
            inverse_bits = tl.where(
                exponent == 254, 4194304, 254 - exponent << 23
            )
            value = value * inverse_bits.to(tl.float32, bitcast=True)
        if G * 32 % B == 0:
            tl.store(Q + index, _encode_stage_value(value))
        else:
            tl.store(Q + index, _encode_stage_value(value), index < G * 32)


@triton.jit
def _normalize_stage(
    X,
    A,
    S,
    N: tl.constexpr,
    G: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    R: tl.constexpr,
):
    offset = tl.arange(0, R * 32)
    for tile in range(tl.program_id(0), tl.cdiv(G, R), tl.num_programs(0)):
        index = tile * R * 32 + offset
        if N % (R * 32) == 0 and CX == 1:
            row = tile // (N // (R * 32))
            column = tile % (N // (R * 32)) * (R * 32)
            address = row * SX + column + offset
        else:
            address = index // N * SX + index % N * CX
        if G % R == 0:
            gate = tl.load(X + address).to(tl.float32)
        else:
            gate = tl.load(X + address, index < G * 32, 0).to(tl.float32)
        if G % R == 0:
            up = tl.load(X + address + N * CX).to(tl.float32)
        else:
            up = tl.load(X + address + N * CX, index < G * 32, 0).to(
                tl.float32
            )
        activated = _activate(gate, up, ALPHA, BETA, LIMIT).reshape((R, 32))
        maximum = tl.maximum(tl.max(tl.abs(activated), 1), 1e-30)
        exponent = tl.minimum(
            tl.maximum(
                tl.ceil(tl.log(maximum / 448.0) * 1.4426950408889634) + 127.0,
                0.0,
            ),
            254.0,
        )
        inverse_bits = tl.where(
            exponent == 254, 4194304, (254 - exponent).to(tl.int32) << 23
        )
        inverse = inverse_bits.to(tl.int32).to(tl.float32, bitcast=True)
        value = activated * inverse[:, None]
        if G % R == 0:
            tl.store(A + index, value.reshape((R * 32,)))
        else:
            tl.store(A + index, value.reshape((R * 32,)), index < G * 32)
        group = tile * R + tl.arange(0, R)
        if G % R == 0:
            tl.store(S + group, exponent.to(tl.uint8))
        else:
            tl.store(S + group, exponent.to(tl.uint8), group < G)


@triton.jit
def _normalize_chunked(
    X,
    A,
    S,
    N: tl.constexpr,
    G: tl.constexpr,
    SX: tl.constexpr,
    ALPHA: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    H: tl.constexpr,
):
    offset = tl.arange(0, H * 32)
    chunks_per_row = N // (H * 32)
    for tile in range(tl.program_id(0), tl.cdiv(G, 2 * H), tl.num_programs(0)):
        chunk = tile * 2 + tl.arange(0, 2)
        row = chunk // chunks_per_row
        column = chunk % chunks_per_row * (H * 32)
        address = row[:, None] * SX + column[:, None] + offset[None, :]
        index = chunk[:, None] * (H * 32) + offset[None, :]
        if G % (2 * H) == 0:
            gate = tl.load(X + address).to(tl.float32)
            up = tl.load(X + address + N).to(tl.float32)
        else:
            gate = tl.load(X + address, index < G * 32, 0).to(tl.float32)
            up = tl.load(X + address + N, index < G * 32, 0).to(tl.float32)
        activated = _activate(gate, up, ALPHA, BETA, LIMIT).reshape(
            (2 * H, 32)
        )
        maximum = tl.maximum(tl.max(tl.abs(activated), 1), 1e-30)
        exponent = tl.minimum(
            tl.maximum(
                tl.ceil(tl.log(maximum / 448.0) * 1.4426950408889634) + 127.0,
                0.0,
            ),
            254.0,
        )
        inverse_bits = tl.where(
            exponent == 254, 4194304, (254 - exponent).to(tl.int32) << 23
        )
        inverse = inverse_bits.to(tl.int32).to(tl.float32, bitcast=True)
        value = activated * inverse[:, None]
        group = tile * (2 * H) + tl.arange(0, 2 * H)
        if G % (2 * H) == 0:
            tl.store(A + index, value.reshape((2, H * 32)))
            tl.store(S + group, exponent.to(tl.uint8))
        else:
            tl.store(A + index, value.reshape((2, H * 32)), index < G * 32)
            tl.store(S + group, exponent.to(tl.uint8), group < G)


def swiglu_oai_mxfp8_quant(gate_up, alpha, beta, limit):
    shape = gate_up.shape
    width = shape[-1]
    n = width // 2
    if n % 32:
        raise ValueError("half of the last dimension must be divisible by 32")
    x = gate_up if gate_up.ndim == 2 else gate_up.reshape(-1, width)
    sx, cx = x.stride()
    groups = x.shape[0] * (n // 32)
    q = x.new_empty((*shape[:-1], n), dtype=torch.float8_e4m3fn)
    scales = x.new_empty((*shape[:-1], n // 32), dtype=torch.uint8)
    if groups:
        q_bytes = q.view(torch.uint8)
        if groups >= 524288:
            temporary = x.new_empty((groups * 32,), dtype=torch.float32)
            if cx == 1 and n % 4096 == 0:
                _normalize_chunked.run(
                    x,
                    temporary,
                    scales,
                    n,
                    groups,
                    sx,
                    alpha,
                    beta,
                    limit,
                    128,
                    num_warps=4,
                    enable_fp_fusion=False,
                    grid=(min(triton.cdiv(groups, 256), 40),),
                    warmup=False,
                )
            else:
                _normalize_stage.run(
                    x,
                    temporary,
                    scales,
                    n,
                    groups,
                    sx,
                    cx,
                    alpha,
                    beta,
                    limit,
                    128,
                    num_warps=4,
                    enable_fp_fusion=False,
                    grid=(min(triton.cdiv(groups, 128), 40),),
                    warmup=False,
                )
            _encode_stage.run(
                temporary,
                q_bytes,
                scales,
                groups,
                4096,
                True,
                False,
                num_warps=4,
                enable_fp_fusion=False,
                grid=(min(triton.cdiv(groups * 32, 4096), 40),),
                warmup=False,
            )
        else:
            rows = 64 if groups > 8192 and n // 32 % 64 == 0 else 16
            grid = min(triton.cdiv(groups, rows), 40)
            if cx == 1 and n // 32 % rows == 0:
                _contiguous.run(
                    x,
                    q_bytes,
                    scales,
                    n,
                    groups,
                    sx,
                    alpha,
                    beta,
                    limit,
                    rows,
                    False,
                    num_warps=4,
                    enable_fp_fusion=False,
                    grid=(grid,),
                    warmup=False,
                )
            else:
                _kernel.run(
                    x,
                    q_bytes,
                    scales,
                    n,
                    groups,
                    sx,
                    cx,
                    alpha,
                    beta,
                    limit,
                    rows,
                    False,
                    num_warps=4,
                    enable_fp_fusion=False,
                    grid=(grid,),
                    warmup=False,
                )
    return (q, scales)


__all__ = ["swiglu_oai_mxfp8_quant"]
