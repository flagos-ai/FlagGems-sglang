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


@tl.core.extern
def _native_rcp(arg0, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [arg0],
        {(tl.float32,): ("llvm.mxc.rcp.f32", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _encode(v):
    bits = v.to(tl.uint32, bitcast=True)
    sign = bits >> 24 & 128
    mag = bits & 2147483647
    rounded = mag + 524287 + (mag >> 20 & 1)
    normal = (rounded >> 20) - 960
    biased = tl.abs(v) + 16384.0
    subnormal = biased.to(tl.uint32, bitcast=True) - 1182793728
    code = tl.where(tl.abs(v) < 0.015625, subnormal, normal)
    code = tl.where(v != v, 127, code)
    return (code | sign).to(tl.uint8)


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
    denom = tl.fma(
        tl.exp2(gate * (-ALPHA * 1.4426950408889634)),
        5.960464477539063e-08,
        5.960464477539063e-08,
    )
    sig = _native_rcp(denom) * 5.960464477539063e-08
    sign = (
        gate.to(tl.uint32, bitcast=True)
        ^ (up + BETA).to(tl.uint32, bitcast=True)
    ) >> 24 & 128
    activated = gate * sig * (up + BETA)
    maximum = tl.maximum(tl.max(tl.abs(activated), 1), 1e-30)
    exponent = tl.minimum(
        tl.maximum(tl.ceil(tl.log2(maximum / 448.0)) + 127.0, 0.0), 254.0
    )
    inverse_bits = tl.where(
        exponent == 254, 4194304, (254 - exponent).to(tl.int32) << 23
    )
    inverse = inverse_bits.to(tl.int32).to(tl.float32, bitcast=True)
    v = activated * inverse[:, None]
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    encoded = _encode(tl.abs(v)) | sign.to(tl.uint8)
    tl.store(
        tl.multiple_of(Q + group[:, None] * 32, [16, 16]) + col[None, :],
        encoded,
        group[:, None] < G,
    )
    tl.store(
        tl.multiple_of(S + tile * R, 16 if R >= 16 else R) + tl.arange(0, R),
        exponent.to(tl.uint8),
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
    SINGLE_TILE: tl.constexpr = META[8]
    if SINGLE_TILE:
        tile = tl.program_id(0)
        _kernel_tile(X, Q, S, N, G, SX, CX, ALPHA, BETA, LIMIT, R, tile)
    else:
        for tile in range(tl.program_id(0), tl.cdiv(G, R), tl.num_programs(0)):
            _kernel_tile(X, Q, S, N, G, SX, CX, ALPHA, BETA, LIMIT, R, tile)


def swiglu_oai_mxfp8_quant(gate_up, alpha, beta, limit):
    shape = gate_up.shape
    width = shape[-1]
    n = width // 2
    if n % 32:
        raise ValueError("half of the last dimension must be divisible by 32")
    x = gate_up if gate_up.ndim == 2 else gate_up.reshape(-1, width)
    groups = x.shape[0] * (n // 32)
    q = x.new_empty((*shape[:-1], n), dtype=torch.float8_e4m3fn)
    scales = x.new_empty((*shape[:-1], n // 32), dtype=torch.uint8)
    if groups:
        q_bytes = q.view(torch.uint8)
        rows = (
            32
            if groups >= 131072 and groups % 32 == 0
            else (
                16
                if groups > 8192 and groups % 16 == 0
                else 8 if groups % 8 == 0 else 4
            )
        )
        _kernel.run(
            x,
            q_bytes,
            scales,
            (
                n,
                groups,
                x.stride(0),
                x.stride(1),
                alpha,
                beta,
                limit,
                rows,
                True,
            ),
            num_warps=(
                1
                if groups >= 131072 and groups % 32 == 0
                else (
                    1
                    if groups >= 2048 and groups <= 8192
                    else 2 if rows >= 16 else 4
                )
            ),
            enable_fp_fusion=False,
            grid=(triton.cdiv(groups, rows),),
            warmup=False,
        )
    return (q, scales)


__all__ = ["swiglu_oai_mxfp8_quant"]
