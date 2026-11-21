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
from triton.runtime.jit import TensorWrapper


class _WordTensor(TensorWrapper):

    def __init__(self, base, dtype):
        self.dtype = dtype
        self.base = base
        self.data = base
        self.device = base.device
        self.shape = base.shape


@triton.jit
def _encode(v):
    raw = v.to(tl.int32, bitcast=True)
    mag = tl.abs(v).to(tl.int32, bitcast=True)
    a = mag.to(tl.float32, bitcast=True)
    exponent = tl.maximum(mag & 2139095040, 1015021568)
    bias = exponent + 167772160
    biased = (a + bias.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
    code = biased.to(tl.int16) + (bias >> 20).to(tl.int16) - 1128
    code = tl.where(mag > 1139277824, 127, code)
    return (code | (raw >> 24 & 128).to(tl.int16)).to(tl.uint16)


@triton.jit
def _kernel(X, Q, S, NT, META: tl.constexpr):
    D: tl.constexpr = META[0]
    B: tl.constexpr = META[1]
    C: tl.constexpr = META[2]
    P: tl.constexpr = META[3]
    XS: tl.constexpr = META[4]
    XI: tl.constexpr = META[5]
    QS: tl.constexpr = META[6]
    QI: tl.constexpr = META[7]
    PACK: tl.constexpr = META[8]
    scale = tl.load(S)
    n = tl.load(NT)
    lane = tl.arange(0, B)
    if XI == 1 and QI == 1:
        sr = scale.to(tl.int32, bitcast=True)
        sm = sr & 2147483647
        if (sr & -2139095041 == 0) & (sm >= 226492416) & (sm <= 1895825408):
            for tile in range(tl.program_id(0), n * C, P):
                row = tile // C
                col = tile % C * B + lane
                src = X + row * XS + col * XI
                g = tl.load(src, col < D, 0.0).to(tl.float32)
                u = tl.load(src + D * XI, col < D, 0.0).to(tl.float32)
                v = g / (1.0 + tl.exp(-g)) * u
                if PACK:
                    code16 = _encode_scaled(v, sr)
                    lo, hi = tl.split(tl.reshape(code16, (B // 2, 2)))
                    packed = lo | hi << 8
                    pair = tile % C * (B // 2) + tl.arange(0, B // 2)
                    tl.store(Q + row * (QS // 2) + pair, packed, pair < D // 2)
                else:
                    tl.store(
                        Q + row * QS + col * QI, _encode_scaled(v, sr), col < D
                    )
        else:
            for tile in range(tl.program_id(0), n * C, P):
                row = tile // C
                col = tile % C * B + lane
                src = X + row * XS + col * XI
                g = tl.load(src, col < D, 0.0).to(tl.float32)
                u = tl.load(src + D * XI, col < D, 0.0).to(tl.float32)
                v = g / (1.0 + tl.exp(-g)) * u / scale
                if PACK:
                    code16 = _encode(v)
                    lo, hi = tl.split(tl.reshape(code16, (B // 2, 2)))
                    packed = lo | hi << 8
                    pair = tile % C * (B // 2) + tl.arange(0, B // 2)
                    tl.store(Q + row * (QS // 2) + pair, packed, pair < D // 2)
                else:
                    tl.store(Q + row * QS + col * QI, _encode(v), col < D)
    else:
        for tile in range(tl.program_id(0), n * C, P):
            row = tile // C
            col = tile % C * B + lane
            src = X + row * XS + col * XI
            g = tl.load(src, col < D, 0.0).to(tl.float32)
            u = tl.load(src + D * XI, col < D, 0.0).to(tl.float32)
            v = g / (1.0 + tl.exp(-g)) * u / scale
            if PACK:
                code16 = _encode(v)
                lo, hi = tl.split(tl.reshape(code16, (B // 2, 2)))
                packed = lo | hi << 8
                pair = tile % C * (B // 2) + tl.arange(0, B // 2)
                tl.store(Q + row * (QS // 2) + pair, packed, pair < D // 2)
            else:
                tl.store(Q + row * QS + col * QI, _encode(v), col < D)


def silu_mul_static_tensorwise_quant(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    m = input.shape[0]
    if m == 0:
        return output
    xs, xi = input.stride()
    qs, qi = output.stride()
    block = min(
        256 if xi != 1 or qi != 1 else 4096,
        triton.next_power_of_2(intermediate_size),
    )
    chunks = triton.cdiv(intermediate_size, block)
    programs = min(40, min(max(1, expected_num_tokens), m) * chunks)
    pack = (
        intermediate_size % 2 == 0
        and qi == 1
        and (qs % 2 == 0)
        and (output.storage_offset() % 2 == 0)
    )
    _kernel.run(
        input,
        _WordTensor(output, tl.uint16 if pack else tl.uint8),
        scale,
        num_tokens_tensor,
        (intermediate_size, block, chunks, programs, xs, xi, qs, qi, pack),
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    return output


@triton.jit
def _encode_scaled(v, scale_bits):
    raw = v.to(tl.int32, bitcast=True)
    mag = tl.abs(v).to(tl.int32, bitcast=True)
    a = mag.to(tl.float32, bitcast=True)
    exponent = tl.maximum(mag & 2139095040, scale_bits - 50331648)
    bias = exponent + 167772160
    biased = (a + bias.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
    code = (
        biased.to(tl.int16)
        + (bias >> 20).to(tl.int16)
        - ((scale_bits >> 20) + 112).to(tl.int16)
    )
    code = tl.where(mag > scale_bits + 73924608, 127, code)
    return (code | (raw >> 24 & 128).to(tl.int16)).to(tl.uint16)


__all__ = ["silu_mul_static_tensorwise_quant"]
