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


@triton.jit
def _divide_corrected(x, scale):
    inverse = 1.0 / scale
    q = x * inverse
    residual = tl.fma(-q, scale, x)
    corrected = tl.fma(residual, inverse, q)
    return tl.where((q == 0.0) | (tl.abs(q) == float("inf")), q, corrected)


@triton.jit
def _divide_group_safe(x, scale):
    inverse = 1.0 / scale
    finite_scale = tl.where(scale == float("inf"), 1.0, scale)
    low = tl.fma(-inverse, finite_scale, 1.0) * inverse
    return tl.fma(x, inverse, x * low)


@triton.jit
def _max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _encode_magnitude(v):
    mag = v.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(mag >> 23, 121)
    magic = exponent + 20 << 23
    rounded = (v + magic.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    code = rounded - magic + (exponent - 121 << 3)
    return tl.minimum(code, 127)


@triton.jit
def _quantize_value(
    x, scale, EPS: tl.constexpr, LO: tl.constexpr, HI: tl.constexpr
):
    if LO == -HI and HI > 0 and (EPS > 0):
        if HI == 448.0 and EPS >= 5.266214694141234e-36:
            q = _divide_group_safe(tl.abs(x), scale[:, None])
        else:
            q = tl.minimum(
                _divide_corrected(tl.abs(x), scale[:, None]),
                HI,
                propagate_nan=tl.PropagateNan.ALL,
            )
        sign = x.to(tl.uint32, bitcast=True) >> 24 & 128
    else:
        signed = tl.minimum(
            tl.maximum(
                _divide_corrected(x, scale[:, None]),
                LO,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            HI,
            propagate_nan=tl.PropagateNan.ALL,
        )
        sign = signed.to(tl.uint32, bitcast=True) >> 24 & 128
        q = tl.abs(signed)
    return (_encode_magnitude(q) | sign).to(tl.uint8)


@triton.jit
def _quant(
    X: tl.pointer_type(tl.bfloat16),
    Q: tl.pointer_type(tl.uint8),
    S: tl.pointer_type(tl.float32),
    N: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    EPS: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
    BG: tl.constexpr,
    BR: tl.constexpr,
):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, BG)
    ng = H // G
    row = r // ng
    col = (r % ng)[:, None] * G + c[None, :]
    x = tl.load(
        X + row[:, None] * X0 + col * X1,
        (r[:, None] < N * ng) & (c[None, :] < G),
        other=0,
    ).to(tl.float32)
    amax = tl.maximum(
        tl.max(tl.abs(x).to(tl.int32, bitcast=True), 1).to(
            tl.float32, bitcast=True
        ),
        EPS,
        propagate_nan=tl.PropagateNan.ALL,
    )
    if HI == 448.0 and EPS >= 1e-10:
        scale = tl.fma(
            amax, 0.002232142724096775, amax * 1.3304608803554885e-10
        )
    else:
        scale = tl.div_rn(amax, HI)
    bits = _quantize_value(x, scale, EPS, LO, HI)
    tl.store(
        Q + row[:, None] * Q0 + col * Q1,
        bits,
        (r[:, None] < N * ng) & (c[None, :] < G),
    )
    tl.store(S + row * S0 + r % ng * S1, scale, r < N * ng)


@triton.jit
def _dense(
    X: tl.pointer_type(tl.bfloat16),
    Q: tl.pointer_type(tl.uint8),
    S: tl.pointer_type(tl.float32),
    P: tl.constexpr,
):
    R: tl.constexpr = P[0]
    G: tl.constexpr = P[1] >> 8
    EPS: tl.constexpr = P[2]
    LO: tl.constexpr = P[3]
    HI: tl.constexpr = P[4]
    BG: tl.constexpr = triton.next_power_of_2(G)
    BR: tl.constexpr = P[1] & 255
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, BG)
    if R % BR == 0 and G == BG:
        x = tl.load(X + r[:, None] * G + c[None, :]).to(tl.float32)
    else:
        x = tl.load(
            X + r[:, None] * G + c[None, :],
            (r[:, None] < R) & (c[None, :] < G),
            other=0,
        ).to(tl.float32)
    amax = tl.maximum(
        tl.max(tl.abs(x).to(tl.int32, bitcast=True), 1).to(
            tl.float32, bitcast=True
        ),
        EPS,
        propagate_nan=tl.PropagateNan.ALL,
    )
    if HI == 448.0 and EPS >= 1e-10:
        scale = tl.fma(
            amax, 0.002232142724096775, amax * 1.3304608803554885e-10
        )
    else:
        scale = tl.div_rn(amax, HI)
    bits = _quantize_value(x, scale, EPS, LO, HI)
    if R % BR == 0 and G == BG:
        tl.store(Q + r[:, None] * G + c[None, :], bits)
        tl.store(S + r, scale)
    else:
        tl.store(
            Q + r[:, None] * G + c[None, :],
            bits,
            (r[:, None] < R) & (c[None, :] < G),
        )
        tl.store(S + r, scale, r < R)


@triton.jit
def _packed_pair(
    X: tl.pointer_type(tl.int32),
    Q: tl.pointer_type(tl.uint16),
    S: tl.pointer_type(tl.float32),
    P: tl.constexpr,
):
    G: tl.constexpr = P[1] >> 8
    BR: tl.constexpr = P[1] & 255
    EPS: tl.constexpr = P[2]
    LO: tl.constexpr = P[3]
    HI: tl.constexpr = P[4]
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, G // 2)
    word = tl.load(X + r[:, None] * (G // 2) + c[None, :])
    x0 = (word << 16).to(tl.float32, bitcast=True)
    x1 = (word & -65536).to(tl.float32, bitcast=True)
    magnitude = tl.maximum(
        tl.abs(x0).to(tl.int32, bitcast=True),
        tl.abs(x1).to(tl.int32, bitcast=True),
    )
    amax = tl.maximum(
        tl.max(magnitude, 1).to(tl.float32, bitcast=True),
        EPS,
        propagate_nan=tl.PropagateNan.ALL,
    )
    if HI == 448.0 and EPS >= 1e-10:
        scale = tl.fma(
            amax, 0.002232142724096775, amax * 1.3304608803554885e-10
        )
    else:
        scale = tl.div_rn(amax, HI)
    bits0 = _quantize_value(x0, scale, EPS, LO, HI).to(tl.uint32)
    bits1 = _quantize_value(x1, scale, EPS, LO, HI).to(tl.uint32)
    packed = (bits0 | (bits1 << 8)).to(tl.uint16)
    tl.store(Q + r[:, None] * (G // 2) + c[None, :], packed)
    tl.store(S + r, scale)


def per_token_group_quant_8bit_v2(
    input, output_q, output_s, group_size, eps, min_8bit, max_8bit
):
    n, h = input.shape
    rows = n * (h // group_size)
    br = 32 if n * h >= 2097152 else 16 if rows >= 1024 else 4
    if (
        rows
        and n * h >= 2097152
        and h % group_size == 0
        and rows % 32 == 0
        and 2 <= group_size <= 128
        and group_size & (group_size - 1) == 0
        and input.is_contiguous()
        and output_q.is_contiguous()
        and output_s.is_contiguous()
        and input.data_ptr() % 4 == 0
        and output_q.data_ptr() % 2 == 0
    ):
        packed_br = 8 if n * h >= 3145728 else 32
        packed_warps = 1 if n * h >= 3145728 else 4
        _packed_pair.run(
            input.data_ptr(),
            output_q.data_ptr(),
            output_s.data_ptr(),
            (rows, group_size << 8 | packed_br, eps, min_8bit, max_8bit),
            grid=(rows // packed_br,),
            warmup=False,
            num_warps=packed_warps,
        )
    elif (
        rows
        and h % group_size == 0
        and input.is_contiguous()
        and output_q.is_contiguous()
        and output_s.is_contiguous()
    ):
        _dense.run(
            input.data_ptr(),
            output_q.data_ptr(),
            output_s.data_ptr(),
            (rows, group_size << 8 | br, eps, min_8bit, max_8bit),
            grid=(triton.cdiv(rows, br),),
            warmup=False,
        )
    elif rows:
        _quant.run(
            input.data_ptr(),
            output_q.data_ptr(),
            output_s.data_ptr(),
            n,
            h,
            group_size,
            *input.stride(),
            *output_q.stride(),
            *output_s.stride(),
            eps,
            min_8bit,
            max_8bit,
            triton.next_power_of_2(group_size),
            br,
            grid=(triton.cdiv(rows, br),),
            warmup=False,
        )
    return (output_q, output_s)


__all__ = ["per_token_group_quant_8bit_v2"]
