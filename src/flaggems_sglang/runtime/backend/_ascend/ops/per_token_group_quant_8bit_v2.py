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
def _max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _packed_code(
    x,
    scale,
    LO: tl.constexpr,
    HI: tl.constexpr,
    EPS: tl.constexpr,
    FUSED: tl.constexpr = False,
):
    if LO == -448.0 and HI == 448.0 and (EPS >= 5.266214694141234e-36):
        r = 1.0 / scale
        finite_scale = tl.where(scale == float("inf"), 1.0, scale)
        high = (r.to(tl.int32, bitcast=True) & -4096).to(
            tl.float32, bitcast=True
        )
        sh = (finite_scale.to(tl.int32, bitcast=True) & -4096).to(
            tl.float32, bitcast=True
        )
        sl = finite_scale - sh
        residual = 1.0 - high * sh - high * sl
        low = residual * r
        value = tl.abs(x)
        if FUSED:
            q = tl.fma(value, high, value * low)
        else:
            q = value * high + value * low
        code = _encode_positive(q, x)
        sign = (x.to(tl.int32, bitcast=True) >> 24 & 128).to(tl.int16)
        code = tl.where(q != q, 127, code) | sign
        return code
    else:
        return _boundary_code(x, scale, LO, HI)


@triton.jit
def _group_code(
    x,
    scale,
    LO: tl.constexpr,
    HI: tl.constexpr,
    EPS: tl.constexpr,
    FUSED: tl.constexpr = False,
):
    if LO == -448.0 and HI == 448.0 and (EPS >= 5.266214694141234e-36):
        r = 1.0 / scale
        finite_scale = tl.where(scale == float("inf"), 1.0, scale)
        high = (r.to(tl.int32, bitcast=True) & -4096).to(
            tl.float32, bitcast=True
        )
        sh = (finite_scale.to(tl.int32, bitcast=True) & -4096).to(
            tl.float32, bitcast=True
        )
        sl = finite_scale - sh
        residual = 1.0 - high * sh - high * sl
        low = residual * r
        value = tl.abs(x)
        if FUSED:
            q = tl.fma(value, high, value * low)
        else:
            q = value * high + value * low
        code = _encode_positive(q, x)
        sign = (x.to(tl.int32, bitcast=True) >> 24 & 128).to(tl.int16)
        code = tl.where(q != q, 127, code) | sign
        return code.to(tl.uint8)
    else:
        return _boundary_code(x, scale, LO, HI)


@triton.jit
def _boundary_code(x, scale, LO: tl.constexpr, HI: tl.constexpr):
    ax = tl.abs(x)
    v = ax / scale
    mag = v.to(tl.int32, bitcast=True) & 2147483647
    lower = (mag >> 20) - 960
    midpoint = (mag & 2146435072 | 524288).to(tl.float32, bitcast=True)
    sub = (v * 512.0).to(tl.int32)
    midpoint = tl.where(
        mag < 1015021568, (sub.to(tl.float32) + 0.5) * 0.001953125, midpoint
    )
    lower = tl.where(mag < 1015021568, sub, lower)
    exponent = midpoint.to(tl.int32, bitcast=True) >> 23 & 255
    half_ulp = (exponent - 24 << 23).to(tl.float32, bitcast=True)
    high = (scale.to(tl.int32, bitcast=True) & -256).to(
        tl.float32, bitcast=True
    )
    low = scale - high
    residual = ax - midpoint * high - midpoint * low
    threshold = scale * half_ulp
    up = tl.where(lower & 1 != 0, residual >= -threshold, residual > threshold)
    code = tl.minimum(lower + up.to(tl.int32), 126)
    code = tl.where(scale == float("inf"), 0, code)
    sign = x.to(tl.int32, bitcast=True) >> 24 & 128
    signed_code = tl.where(sign != 0, -code, code)
    lo = _encode(tl.full((), LO, tl.float32)).to(tl.int32)
    hi = _encode(tl.full((), HI, tl.float32)).to(tl.int32)
    lo_order = tl.where(lo & 128 != 0, -(lo & 127), lo & 127)
    hi_order = tl.where(hi & 128 != 0, -(hi & 127), hi & 127)
    result = tl.where(
        signed_code < lo_order,
        lo,
        tl.where(signed_code > hi_order, hi, code | sign),
    )
    result = tl.where(
        (scale != scale) | (ax == float("inf")) | (v != v), 127 | sign, result
    )
    return result.to(tl.uint8)


@triton.jit
def _encode(v):
    bits = v.to(tl.int32, bitcast=True)
    mag = bits & 2147483647
    sign = bits >> 24 & 128
    exponent = tl.maximum(mag >> 23, 121)
    magic = exponent + 20 << 23
    rounded = (tl.abs(v) + magic.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    code = (rounded & 31) + (exponent - 121 << 3)
    return (tl.minimum(code, 127) | sign).to(tl.uint8)


@triton.jit
def _encode_positive(v, x):
    bits = v.to(tl.int32, bitcast=True)
    exponent = tl.maximum(bits & 2139095040, 1015021568)
    magic = exponent + 167772160
    rounded = (v + magic.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    code = rounded.to(tl.int16) + (exponent >> 20).to(tl.int16) - 968
    return code


@triton.jit
def _quant(
    X: tl.pointer_type(tl.bfloat16),
    Q: tl.pointer_type(tl.uint8),
    S: tl.pointer_type(tl.float32),
    P: tl.constexpr,
):
    R: tl.constexpr = P[0]
    G: tl.constexpr = P[1] >> 8
    BR: tl.constexpr = P[1] & 255
    EPS: tl.constexpr = P[2]
    LO: tl.constexpr = P[3]
    HI: tl.constexpr = P[4]
    BG: tl.constexpr = triton.next_power_of_2(G)
    for block in range(tl.program_id(0), tl.cdiv(R, BR), tl.num_programs(0)):
        r = block * BR + tl.arange(0, BR)
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
            tl.reduce(tl.abs(x), 1, _max_nan),
            EPS,
            propagate_nan=tl.PropagateNan.ALL,
        )
        scale = tl.div_rn(amax, HI)
        bits = _group_code(x, scale[:, None], LO, HI, EPS)
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
def _split(
    X: tl.pointer_type(tl.bfloat16),
    Q: tl.pointer_type(tl.uint8),
    S: tl.pointer_type(tl.float32),
    P: tl.constexpr,
):
    R: tl.constexpr = P[0]
    G: tl.constexpr = P[1] >> 8
    BR: tl.constexpr = P[1] & 255
    EPS: tl.constexpr = P[2]
    LO: tl.constexpr = P[3]
    HI: tl.constexpr = P[4]
    for block in range(tl.program_id(0), tl.cdiv(R, BR), tl.num_programs(0)):
        r = block * BR + tl.arange(0, BR)
        c = tl.arange(0, G // 2)
        x0 = tl.load(X + r[:, None] * G + c[None, :]).to(tl.float32)
        x1 = tl.load(X + r[:, None] * G + c[None, :] + G // 2).to(tl.float32)
        amax = tl.maximum(
            tl.maximum(
                tl.reduce(tl.abs(x0), 1, _max_nan),
                tl.reduce(tl.abs(x1), 1, _max_nan),
                propagate_nan=tl.PropagateNan.ALL,
            ),
            EPS,
            propagate_nan=tl.PropagateNan.ALL,
        )
        scale = tl.div_rn(amax, HI)
        bits0 = _group_code(x0, scale[:, None], LO, HI, EPS)
        tl.store(Q + r[:, None] * G + c[None, :], bits0)
        bits1 = _group_code(x1, scale[:, None], LO, HI, EPS)
        tl.store(Q + r[:, None] * G + c[None, :] + G // 2, bits1)
        tl.store(S + r, scale)


@triton.jit
def _packed_pair(
    X: tl.pointer_type(tl.int32),
    Q: tl.pointer_type(tl.int16),
    S: tl.pointer_type(tl.float32),
    P: tl.constexpr,
):
    R: tl.constexpr = P[0]
    G: tl.constexpr = P[1] >> 8
    BR: tl.constexpr = P[1] & 255
    EPS: tl.constexpr = P[2]
    LO: tl.constexpr = P[3]
    HI: tl.constexpr = P[4]
    for block in range(tl.program_id(0), tl.cdiv(R, BR), tl.num_programs(0)):
        r = block * BR + tl.arange(0, BR)
        c = tl.arange(0, G // 2)
        word = tl.load(X + r[:, None] * (G // 2) + c[None, :])
        x0 = (word << 16).to(tl.float32, bitcast=True)
        x1 = (word & -65536).to(tl.float32, bitcast=True)
        magnitudes = tl.maximum(
            tl.abs(x0), tl.abs(x1), propagate_nan=tl.PropagateNan.ALL
        )
        amax = tl.maximum(
            tl.reduce(magnitudes, 1, _max_nan),
            EPS,
            propagate_nan=tl.PropagateNan.ALL,
        )
        scale = tl.div_rn(amax, HI)
        tl.store(S + r, scale)
        bits0 = _packed_magnitude_code(x0, scale[:, None], LO, HI, EPS, False)
        bits1 = _packed_magnitude_code(x1, scale[:, None], LO, HI, EPS, False)
        signs = ((word >> 8 & 128) | (word >> 16 & 32768)).to(tl.int16)
        packed = bits0.to(tl.int16) | (bits1.to(tl.int16) << 8) | signs
        tl.store(Q + r[:, None] * (G // 2) + c[None, :], packed)


@triton.jit
def _fold_reduce(
    X: tl.pointer_type(tl.bfloat16),
    Q: tl.pointer_type(tl.uint8),
    S: tl.pointer_type(tl.float32),
    P: tl.constexpr,
):
    R: tl.constexpr = P[0]
    G: tl.constexpr = P[1] >> 8
    BR: tl.constexpr = P[1] & 255
    EPS: tl.constexpr = P[2]
    LO: tl.constexpr = P[3]
    HI: tl.constexpr = P[4]
    for block in range(tl.program_id(0), tl.cdiv(R, BR), tl.num_programs(0)):
        r = block * BR + tl.arange(0, BR)
        c = tl.arange(0, G // 2)
        x0 = tl.load(X + r[:, None] * G + c[None, :]).to(tl.float32)
        x1 = tl.load(X + r[:, None] * G + c[None, :] + 1 * (G // 2)).to(
            tl.float32
        )
        magnitudes = tl.maximum(
            tl.abs(x0), tl.abs(x1), propagate_nan=tl.PropagateNan.ALL
        )
        amax = tl.maximum(
            tl.reduce(magnitudes, 1, _max_nan),
            EPS,
            propagate_nan=tl.PropagateNan.ALL,
        )
        scale = tl.div_rn(amax, HI)
        bits0 = _group_code(x0, scale[:, None], LO, HI, EPS, False)
        tl.store(Q + r[:, None] * G + c[None, :], bits0)
        bits1 = _group_code(x1, scale[:, None], LO, HI, EPS, False)
        tl.store(Q + r[:, None] * G + c[None, :] + 1 * (G // 2), bits1)
        tl.store(S + r, scale)


@triton.jit
def _wide_scale(
    X: tl.pointer_type(tl.bfloat16),
    S: tl.pointer_type(tl.float32),
    R: tl.constexpr,
    G: tl.constexpr,
    EPS: tl.constexpr,
    HI: tl.constexpr,
):
    c = tl.arange(0, triton.next_power_of_2(G))
    one = tl.arange(0, 1)
    for r in range(tl.program_id(0), R, tl.num_programs(0)):
        x = tl.load(X + r * G + c[None, :], c[None, :] < G, other=0).to(
            tl.float32
        )
        amax = tl.maximum(
            tl.reduce(tl.abs(x), 1, _max_nan),
            EPS,
            propagate_nan=tl.PropagateNan.ALL,
        )
        tl.store(S + r + one, tl.div_rn(amax, HI))


@triton.jit
def _wide_apply(
    X: tl.pointer_type(tl.bfloat16),
    Q: tl.pointer_type(tl.uint8),
    S: tl.pointer_type(tl.float32),
    R: tl.constexpr,
    G: tl.constexpr,
    EPS: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
):
    C: tl.constexpr = triton.cdiv(G, 128)
    one = tl.arange(0, 1)
    for tile in range(tl.program_id(0), R * C, tl.num_programs(0)):
        r = tile // C
        c = tile % C * 128 + tl.arange(0, 128)
        x = tl.load(X + r * G + c[None, :], c[None, :] < G, other=0).to(
            tl.float32
        )
        scale = tl.load(S + r + one)
        code = _group_code(x, scale[:, None], LO, HI, EPS)
        tl.store(Q + r * G + c[None, :], code, c[None, :] < G)


@triton.jit
def _packed_magnitude_code(
    x,
    scale,
    LO: tl.constexpr,
    HI: tl.constexpr,
    EPS: tl.constexpr,
    FUSED: tl.constexpr = False,
):
    if LO == -448.0 and HI == 448.0 and (EPS >= 5.266214694141234e-36):
        r = 1.0 / scale
        finite_scale = tl.where(scale == float("inf"), 1.0, scale)
        high = (r.to(tl.int32, bitcast=True) & -4096).to(
            tl.float32, bitcast=True
        )
        sh = (finite_scale.to(tl.int32, bitcast=True) & -4096).to(
            tl.float32, bitcast=True
        )
        sl = finite_scale - sh
        residual = 1.0 - high * sh - high * sl
        low = residual * r
        value = tl.abs(x)
        if FUSED:
            q = tl.fma(value, high, value * low)
        else:
            q = value * high + value * low
        code = _encode_positive(q, x)
        code = tl.where(q != q, 127, code)
        return code
    else:
        return _boundary_code(x, scale, LO, HI)


def per_token_group_quant_8bit_v2(
    input, output_q, output_s, group_size, eps, min_8bit, max_8bit
):
    original_q, original_s = (output_q, output_s)
    input = input.contiguous()
    if not output_q.is_contiguous():
        output_q = torch.empty(
            output_q.shape, dtype=output_q.dtype, device=output_q.device
        )
    if not output_s.is_contiguous():
        output_s = torch.empty(
            output_s.shape, dtype=output_s.dtype, device=output_s.device
        )
    n, h = input.shape
    rows = n * (h // group_size)
    br = (
        16
        if group_size <= 128
        else max(1, 1024 // triton.next_power_of_2(group_size))
    )
    split = (
        n * h >= 8388608
        and rows % 64 == 0
        and (2 <= group_size <= 128)
        and (group_size & group_size - 1 == 0)
        and (min_8bit == -448.0)
        and (max_8bit == 448.0)
        and (eps >= 5.266214694141234e-36)
    )
    if split:
        br = 16
    kernel = _split if split else _quant
    if rows % 32 == 0 and group_size <= 128:
        br = 32
        kernel = _quant
    if (
        n * h >= 8388608
        and group_size == 128
        and rows % 64 == 0
        and min_8bit == -448.0
        and max_8bit == 448.0
        and eps >= 5.266214694141234e-36
    ):
        br = 64
        kernel = _fold_reduce
        if input.data_ptr() % 4 == 0 and output_q.data_ptr() % 2 == 0:
            br = 64
            kernel = _packed_pair
    if rows and (group_size > 128 or group_size & group_size - 1):
        _wide_scale.run(
            input.data_ptr(),
            output_s.data_ptr(),
            rows,
            group_size,
            eps,
            max_8bit,
            grid=(min(rows, 40),),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _wide_apply.run(
            input.data_ptr(),
            output_q.data_ptr(),
            output_s.data_ptr(),
            rows,
            group_size,
            eps,
            min_8bit,
            max_8bit,
            grid=(min(rows * triton.cdiv(group_size, 128), 40),),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    elif rows:
        kernel.run(
            input.data_ptr(),
            output_q.data_ptr(),
            output_s.data_ptr(),
            (rows, group_size << 8 | br, eps, min_8bit, max_8bit),
            grid=(min(triton.cdiv(rows, br), 40),),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    if output_q is not original_q:
        original_q.view(torch.uint8).copy_(output_q.view(torch.uint8))
    if output_s is not original_s:
        original_s.copy_(output_s)
    return (original_q, original_s)


__all__ = ["per_token_group_quant_8bit_v2"]
