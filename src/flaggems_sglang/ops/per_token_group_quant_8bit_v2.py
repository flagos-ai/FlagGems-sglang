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

_ROCM = torch.version.hip is not None
if _ROCM:
    gluon = triton
    gl = tl
else:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl


@triton.jit
def _divide_corrected(x, scale, ROCM: tl.constexpr):
    if ROCM:
        inverse = libdevice.fast_dividef(1.0, scale)
    else:
        inverse = 1.0 / scale
    q = x * inverse
    residual = tl.fma(-q, scale, x)
    corrected = tl.fma(residual, inverse, q)
    return tl.where((q == 0.0) | (tl.abs(q) == float("inf")), q, corrected)


@triton.jit
def _divide_group_safe(x, scale, ROCM: tl.constexpr):
    if ROCM:
        inverse = libdevice.fast_dividef(1.0, scale)
    else:
        inverse = 1.0 / scale
    finite_scale = tl.where(scale == float("inf"), 1.0, scale)
    q = x * inverse
    residual = tl.fma(-q, finite_scale, x)
    return tl.fma(residual, inverse, q)


@triton.jit
def _max_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _encode_magnitude(v):
    bits = v.to(tl.uint32, bitcast=True)
    mag = bits
    normal = mag + 3288858623 + (mag >> 20 & 1) >> 20
    sub = (v + 16384.0).to(tl.uint32, bitcast=True) - 1182793728
    result = tl.where(mag < 1015021568, sub, tl.minimum(normal, 127))
    return result


@triton.jit
def _quantize_value(
    x,
    scale,
    EPS: tl.constexpr,
    LO: tl.constexpr,
    HI: tl.constexpr,
    ROCM: tl.constexpr,
):
    if LO == -HI and HI > 0 and (EPS > 0):
        if HI == 448.0 and EPS >= 5.266214694141234e-36:
            q = _divide_group_safe(tl.abs(x), scale[:, None], ROCM)
        else:
            q = tl.minimum(
                _divide_corrected(tl.abs(x), scale[:, None], ROCM),
                HI,
                propagate_nan=tl.PropagateNan.ALL,
            )
        sign = x.to(tl.uint32, bitcast=True) >> 24 & 128
    else:
        signed = tl.minimum(
            tl.maximum(
                _divide_corrected(x, scale[:, None], ROCM),
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
    ROCM: tl.constexpr,
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
        tl.reduce(tl.abs(x), 1, _max_nan),
        tl.cast(EPS, tl.float32),
        propagate_nan=tl.PropagateNan.ALL,
    )
    scale = tl.div_rn(amax, HI)
    bits = _quantize_value(x, scale, EPS, LO, HI, ROCM)
    tl.store(
        Q + row[:, None] * Q0 + col * Q1,
        bits,
        (r[:, None] < N * ng) & (c[None, :] < G),
    )
    tl.store(S + row * S0 + r % ng * S1, scale, r < N * ng)


@triton.jit(do_not_specialize=["X", "Q", "S"])
def _dense(X, Q, S, P: tl.constexpr):
    X = tl.multiple_of(X.to(tl.pointer_type(tl.bfloat16)), 16)
    Q = tl.multiple_of(Q.to(tl.pointer_type(tl.uint8)), 16)
    S = tl.multiple_of(S.to(tl.pointer_type(tl.float32)), 16)
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
        x = tl.load(
            X + r[:, None] * G + c[None, :],
            cache_modifier=".cg" if P[5] else "",
            eviction_policy=(
                "evict_first" if not P[5] and R * G >= 8388608 else ""
            ),
        ).to(tl.float32)
    else:
        x = tl.load(
            X + r[:, None] * G + c[None, :],
            (r[:, None] < R) & (c[None, :] < G),
            other=0,
            cache_modifier=".cg" if P[5] else "",
            eviction_policy=(
                "evict_first" if not P[5] and R * G >= 8388608 else ""
            ),
        ).to(tl.float32)
    amax = tl.maximum(
        tl.reduce(tl.abs(x), 1, _max_nan),
        tl.cast(EPS, tl.float32),
        propagate_nan=tl.PropagateNan.ALL,
    )
    scale = tl.div_rn(amax, HI)
    bits = _quantize_value(x, scale, EPS, LO, HI, P[5])
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


@gluon.jit
def _g_divide_corrected(x, scale, ROCM: gl.constexpr):
    if ROCM:
        inverse = libdevice.fast_dividef(1.0, scale)
    else:
        inverse = 1.0 / scale
    q = x * inverse
    residual = gl.fma(-q, scale, x)
    corrected = gl.fma(residual, inverse, q)
    return gl.where((q == 0.0) | (gl.abs(q) == float("inf")), q, corrected)


@gluon.jit
def _g_divide_group_safe(x, scale, ROCM: gl.constexpr):
    if ROCM:
        inverse = libdevice.fast_dividef(1.0, scale)
    else:
        inverse = 1.0 / scale
    finite_scale = gl.where(scale == float("inf"), 1.0, scale)
    q = x * inverse
    residual = gl.fma(-q, finite_scale, x)
    return gl.fma(residual, inverse, q)


@gluon.jit
def _g_max_nan(a, b):
    return gl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@gluon.jit
def _g_encode_magnitude(v, MAGIC: gl.constexpr):
    if MAGIC:
        mag = v.to(gl.uint32, bitcast=True)
        exponent = gl.maximum(mag >> 23, 121)
        magic = exponent + 20 << 23
        rounded = (v + magic.to(gl.float32, bitcast=True)).to(
            gl.uint32, bitcast=True
        )
        code = rounded - magic + (exponent - 121 << 3)
        return gl.minimum(code, 127)
    else:
        bits = v.to(gl.uint32, bitcast=True)
        mag = bits
        normal = mag + 3288858623 + (mag >> 20 & 1) >> 20
        sub = (v + 16384.0).to(gl.uint32, bitcast=True) - 1182793728
        result = gl.where(mag < 1015021568, sub, gl.minimum(normal, 127))
        return result


@gluon.jit
def _g_quantize_value(
    x,
    scale,
    EPS: gl.constexpr,
    LO: gl.constexpr,
    HI: gl.constexpr,
    ROCM: gl.constexpr,
    MAGIC: gl.constexpr,
):
    if LO == -HI and HI > 0 and (EPS > 0):
        if HI == 448.0 and EPS >= 5.266214694141234e-36:
            q = _g_divide_group_safe(gl.abs(x), scale[:, None], ROCM)
        else:
            q = gl.minimum(
                _g_divide_corrected(gl.abs(x), scale[:, None], ROCM),
                HI,
                propagate_nan=tl.PropagateNan.ALL,
            )
        sign = x.to(gl.uint32, bitcast=True) >> 24 & 128
    else:
        signed = gl.minimum(
            gl.maximum(
                _g_divide_corrected(x, scale[:, None], ROCM),
                LO,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            HI,
            propagate_nan=tl.PropagateNan.ALL,
        )
        sign = signed.to(gl.uint32, bitcast=True) >> 24 & 128
        q = gl.abs(signed)
    return (_g_encode_magnitude(q, MAGIC) | sign).to(gl.uint8)


@gluon.jit(do_not_specialize=["X", "Q", "S"])
def _g_dense(X, Q, S, P: gl.constexpr):
    X = gl.multiple_of(X.to(gl.pointer_type(gl.bfloat16)), 16)
    Q = gl.multiple_of(Q.to(gl.pointer_type(gl.uint8)), 16)
    S = gl.multiple_of(S.to(gl.pointer_type(gl.float32)), 16)
    R: gl.constexpr = P[0]
    G: gl.constexpr = P[1] >> 8
    EPS: gl.constexpr = P[2]
    LO: gl.constexpr = P[3]
    HI: gl.constexpr = P[4]
    BG: gl.constexpr = triton.next_power_of_2(G)
    BR: gl.constexpr = P[1] & 255
    vector: gl.constexpr = 8 if R * G < 16777216 else 16
    layout: gl.constexpr = gl.BlockedLayout(
        [1, vector], [4, 8], [2, 1], [1, 0]
    )
    r = gl.program_id(0) * BR + gl.arange(
        0, BR, layout=gl.SliceLayout(1, layout)
    )
    c = gl.arange(0, BG, layout=gl.SliceLayout(0, layout))
    if R % BR == 0 and G == BG:
        x = gl.load(
            X + r[:, None] * G + c[None, :],
            cache_modifier=".cg" if P[5] else "",
            eviction_policy=(
                "evict_first" if not P[5] and R * G >= 8388608 else ""
            ),
        ).to(gl.float32)
    else:
        x = gl.load(
            X + r[:, None] * G + c[None, :],
            (r[:, None] < R) & (c[None, :] < G),
            other=0,
            cache_modifier=".cg" if P[5] else "",
            eviction_policy=(
                "evict_first" if not P[5] and R * G >= 8388608 else ""
            ),
        ).to(gl.float32)
    amax = gl.maximum(
        gl.reduce(gl.abs(x), 1, _g_max_nan),
        gl.cast(EPS, gl.float32),
        propagate_nan=tl.PropagateNan.ALL,
    )
    scale = gl.div_rn(amax, HI)
    bits = _g_quantize_value(x, scale, EPS, LO, HI, P[5], R * G < 16777216)
    if R % BR == 0 and G == BG:
        gl.store(Q + r[:, None] * G + c[None, :], bits)
        gl.store(S + r, scale)
    else:
        gl.store(
            Q + r[:, None] * G + c[None, :],
            bits,
            (r[:, None] < R) & (c[None, :] < G),
        )
        gl.store(S + r, scale, r < R)


def per_token_group_quant_8bit_v2(
    input, output_q, output_s, group_size, eps, min_8bit, max_8bit
):
    xp = input.data_ptr()
    qp = output_q.data_ptr()
    sp = output_s.data_ptr()
    n, h = input.shape
    rows = n * (h // group_size)
    br = 16 if rows >= 1024 else 4
    if n * h >= 8388608:
        br = 8 if _ROCM else 16
    if (
        not _ROCM
        and 8388608 <= n * h <= 16777216
        and group_size == 128
        and rows % 8 == 0
        and h % group_size == 0
        and input.is_contiguous()
        and output_q.is_contiguous()
        and output_s.is_contiguous()
        and (xp | qp | sp) & 15 == 0
    ):
        _g_dense.run(
            xp,
            qp,
            sp,
            (rows, group_size << 8 | 8, eps, min_8bit, max_8bit, False),
            grid=(rows // 8,),
            warmup=False,
            num_warps=2,
        )
        return output_q, output_s
    if (
        rows
        and h % group_size == 0
        and input.is_contiguous()
        and output_q.is_contiguous()
        and output_s.is_contiguous()
    ) and (xp | qp | sp) & 15 == 0:
        if br == 64:
            _dense.run(
                xp,
                qp,
                sp,
                (rows, group_size << 8 | br, eps, min_8bit, max_8bit, _ROCM),
                grid=(triton.cdiv(rows, br),),
                warmup=False,
                num_warps=8,
            )
        else:
            _dense.run(
                xp,
                qp,
                sp,
                (rows, group_size << 8 | br, eps, min_8bit, max_8bit, _ROCM),
                grid=(triton.cdiv(rows, br),),
                warmup=False,
            )
    elif rows:
        if br == 64:
            _quant.run(
                xp,
                qp,
                sp,
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
                _ROCM,
                grid=(triton.cdiv(rows, br),),
                warmup=False,
                num_warps=8,
            )
        else:
            _quant.run(
                xp,
                qp,
                sp,
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
                _ROCM,
                grid=(triton.cdiv(rows, br),),
                warmup=False,
            )
    return (output_q, output_s)


__all__ = ["per_token_group_quant_8bit_v2"]
