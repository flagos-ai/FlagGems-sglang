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
import triton as tr
import triton.language as tl


@tr.jit
def _scale_exact(bits):
    exponent = bits >> 23
    fraction = bits >> 16 & 127
    mantissa = fraction | 128
    high = mantissa >= 224
    shift = tl.where(high, 18, 19)
    rounded = ((mantissa << shift) + 3) // 7
    normal = (exponent - tl.where(high, 8, 9) << 23) + rounded - 8388608
    tiny_shift = tl.minimum(tl.maximum(exponent, 1) + 9, 18)
    tiny_mantissa = tl.where(exponent == 0, fraction, mantissa)
    tiny = ((tiny_mantissa << tiny_shift) + 3) // 7
    result = tl.where(exponent < 10, tiny, normal)
    special = bits | tl.where(fraction != 0, 4194304, 0)
    result = tl.where(exponent == 255, special, result)
    return result.to(tl.int32).to(tl.float32, bitcast=True)


@tr.jit
def _scale(bits, N: tl.constexpr):
    value = bits.to(tl.float32, bitcast=True)
    if N < 4096:
        return value / 448.0
    elif N < 16384:
        q = value * 0.0022321429569274187
        residual = tl.fma(q, -448.0, value)
        scale = tl.fma(residual, 0.0022321429569274187, q)
        special = (bits | tl.where(bits & 8388607 != 0, 4194304, 0)).to(
            tl.float32, bitcast=True
        )
        return tl.where(bits >> 23 == 255, special, scale)
    else:
        return _scale_exact(bits)


@tr.jit
def _encode(v):
    bits = v.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(bits >> 23 & 255, 121)
    magic = (exponent + 20 << 23).to(tl.float32, bitcast=True)
    rounded = (tl.abs(v) + magic).to(tl.uint32, bitcast=True)
    code = rounded + (exponent << 3) - 968
    code = tl.where(v != v, 127, code)
    return (code | bits >> 24 & 128).to(tl.uint8)


@tr.jit
def _pack(X, R, output_ptr, M: tl.constexpr, BR: tl.constexpr):
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
    P: tl.constexpr = 1
    N: tl.constexpr = M[0]
    D: tl.constexpr = M[1]
    DR: tl.constexpr = M[2]
    G: tl.constexpr = M[3]
    XS: tl.constexpr = M[4]
    XC: tl.constexpr = M[5]
    RS: tl.constexpr = M[6]
    RC: tl.constexpr = M[7]
    NG: tl.constexpr = D // G
    END: tl.constexpr = D + NG * 4
    WIDTH: tl.constexpr = END + DR * 2
    BG: tl.constexpr = tr.next_power_of_2(BR * NG)
    BC: tl.constexpr = tr.next_power_of_2(G // P)
    groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
    rows = groups // NG
    tiles = groups % NG
    c = tl.arange(0, BC)
    mask = (
        (groups[:, None] < N * NG)
        & (c[None, :] < G // P)
        & (tl.arange(0, BG)[:, None] < BR * NG)
    )
    if (
        N % BR == 0
        and BG == BR * NG
        and (BC == G)
        and (DR % NG == 0)
        and (tr.next_power_of_2(DR // NG) == DR // NG)
    ):
        rc = tl.arange(0, tr.next_power_of_2(DR // NG))
        x = tl.load(
            X + rows[:, None] * XS + (tiles[:, None] * G + c[None, :]) * XC
        ).to(tl.float32)
        mag = x.to(tl.int32, bitcast=True) & 2147483647
        scale = _scale(tl.max(mag, 1).to(tl.uint32), N)
        factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
        denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
        if N < 4096:
            inv = tl.where(scale == float("inf"), 0.0, 1.0 / denominator)
        else:
            inv = tl.where(
                scale == float("inf"),
                0.0,
                tl.inline_asm_elementwise(
                    "v_rcp_f32 $0, $1",
                    constraints="=v,v",
                    args=[denominator],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                ),
            )
        if N >= 768 and N < 4096:
            rope = tl.load(
                R.to(tl.pointer_type(tl.int16))
                + rows[:, None] * RS
                + (tiles[:, None] * (DR // NG) + rc[None, :]) * RC
            )
            tl.store(
                (output_ptr + END).to(tl.pointer_type(tl.int16))
                + rows[:, None] * (WIDTH // 2)
                + tiles[:, None] * (DR // NG)
                + rc[None, :],
                rope,
            )
        tl.store(
            output_ptr
            + rows[:, None] * WIDTH
            + tiles[:, None] * G
            + c[None, :],
            _quant(x, denominator[:, None], factor[:, None], inv[:, None]),
        )
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rows * (WIDTH // 4)
            + tiles,
            scale,
        )
        if N < 768 or N >= 4096:
            rope = tl.load(
                R.to(tl.pointer_type(tl.int16))
                + rows[:, None] * RS
                + (tiles[:, None] * (DR // NG) + rc[None, :]) * RC
            )
            tl.store(
                (output_ptr + END).to(tl.pointer_type(tl.int16))
                + rows[:, None] * (WIDTH // 2)
                + tiles[:, None] * (DR // NG)
                + rc[None, :],
                rope,
            )
    else:
        x = tl.load(
            X + rows[:, None] * XS + (tiles[:, None] * G + c[None, :]) * XC,
            mask,
            0,
        ).to(tl.float32)
        mag = x.to(tl.int32, bitcast=True) & 2147483647
        scale = _scale(tl.max(mag, 1).to(tl.uint32), N)
        factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
        denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
        if N < 4096:
            inv = tl.where(scale == float("inf"), 0.0, 1.0 / denominator)
        else:
            inv = tl.where(
                scale == float("inf"),
                0.0,
                tl.inline_asm_elementwise(
                    "v_rcp_f32 $0, $1",
                    constraints="=v,v",
                    args=[denominator],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=1,
                ),
            )
        if N >= 768 and N < 4096:
            rr = tl.program_id(0) * BR + tl.arange(0, BR)
            rc = tl.arange(0, tr.next_power_of_2(DR))
            rmask = (rr[:, None] < N) & (rc[None, :] < DR)
            rope = tl.load(
                R.to(tl.pointer_type(tl.int16))
                + rr[:, None] * RS
                + rc[None, :] * RC,
                rmask,
                0,
            )
            tl.store(
                (output_ptr + END).to(tl.pointer_type(tl.int16))
                + rr[:, None] * (WIDTH // 2)
                + rc[None, :],
                rope,
                rmask,
            )
        tl.store(
            output_ptr
            + rows[:, None] * WIDTH
            + tiles[:, None] * G
            + c[None, :],
            _quant(x, denominator[:, None], factor[:, None], inv[:, None]),
            mask,
        )
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rows * (WIDTH // 4)
            + tiles,
            scale,
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
        if N < 768 or N >= 4096:
            rr = tl.program_id(0) * BR + tl.arange(0, BR)
            rc = tl.arange(0, tr.next_power_of_2(DR))
            rmask = (rr[:, None] < N) & (rc[None, :] < DR)
            rope = tl.load(
                R.to(tl.pointer_type(tl.int16))
                + rr[:, None] * RS
                + rc[None, :] * RC,
                rmask,
                0,
            )
            tl.store(
                (output_ptr + END).to(tl.pointer_type(tl.int16))
                + rr[:, None] * (WIDTH // 2)
                + rc[None, :],
                rope,
                rmask,
            )


@tr.jit
def _quant(x, s, factor, inv):
    numerator = x * factor
    q = numerator * inv
    residual = tl.fma(q, -s, numerator)
    return _encode(tl.fma(residual, inv, q))


def quantize_k_cache_fast(k_nope, k_rope, group_size=128):
    n, d = k_nope.shape
    dr = k_rope.shape[1]
    width = d + d // group_size * 4 + dr * 2
    output = k_nope.new_empty((n, width), dtype=torch.float8_e4m3fn)
    if n:
        br = 1 if n <= 16 else 4
        meta = (n, d, dr, group_size, *k_nope.stride(), *k_rope.stride())
        _pack.run(
            k_nope,
            k_rope,
            output,
            meta,
            br,
            grid=(tr.cdiv(n, br),),
            warmup=False,
            num_warps=4 if n < 768 else 2,
        )
    return output


__all__ = ["quantize_k_cache_fast"]
