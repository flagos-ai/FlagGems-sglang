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
def _encode(v):
    bits = v.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(bits >> 23 & 255, 121)
    magic = (exponent + 20 << 23).to(tl.float32, bitcast=True)
    rounded = (tl.abs(v) + magic).to(tl.uint32, bitcast=True)
    code = rounded + (exponent << 3) - 968
    code = tl.where(v != v, 127, code)
    return (code | bits >> 24 & 128).to(tl.uint8)


@tr.jit
def _quant(x, s, factor, inv, inv_lo):
    numerator = x * factor
    return _encode(tl.fma(numerator, inv, numerator * inv_lo))


@tr.jit
def _pack(X, R, output_ptr, M: tl.constexpr, BR: tl.constexpr):
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
    output_ptr = tl.multiple_of(output_ptr, 16)
    N: tl.constexpr = M[0]
    D: tl.constexpr = 512
    DR: tl.constexpr = 64
    G: tl.constexpr = M[1]
    XS: tl.constexpr = M[2]
    XC: tl.constexpr = M[3]
    RS: tl.constexpr = M[4]
    RC: tl.constexpr = M[5]
    NG: tl.constexpr = D // G
    END: tl.constexpr = D + NG * 4
    WIDTH: tl.constexpr = END + DR * 2
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    cc = tl.arange(0, D)
    x = tl.load(
        X + rr[:, None] * XS + cc[None, :] * XC, rr[:, None] < N, 0
    ).to(tl.float32)
    tiles = tl.reshape(x, (BR, NG, G))
    bits = tl.max(tiles.to(tl.int32, bitcast=True) & 2147483647, 2)
    scale = tl.div_rn(bits.to(tl.float32, bitcast=True), 448.0)
    factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
    denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
    if N >= 1024:
        root = tl.rsqrt(denominator)
        estimate = root * root
        estimate = estimate * tl.fma(-estimate, denominator, 2.0)
        inv = tl.where(scale == float("inf"), 0.0, estimate)
    else:
        inv = tl.where(scale == float("inf"), 0.0, 1.0 / denominator)
    inv_lo = inv * tl.fma(-inv, denominator, 1.0)
    quant = tl.reshape(
        _quant(
            tiles,
            denominator[:, :, None],
            factor[:, :, None],
            inv[:, :, None],
            inv_lo[:, :, None],
        ),
        (BR, D),
    )
    rc = tl.arange(0, DR)
    rope = tl.load(
        R.to(tl.pointer_type(tl.uint16)) + rr[:, None] * RS + rc[None, :] * RC,
        rr[:, None] < N,
        0,
    )
    tl.store(
        output_ptr + rr[:, None] * WIDTH + cc[None, :], quant, rr[:, None] < N
    )
    sg = tl.arange(0, NG)
    tl.store(
        (output_ptr + D).to(tl.pointer_type(tl.float32))
        + rr[:, None] * (WIDTH // 4)
        + sg[None, :],
        scale,
        rr[:, None] < N,
    )
    tl.store(
        (output_ptr + END).to(tl.pointer_type(tl.uint16))
        + rr[:, None] * (WIDTH // 2)
        + rc[None, :],
        rope,
        rr[:, None] < N,
    )


@tr.jit
def _pack_pair(X, R, output_ptr, M: tl.constexpr):
    output_ptr = tl.multiple_of(output_ptr.to(tl.pointer_type(tl.uint8)), 16)
    G: tl.constexpr = M[1]
    XS: tl.constexpr = M[2]
    RS: tl.constexpr = M[4]
    RC: tl.constexpr = M[5]
    NG: tl.constexpr = 512 // G
    END: tl.constexpr = 512 + NG * 4
    WIDTH: tl.constexpr = END + 128
    row = tl.program_id(0)
    cc = tl.arange(0, 256)
    raw = tl.load(X.to(tl.pointer_type(tl.uint32)) + row * (XS // 2) + cc)
    tiles = tl.reshape(raw, (NG, G // 2))
    low = tiles & 32767
    high = tiles >> 16 & 32767
    magnitude = tl.max(tl.maximum(low, high), 1) << 16
    peak = magnitude.to(tl.float32, bitcast=True)
    scale = tl.div_rn(magnitude.to(tl.float32, bitcast=True), 448.0)
    factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
    denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
    root = tl.rsqrt(peak * (factor * 0.001953125))
    inv = tl.where(scale == float("inf"), 0.0, root * root * 0.875)
    inv_lo = inv * tl.fma(-inv, denominator, 1.0)
    a = (tiles << 16).to(tl.float32, bitcast=True)
    b = (tiles & 4294901760).to(tl.float32, bitcast=True)
    qa = _quant_pair(
        a, denominator[:, None], factor[:, None], inv[:, None], inv_lo[:, None]
    ).to(tl.uint16)
    qb = _quant_pair(
        b, denominator[:, None], factor[:, None], inv[:, None], inv_lo[:, None]
    ).to(tl.uint16)
    packed = tl.reshape(qa | qb << 8, (256,))
    tl.store(
        output_ptr.to(tl.pointer_type(tl.uint16)) + row * (WIDTH // 2) + cc,
        packed,
    )
    sg = tl.arange(0, NG)
    tl.store(
        (output_ptr + 512).to(tl.pointer_type(tl.float32))
        + row * (WIDTH // 4)
        + sg,
        scale,
    )
    rc = tl.arange(0, 64)
    rope = tl.load(R.to(tl.pointer_type(tl.uint16)) + row * RS + rc * RC)
    tl.store(
        (output_ptr + END).to(tl.pointer_type(tl.uint16))
        + row * (WIDTH // 2)
        + rc,
        rope,
    )


@tr.jit
def _encode_pair(v):
    bits = v.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(bits >> 23 & 255, 121)
    bias = exponent * 8388616 + 167771192
    code = (tl.abs(v) + bias.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    code = tl.where(v != v, 127, code)
    return (code | bits >> 24 & 128).to(tl.uint8)


@tr.jit
def _quant_pair(x, s, factor, inv, inv_lo):
    numerator = x * factor
    return _encode_pair(tl.fma(numerator, inv, numerator * inv_lo))


@tr.jit
def _shortscale323(value):
    product = (value * 0.0022321429569274187).to(tl.uint32, bitcast=True)
    return (product - (5386 >> (product & 15) & 1)).to(
        tl.float32, bitcast=True
    )


@tr.jit
def _pack_scale370(X, R, output_ptr, M: tl.constexpr, BR: tl.constexpr):
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
    output_ptr = tl.multiple_of(output_ptr, 16)
    N: tl.constexpr = M[0]
    D: tl.constexpr = 512
    DR: tl.constexpr = 64
    G: tl.constexpr = M[1]
    XS: tl.constexpr = M[2]
    XC: tl.constexpr = M[3]
    RS: tl.constexpr = M[4]
    RC: tl.constexpr = M[5]
    NG: tl.constexpr = D // G
    END: tl.constexpr = D + NG * 4
    WIDTH: tl.constexpr = END + DR * 2
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    cc = tl.arange(0, D)
    x = tl.load(
        X + rr[:, None] * XS + cc[None, :] * XC, rr[:, None] < N, 0
    ).to(tl.float32)
    tiles = tl.reshape(x, (BR, NG, G))
    bits = tl.max(tiles.to(tl.int32, bitcast=True) & 2147483647, 2)
    scale = (
        _shortscale323(bits.to(tl.float32, bitcast=True))
        if M[0] < 512
        else tl.div_rn(bits.to(tl.float32, bitcast=True), 448.0)
    )
    factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
    denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
    if N >= 1024:
        root = tl.rsqrt(denominator)
        estimate = root * root
        estimate = estimate * tl.fma(-estimate, denominator, 2.0)
        inv = tl.where(scale == float("inf"), 0.0, estimate)
    else:
        inv = tl.where(scale == float("inf"), 0.0, 1.0 / denominator)
    inv_lo = inv * tl.fma(-inv, denominator, 1.0)
    quant = tl.reshape(
        _quant(
            tiles,
            denominator[:, :, None],
            factor[:, :, None],
            inv[:, :, None],
            inv_lo[:, :, None],
        ),
        (BR, D),
    )
    rc = tl.arange(0, DR)
    rope = tl.load(
        R.to(tl.pointer_type(tl.uint16)) + rr[:, None] * RS + rc[None, :] * RC,
        rr[:, None] < N,
        0,
    )
    tl.store(
        output_ptr + rr[:, None] * WIDTH + cc[None, :], quant, rr[:, None] < N
    )
    sg = tl.arange(0, NG)
    tl.store(
        (output_ptr + D).to(tl.pointer_type(tl.float32))
        + rr[:, None] * (WIDTH // 4)
        + sg[None, :],
        scale,
        rr[:, None] < N,
    )
    tl.store(
        (output_ptr + END).to(tl.pointer_type(tl.uint16))
        + rr[:, None] * (WIDTH // 2)
        + rc[None, :],
        rope,
        rr[:, None] < N,
    )


def _fallback414(k_nope, k_rope, group_size=128):
    n = k_nope.shape[0]
    width = 640 + 2048 // group_size
    if n >= 1024:
        output = k_nope.new_empty((n, width), dtype=torch.uint8)
    else:
        output = k_nope.new_empty((n, width), dtype=torch.float8_e4m3fn)
    if n:
        br = (
            1 if n < 512 or (n < 1024 and torch.version.hip is not None) else 4
        )
        meta = (n, group_size, *k_nope.stride(), *k_rope.stride())
        if n >= 1024:
            if (
                torch.version.hip is not None
                and meta[3] == 1
                and (meta[2] % 2 == 0)
                and (k_nope.data_ptr() % 4 == 0)
                and (group_size % 2 == 0)
            ):
                _pack_pair.run(
                    k_nope,
                    k_rope,
                    output,
                    meta,
                    grid=(n,),
                    warmup=False,
                    num_warps=1,
                )
            else:
                _pack.run(
                    k_nope,
                    k_rope,
                    output,
                    meta,
                    1,
                    grid=(n,),
                    warmup=False,
                    num_warps=1,
                )
        elif torch.version.hip is not None:
            _pack.run(
                k_nope,
                k_rope,
                output.data_ptr(),
                meta,
                br,
                grid=(tr.cdiv(n, br),),
                warmup=False,
                num_warps=1,
            )
        else:
            _pack_scale370.run(
                k_nope,
                k_rope,
                output.data_ptr(),
                meta,
                br,
                grid=(tr.cdiv(n, br),),
                warmup=False,
            )
    return output.view(torch.float8_e4m3fn) if n >= 1024 else output


@tr.jit
def _scalar_direct414(X, R, output_ptr):
    BR: tl.constexpr = 1
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
    output_ptr = tl.multiple_of(output_ptr, 16)
    D: tl.constexpr = 512
    DR: tl.constexpr = 64
    G: tl.constexpr = 128
    XS: tl.constexpr = 512
    XC: tl.constexpr = 1
    RS: tl.constexpr = 64
    RC: tl.constexpr = 1
    NG: tl.constexpr = D // G
    END: tl.constexpr = D + NG * 4
    WIDTH: tl.constexpr = END + DR * 2
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    cc = tl.arange(0, D)
    x = tl.load(X + rr[:, None] * XS + cc[None, :] * XC, True, 0).to(
        tl.float32
    )
    tiles = tl.reshape(x, (BR, NG, G))
    bits = tl.max(tiles.to(tl.int32, bitcast=True) & 2147483647, 2)
    scale = tl.div_rn(bits.to(tl.float32, bitcast=True), 448.0)
    factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
    denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
    root = tl.rsqrt(denominator)
    estimate = root * root
    estimate = estimate * tl.fma(-estimate, denominator, 2.0)
    inv = tl.where(scale == float("inf"), 0.0, estimate)
    inv_lo = inv * tl.fma(-inv, denominator, 1.0)
    quant = tl.reshape(
        _quant(
            tiles,
            denominator[:, :, None],
            factor[:, :, None],
            inv[:, :, None],
            inv_lo[:, :, None],
        ),
        (BR, D),
    )
    rc = tl.arange(0, DR)
    rope = tl.load(
        R.to(tl.pointer_type(tl.uint16)) + rr[:, None] * RS + rc[None, :] * RC,
        True,
        0,
    )
    tl.store(output_ptr + rr[:, None] * WIDTH + cc[None, :], quant, True)
    sg = tl.arange(0, NG)
    tl.store(
        (output_ptr + D).to(tl.pointer_type(tl.float32))
        + rr[:, None] * (WIDTH // 4)
        + sg[None, :],
        scale,
        True,
    )
    tl.store(
        (output_ptr + END).to(tl.pointer_type(tl.uint16))
        + rr[:, None] * (WIDTH // 2)
        + rc[None, :],
        rope,
        True,
    )


@tr.jit
def _pair_direct414(X, R, output_ptr):
    output_ptr = tl.multiple_of(output_ptr.to(tl.pointer_type(tl.uint8)), 16)
    G: tl.constexpr = 128
    XS: tl.constexpr = 512
    RS: tl.constexpr = 64
    RC: tl.constexpr = 1
    NG: tl.constexpr = 512 // G
    END: tl.constexpr = 512 + NG * 4
    WIDTH: tl.constexpr = END + 128
    row = tl.program_id(0)
    cc = tl.arange(0, 256)
    raw = tl.load(X.to(tl.pointer_type(tl.uint32)) + row * (XS // 2) + cc)
    tiles = tl.reshape(raw, (NG, G // 2))
    low = tiles & 32767
    high = tiles >> 16 & 32767
    magnitude = tl.max(tl.maximum(low, high), 1) << 16
    peak = magnitude.to(tl.float32, bitcast=True)
    scale = tl.div_rn(magnitude.to(tl.float32, bitcast=True), 448.0)
    factor = tl.where(scale < 7.888609052210118e-31, 4294967296.0, 1.0)
    denominator = tl.where(scale == float("inf"), 1.0, scale * factor)
    root = tl.rsqrt(peak * (factor * 0.001953125))
    inv = tl.where(scale == float("inf"), 0.0, root * root * 0.875)
    inv_lo = inv * tl.fma(-inv, denominator, 1.0)
    a = (tiles << 16).to(tl.float32, bitcast=True)
    b = (tiles & 4294901760).to(tl.float32, bitcast=True)
    qa = _quant_pair(
        a, denominator[:, None], factor[:, None], inv[:, None], inv_lo[:, None]
    ).to(tl.uint16)
    qb = _quant_pair(
        b, denominator[:, None], factor[:, None], inv[:, None], inv_lo[:, None]
    ).to(tl.uint16)
    packed = tl.reshape(qa | qb << 8, (256,))
    tl.store(
        output_ptr.to(tl.pointer_type(tl.uint16)) + row * (WIDTH // 2) + cc,
        packed,
    )
    sg = tl.arange(0, NG)
    tl.store(
        (output_ptr + 512).to(tl.pointer_type(tl.float32))
        + row * (WIDTH // 4)
        + sg,
        scale,
    )
    rc = tl.arange(0, 64)
    rope = tl.load(R.to(tl.pointer_type(tl.uint16)) + row * RS + rc * RC)
    tl.store(
        (output_ptr + END).to(tl.pointer_type(tl.uint16))
        + row * (WIDTH // 2)
        + rc,
        rope,
    )


def quantize_k_cache_fast(k_nope, k_rope, group_size=128):
    n = k_nope.shape[0]
    if (
        n < 1024
        or group_size != 128
        or (not k_nope.is_contiguous())
        or (not k_rope.is_contiguous())
        or k_nope.data_ptr() % 4
    ):
        return _fallback414(k_nope, k_rope, group_size)
    output = k_nope.new_empty((n, 656), dtype=torch.float8_e4m3fn)
    if torch.version.hip is not None:
        _pair_direct414.run(
            k_nope,
            k_rope,
            output.data_ptr(),
            grid=(n,),
            warmup=False,
            num_warps=1,
        )
    else:
        _scalar_direct414.run(
            k_nope,
            k_rope,
            output.data_ptr(),
            grid=(n,),
            warmup=False,
            num_warps=1,
        )
    return output


__all__ = ["quantize_k_cache_fast"]
