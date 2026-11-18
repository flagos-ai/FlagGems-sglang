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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl


def _prepare_metadata(x, x_scale, masked_m, e, t, k):
    xs = x.stride()
    ss = x_scale.stride()
    ms = masked_m.stride()
    limit = 2147483647
    if (
        0 < e * t * k <= limit
        and xs == (t * k, k, 1)
        and (ss == (t * (k // 128), k // 128, 1))
        and (ms == (1,))
    ):
        return (xs, ss, ms, False)
    wide = x.numel() > limit
    if not wide:
        for tensor, strides in ((x, xs), (x_scale, ss), (masked_m, ms)):
            if max(strides) > limit:
                wide = True
                break
            if tensor.is_contiguous():
                continue
            if (
                sum(
                    (
                        (size - 1) * stride
                        for size, stride in zip(tensor.shape, strides)
                    )
                )
                > limit
            ):
                wide = True
                break
    return (xs, ss, ms, wide)


@triton.jit
def _wide_index(index, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64)
    else:
        return index


@triton.jit
def _decode(raw):
    v = raw.to(tl.uint32)
    mag = v & 127
    normal = ((mag << 20) + (120 << 23)).to(tl.float32, bitcast=True)
    value = tl.where(mag < 8, tl.fma(normal, 2.0, -0.015625), normal)
    value = tl.where(mag == 127, float("nan"), value)
    return value


@triton.jit
def _decode_selected(raw, E4B15: tl.constexpr):
    if E4B15:
        code = raw.to(tl.uint8)
        carrier = code.to(tl.float8e4b15, bitcast=True).to(tl.float16)
        value = carrier * tl.full((), 256.0, tl.float16)
        repaired = tl.inline_asm_elementwise(
            "{ .reg .b32 value, magnitude, match, constant; mov.b32 "
            "value, $1; abs.f16x2 magnitude, value; mov.b32 constant, "
            "0x5f805f80; set.eq.u32.f16x2 match, magnitude, constant; "
            "and.b32 match, match, 0x21802180; xor.b32 $0, value, match; "
            "}",
            constraints="=r,r",
            args=[value],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )
        return repaired.to(tl.float32)
    return _decode(raw)


@triton.jit
def _encode_original(v, DEFERRED: tl.constexpr = False):
    raw = v.to(tl.uint32, bitcast=True)
    mag = raw & 2147483647
    normal = (mag + 524287 + (mag >> 20 & 1) >> 20) - (0 if DEFERRED else 960)
    absolute = tl.abs(v)
    sub = (absolute + (16384.125 if DEFERRED else 16384.0)).to(
        tl.uint32, bitcast=True
    )
    code = tl.where(absolute < 0.015625, sub, normal)
    code = tl.where(mag > 1139277824, 63 if DEFERRED else 127, code)
    return code.to(tl.uint8)


@triton.jit
def _encode_dynamic(v):
    raw = v.to(tl.uint32, bitcast=True)
    absolute = tl.minimum(tl.abs(v), 480.0, propagate_nan=tl.PropagateNan.NONE)
    mag = absolute.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(mag >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192 | raw >> 24 & 128
    rounded = (absolute + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    return rounded.to(tl.uint8)


@triton.jit
def _encode(v, E4B15: tl.constexpr, DEFERRED: tl.constexpr = False):
    if E4B15:
        return _encode_dynamic(v)
    return _encode_original(v, DEFERRED)


@triton.jit
def _kernel(
    X, S, M, OS, output_ptr, META: tl.constexpr, WIDE: tl.constexpr = False
):
    T: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    X0: tl.constexpr = META[2]
    X1: tl.constexpr = META[3]
    X2: tl.constexpr = META[4]
    S0: tl.constexpr = META[5]
    S1: tl.constexpr = META[6]
    S2: tl.constexpr = META[7]
    MS: tl.constexpr = META[8]
    B: tl.constexpr = META[9]
    E4B15: tl.constexpr = META[10]
    tile = _wide_index(tl.program_id(0), WIDE)
    tiles: tl.constexpr = triton.cdiv(K, B)
    row = tile // tiles
    e = row // T
    token = row % T
    count = tl.load(M + e * MS)
    if token < count:
        k = tile % tiles * B + tl.arange(0, B)
        raw = tl.load(
            X + e * X0 + token * X1 + k * X2,
            k < K,
            other=0,
            cache_modifier=".cg",
        )
        scale = tl.load(
            S + e * S0 + token * S1 + k // 128 * S2, k < K, other=0
        )
        inv = tl.div_rn(1.0, tl.load(OS).to(tl.float32))
        value = _decode_selected(raw, E4B15) * scale
        value = value * inv
        encoded = _encode(value, E4B15)
        if not E4B15:
            sign = (
                raw.to(tl.uint32) << 24
                ^ scale.to(tl.uint32, bitcast=True)
                ^ inv.to(tl.uint32, bitcast=True)
            ) >> 24 & 128
            encoded = (encoded.to(tl.uint32) | sign).to(tl.uint8)
        tl.store(
            output_ptr + row * K + k, encoded, k < K, cache_modifier=".cs"
        )


@triton.jit
def _packed(
    X, S, M, OS, output_ptr, META: tl.constexpr, WIDE: tl.constexpr = False
):
    T: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    X0: tl.constexpr = META[2]
    X1: tl.constexpr = META[3]
    S0: tl.constexpr = META[4]
    S1: tl.constexpr = META[5]
    S2: tl.constexpr = META[6]
    MS: tl.constexpr = META[7]
    B: tl.constexpr = META[8]
    E4B15: tl.constexpr = META[9]
    tile = _wide_index(tl.program_id(0), WIDE)
    tiles: tl.constexpr = triton.cdiv(K // 4, B)
    row = tile // tiles
    e = row // T
    token = row % T
    count = tl.load(M + e * MS)
    if token < count:
        w = tile % tiles * B + tl.arange(0, B)
        raw = tl.load(
            X + e * X0 + token * X1 + w,
            w < K // 4,
            other=0,
            cache_modifier=".cg",
        ).to(tl.uint32)
        scale = tl.load(
            S + e * S0 + token * S1 + w // 32 * S2, w < K // 4, other=0
        )
        inv = tl.div_rn(1.0, tl.load(OS).to(tl.float32))
        a = _decode_selected(raw, E4B15) * scale
        b = _decode_selected(raw >> 8, E4B15) * scale
        c = _decode_selected(raw >> 16, E4B15) * scale
        d = _decode_selected(raw >> 24, E4B15) * scale
        a = a * inv
        b = b * inv
        c = c * inv
        d = d * inv
        code = _encode(a, E4B15, True).to(tl.uint32)
        code = code | _encode(b, E4B15, True).to(tl.uint32) << 8
        code = code | _encode(c, E4B15, True).to(tl.uint32) << 16
        code = code | _encode(d, E4B15, True).to(tl.uint32) << 24
        if not E4B15:
            code = (code & 2139062143) + 1077952576 & 2139062143
            factor_sign = (
                (
                    scale.to(tl.int32, bitcast=True)
                    ^ inv.to(tl.int32, bitcast=True)
                )
                >> 31
            ).to(tl.uint32)
            code = code | (raw ^ factor_sign) & 2155905152
        tl.store(
            output_ptr + row * (K // 4) + w,
            code,
            w < K // 4,
            cache_modifier=".cs",
        )


@triton.jit(do_not_specialize=["X", "S", "OS", "O"])
def _address_kernel(
    X, S, M, OS, output_ptr, META: tl.constexpr, WIDE: tl.constexpr = False
):
    XA: tl.constexpr = META[-2]
    OA: tl.constexpr = META[-1]
    x = X.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    out = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    scale = S.to(tl.uint64).to(tl.pointer_type(tl.float32))
    output_scale = OS.to(tl.uint64).to(tl.pointer_type(tl.float32))
    x = tl.multiple_of(x, XA)
    out = tl.multiple_of(out, OA)
    _kernel(x, scale, M, output_scale, out, META, WIDE)


@triton.jit(do_not_specialize=["X", "S", "OS", "O"])
def _address_packed(
    X, S, M, OS, output_ptr, META: tl.constexpr, WIDE: tl.constexpr = False
):
    XA: tl.constexpr = META[-2]
    OA: tl.constexpr = META[-1]
    x = X.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    out = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    scale = S.to(tl.uint64).to(tl.pointer_type(tl.float32))
    output_scale = OS.to(tl.uint64).to(tl.pointer_type(tl.float32))
    x = tl.multiple_of(x, XA)
    out = tl.multiple_of(out, OA)
    _packed(
        x.to(tl.pointer_type(tl.int32)),
        scale,
        M,
        output_scale,
        out.to(tl.pointer_type(tl.int32)),
        META,
        WIDE,
    )


def fp8_per_token_to_per_tensor_quant(x, x_scale, masked_m, output_scale):
    e, t, k = x.shape
    xs, ss, ms, wide = _prepare_metadata(x, x_scale, masked_m, e, t, k)
    if torch.version.hip is not None and (
        k % 4 == 0
        and xs[2] == 1
        and (xs[0] % 4 == 0)
        and (xs[1] % 4 == 0)
        and (x.storage_offset() % 4 == 0)
    ):
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if e * t * k:
            block = 512 if torch.version.hip is not None else 1024
            input_address = x.data_ptr()
            output_address = out.data_ptr()
            scale_address = x_scale.data_ptr()
            output_scale_address = output_scale.data_ptr()
            _address_packed.run(
                input_address,
                scale_address,
                masked_m,
                output_scale_address,
                output_address,
                (
                    t,
                    k,
                    xs[0] // 4,
                    xs[1] // 4,
                    *ss,
                    ms[0],
                    block,
                    torch.version.hip is None,
                    min(16, input_address & -input_address),
                    min(16, output_address & -output_address),
                ),
                num_warps=4,
                enable_fp_fusion=False,
                WIDE=wide,
                grid=(e * t * triton.cdiv(k // 4, block),),
                warmup=False,
            )
    else:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if e * t * k:
            block = min(4096, 1 << (k - 1).bit_length())
            input_address = x.data_ptr()
            output_address = out.data_ptr()
            scale_address = x_scale.data_ptr()
            output_scale_address = output_scale.data_ptr()
            _address_kernel.run(
                input_address,
                scale_address,
                masked_m,
                output_scale_address,
                output_address,
                (
                    t,
                    k,
                    *xs,
                    *ss,
                    ms[0],
                    block,
                    torch.version.hip is None,
                    min(16, input_address & -input_address),
                    min(16, output_address & -output_address),
                ),
                num_warps=4,
                enable_fp_fusion=False,
                WIDE=wide,
                grid=(e * t * ((k + block - 1) // block),),
                warmup=False,
            )
    return out


__all__ = ["fp8_per_token_to_per_tensor_quant"]
