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


def _prepare_metadata_S(x, x_scale, masked_m, e, t, k):
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
def _wide_index_S(index, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64)
    else:
        return index


@triton.jit
def _decode_S(raw):
    v = raw.to(tl.uint32)
    mag = v & 127
    normal = ((mag << 20) + (120 << 23)).to(tl.float32, bitcast=True)
    value = tl.where(mag < 8, tl.fma(normal, 2.0, -0.015625), normal)
    value = tl.where(mag == 127, float("nan"), value)
    return value


@triton.jit
def _encode_S(v):
    raw = v.to(tl.uint32, bitcast=True)
    mag = raw & 2147483647
    mag = tl.minimum(mag, 1139802112)
    exponent = tl.maximum(mag & 2139095040, 1015021568)
    magic_bits = exponent + 167772160
    magic = magic_bits.to(tl.float32, bitcast=True)
    absolute = mag.to(tl.float32, bitcast=True)
    rounded = (absolute + magic).to(tl.uint32, bitcast=True) - magic_bits
    code = rounded + (exponent >> 20) - 968
    return code.to(tl.uint8)


@triton.jit
def _kernel_S(
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
    tile = _wide_index_S(tl.program_id(0), WIDE)
    tiles: tl.constexpr = triton.cdiv(K, B)
    row = tile // tiles
    e = row // T
    token = row % T
    count = tl.load(M + e * MS)
    if token < count:
        k = tile % tiles * B + tl.arange(0, B)
        raw = tl.load(X + e * X0 + token * X1 + k * X2, k < K, other=0)
        scale = tl.load(
            S + e * S0 + token * S1 + k // 128 * S2, k < K, other=0
        )
        inv = tl.div_rn(1.0, tl.load(OS).to(tl.float32))
        value = _decode_S(raw) * scale
        value = value * inv
        encoded = _encode_S(value)
        sign = (
            raw.to(tl.uint32) << 24
            ^ scale.to(tl.uint32, bitcast=True)
            ^ inv.to(tl.uint32, bitcast=True)
        ) >> 24 & 128
        encoded = (encoded.to(tl.uint32) | sign).to(tl.uint8)
        tl.store(output_ptr + row * K + k, encoded, k < K)


@triton.jit
def _packed_S(
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
    tile = _wide_index_S(tl.program_id(0), WIDE)
    tiles: tl.constexpr = triton.cdiv(K // 4, B)
    row = tile // tiles
    e = row // T
    token = row % T
    count = tl.load(M + e * MS)
    if token < count:
        w = tile % tiles * B + tl.arange(0, B)
        raw = tl.load(X + e * X0 + token * X1 + w, w < K // 4, other=0).to(
            tl.uint32
        )
        scale = tl.load(
            S + e * S0 + token * S1 + w // 32 * S2, w < K // 4, other=0
        )
        inv = tl.div_rn(1.0, tl.load(OS).to(tl.float32))
        a = _decode_S(raw) * scale
        b = _decode_S(raw >> 8) * scale
        c = _decode_S(raw >> 16) * scale
        d = _decode_S(raw >> 24) * scale
        a = a * inv
        b = b * inv
        c = c * inv
        d = d * inv
        code = _encode_S(a).to(tl.uint32)
        code = code | _encode_S(b).to(tl.uint32) << 8
        code = code | _encode_S(c).to(tl.uint32) << 16
        code = code | _encode_S(d).to(tl.uint32) << 24
        factor_sign = (
            (scale.to(tl.int32, bitcast=True) ^ inv.to(tl.int32, bitcast=True))
            >> 31
        ).to(tl.uint32)
        code = code | (raw ^ factor_sign) & 2155905152
        tl.store(output_ptr + row * (K // 4) + w, code, w < K // 4)


@triton.jit(do_not_specialize=["X", "S", "OS", "O"])
def _address_kernel_S(
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
    _kernel_S(x, scale, M, output_scale, out, META, WIDE)


@triton.jit(do_not_specialize=["X", "S", "OS", "O"])
def _address_packed_S(
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
    _packed_S(
        x.to(tl.pointer_type(tl.int32)),
        scale,
        M,
        output_scale,
        out.to(tl.pointer_type(tl.int32)),
        META,
        WIDE,
    )


def fp8_per_token_to_per_tensor_quant_small(
    x, x_scale, masked_m, output_scale
):
    e, t, k = x.shape
    xs, ss, ms, wide = _prepare_metadata_S(x, x_scale, masked_m, e, t, k)
    if (
        k % 4 == 0
        and xs[2] == 1
        and (xs[0] % 4 == 0)
        and (xs[1] % 4 == 0)
        and (x.storage_offset() % 4 == 0)
    ):
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if e * t * k:
            input_address = x.data_ptr()
            output_address = out.data_ptr()
            scale_address = x_scale.data_ptr()
            output_scale_address = output_scale.data_ptr()
            _address_packed_S.run(
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
                    512,
                    min(16, input_address & -input_address),
                    min(16, output_address & -output_address),
                ),
                num_warps=4,
                enable_fp_fusion=False,
                WIDE=wide,
                grid=(e * t * ((k // 4 + 512 - 1) // 512),),
                warmup=False,
            )
    else:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if e * t * k:
            input_address = x.data_ptr()
            output_address = out.data_ptr()
            scale_address = x_scale.data_ptr()
            output_scale_address = output_scale.data_ptr()
            _address_kernel_S.run(
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
                    4096,
                    min(16, input_address & -input_address),
                    min(16, output_address & -output_address),
                ),
                num_warps=4,
                enable_fp_fusion=False,
                WIDE=wide,
                grid=(e * t * ((k + 4096 - 1) // 4096),),
                warmup=False,
            )
    return out


def _prepare_metadata_L(x, x_scale, masked_m, e, t, k):
    if (
        0 < e * t * k <= 2147483647
        and x.is_contiguous()
        and x_scale.is_contiguous()
        and masked_m.is_contiguous()
    ):
        return (None, False)
    xs = x.stride()
    ss = x_scale.stride()
    ms = masked_m.stride()
    limit = 2147483647
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
    return ((*xs, *ss, ms[0]), wide)


@triton.jit
def _wide_index_L(index, WIDE: tl.constexpr):
    if WIDE:
        return index.to(tl.int64)
    else:
        return index


@triton.jit
def _decode_L(raw):
    v = raw.to(tl.uint32)
    mag = v & 127
    normal = ((mag << 20) + (120 << 23)).to(tl.float32, bitcast=True)
    value = tl.minimum(
        tl.fma(normal, 2.0, -0.015625),
        normal,
        propagate_nan=tl.PropagateNan.NONE,
    )
    value = tl.where(mag == 127, float("nan"), value)
    return value


@triton.jit
def _encode_L(v, MAGNITUDE: tl.constexpr = False):
    raw = v.to(tl.uint32, bitcast=True)
    mag = raw if MAGNITUDE else raw & 2147483647
    mag = tl.minimum(mag, 1139802112)
    exponent = tl.maximum(mag >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192
    magic = magic_bits.to(tl.float32, bitcast=True)
    absolute = mag.to(tl.float32, bitcast=True)
    return (absolute + magic).to(tl.uint32, bitcast=True).to(tl.uint8)


@triton.jit
def _kernel_tile_L(
    X, S, M, OS, output_ptr, META: tl.constexpr, WIDE: tl.constexpr, tile
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
    tiles: tl.constexpr = triton.cdiv(K, B)
    row = tile // tiles
    e = row // T
    token = row % T
    count = tl.load(M + e * MS)
    if token < count:
        k = tile % tiles * B + tl.arange(0, B)
        raw = tl.load(X + e * X0 + token * X1 + k * X2, k < K, other=0)
        scale = tl.load(
            S + e * S0 + token * S1 + k // 128 * S2, k < K, other=0
        )
        inv = tl.div_rn(1.0, tl.load(OS).to(tl.float32))
        value = _decode_L(raw) * scale
        value = value * inv
        encoded = _encode_L(value)
        sign = (
            raw.to(tl.uint32) << 24
            ^ scale.to(tl.uint32, bitcast=True)
            ^ inv.to(tl.uint32, bitcast=True)
        ) >> 24 & 128
        encoded = (encoded.to(tl.uint32) | sign).to(tl.uint8)
        tl.store(output_ptr + row * K + k, encoded, k < K)


@triton.jit
def _kernel_L(
    X,
    S,
    M,
    OS,
    output_ptr,
    META: tl.constexpr,
    WIDE: tl.constexpr = False,
    PROGRAMS: tl.constexpr = 0,
):
    if PROGRAMS:
        tile = tl.program_id(0).to(tl.int64)
        limit = tl.full((), PROGRAMS, tl.int64)
        step = tl.num_programs(0).to(tl.int64)
        while tile < limit:
            _kernel_tile_L(X, S, M, OS, output_ptr, META, WIDE, tile)
            tile += tl.minimum(step, limit - tile)
    else:
        _kernel_tile_L(
            X,
            S,
            M,
            OS,
            output_ptr,
            META,
            WIDE,
            _wide_index_L(tl.program_id(0), WIDE),
        )


@triton.jit
def _packed_tile_L(
    X, S, M, OS, output_ptr, META: tl.constexpr, WIDE: tl.constexpr, tile
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
    tiles: tl.constexpr = triton.cdiv(K // 4, B)
    row = tile // tiles
    e = row // T
    token = row % T
    count = tl.load(M + e * MS)
    if token < count:
        w = tile % tiles * B + tl.arange(0, B)
        raw = tl.load(X + e * X0 + token * X1 + w, w < K // 4, other=0).to(
            tl.uint32
        )
        scale = tl.load(
            S + e * S0 + token * S1 + w // 32 * S2, w < K // 4, other=0
        )
        inv = tl.div_rn(1.0, tl.load(OS).to(tl.float32))
        magnitude_scale = tl.abs(scale)
        magnitude_inv = tl.abs(inv)
        a = _decode_L(raw) * magnitude_scale
        b = _decode_L(raw >> 8) * magnitude_scale
        c = _decode_L(raw >> 16) * magnitude_scale
        d = _decode_L(raw >> 24) * magnitude_scale
        a = a * magnitude_inv
        b = b * magnitude_inv
        c = c * magnitude_inv
        d = d * magnitude_inv
        code = _encode_L(a, True).to(tl.uint32)
        code = code | _encode_L(b, True).to(tl.uint32) << 8
        code = code | _encode_L(c, True).to(tl.uint32) << 16
        code = code | _encode_L(d, True).to(tl.uint32) << 24
        factor_sign = (
            (scale.to(tl.int32, bitcast=True) ^ inv.to(tl.int32, bitcast=True))
            >> 31
        ).to(tl.uint32)
        code = code | (raw ^ factor_sign) & 2155905152
        tl.store(output_ptr + row * (K // 4) + w, code, w < K // 4)


@triton.jit
def _packed_L(
    X,
    S,
    M,
    OS,
    output_ptr,
    META: tl.constexpr,
    WIDE: tl.constexpr = False,
    PROGRAMS: tl.constexpr = 0,
):
    if PROGRAMS:
        tile = tl.program_id(0).to(tl.int64)
        limit = tl.full((), PROGRAMS, tl.int64)
        step = tl.num_programs(0).to(tl.int64)
        while tile < limit:
            _packed_tile_L(X, S, M, OS, output_ptr, META, WIDE, tile)
            tile += tl.minimum(step, limit - tile)
    else:
        _packed_tile_L(
            X,
            S,
            M,
            OS,
            output_ptr,
            META,
            WIDE,
            _wide_index_L(tl.program_id(0), WIDE),
        )


@triton.jit(do_not_specialize=["X", "S", "OS", "O"])
def _address_kernel_L(
    X,
    S,
    M,
    OS,
    output_ptr,
    META: tl.constexpr,
    WIDE: tl.constexpr = False,
    PROGRAMS: tl.constexpr = 0,
):
    T: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    LAYOUT: tl.constexpr = META[2]
    if LAYOUT is None:
        STRIDES: tl.constexpr = (T * K, K, 1, T * (K // 128), K // 128, 1, 1)
    else:
        STRIDES: tl.constexpr = LAYOUT
    FULL_META: tl.constexpr = (
        T,
        K,
        STRIDES[0],
        STRIDES[1],
        STRIDES[2],
        STRIDES[3],
        STRIDES[4],
        STRIDES[5],
        STRIDES[6],
        META[3],
        META[4],
        META[5],
    )
    XA: tl.constexpr = META[-2]
    OA: tl.constexpr = META[-1]
    x = X.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    out = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    scale = S.to(tl.uint64).to(tl.pointer_type(tl.float32))
    output_scale = OS.to(tl.uint64).to(tl.pointer_type(tl.float32))
    x = tl.multiple_of(x, XA)
    out = tl.multiple_of(out, OA)
    _kernel_L(x, scale, M, output_scale, out, FULL_META, WIDE, PROGRAMS)


@triton.jit(do_not_specialize=["X", "S", "OS", "O"])
def _address_packed_L(
    X,
    S,
    M,
    OS,
    output_ptr,
    META: tl.constexpr,
    WIDE: tl.constexpr = False,
    PROGRAMS: tl.constexpr = 0,
):
    T: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    LAYOUT: tl.constexpr = META[2]
    if LAYOUT is None:
        STRIDES: tl.constexpr = (T * K, K, 1, T * (K // 128), K // 128, 1, 1)
    else:
        STRIDES: tl.constexpr = LAYOUT
    FULL_META: tl.constexpr = (
        T,
        K,
        STRIDES[0] // 4,
        STRIDES[1] // 4,
        STRIDES[3],
        STRIDES[4],
        STRIDES[5],
        STRIDES[6],
        META[3],
        META[4],
        META[5],
    )
    XA: tl.constexpr = META[-2]
    OA: tl.constexpr = META[-1]
    x = X.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    out = output_ptr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
    scale = S.to(tl.uint64).to(tl.pointer_type(tl.float32))
    output_scale = OS.to(tl.uint64).to(tl.pointer_type(tl.float32))
    x = tl.multiple_of(x, XA)
    out = tl.multiple_of(out, OA)
    _packed_L(
        x.to(tl.pointer_type(tl.int32)),
        scale,
        M,
        output_scale,
        out.to(tl.pointer_type(tl.int32)),
        FULL_META,
        WIDE,
        PROGRAMS,
    )


def fp8_per_token_to_per_tensor_quant_large(
    x, x_scale, masked_m, output_scale
):
    e, t, k = x.shape
    layout, wide = _prepare_metadata_L(x, x_scale, masked_m, e, t, k)
    if (
        k % 4 == 0
        and (
            layout is None
            or (layout[2] == 1 and layout[0] % 4 == 0 and (layout[1] % 4 == 0))
        )
        and (x.storage_offset() % 4 == 0)
    ):
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if e * t * k:
            input_address = x.data_ptr()
            output_address = out.data_ptr()
            scale_address = x_scale.data_ptr()
            output_scale_address = output_scale.data_ptr()
            programs = e * t * ((k // 4 + 512 - 1) // 512)
            if programs > 16777215:
                _address_packed_L.run(
                    input_address,
                    scale_address,
                    masked_m,
                    output_scale_address,
                    output_address,
                    (
                        t,
                        k,
                        layout,
                        512,
                        (16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1)[
                            input_address & 15
                        ],
                        (16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1)[
                            output_address & 15
                        ],
                    ),
                    num_warps=4,
                    enable_fp_fusion=False,
                    WIDE=wide,
                    grid=(65535,),
                    warmup=False,
                    PROGRAMS=programs,
                )
            else:
                _address_packed_L.run(
                    input_address,
                    scale_address,
                    masked_m,
                    output_scale_address,
                    output_address,
                    (
                        t,
                        k,
                        layout,
                        512,
                        (16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1)[
                            input_address & 15
                        ],
                        (16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1)[
                            output_address & 15
                        ],
                    ),
                    num_warps=4,
                    enable_fp_fusion=False,
                    WIDE=wide,
                    grid=(programs,),
                    warmup=False,
                )
    else:
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if e * t * k:
            input_address = x.data_ptr()
            output_address = out.data_ptr()
            scale_address = x_scale.data_ptr()
            output_scale_address = output_scale.data_ptr()
            programs = e * t * ((k + 4096 - 1) // 4096)
            if programs > 16777215:
                _address_kernel_L.run(
                    input_address,
                    scale_address,
                    masked_m,
                    output_scale_address,
                    output_address,
                    (
                        t,
                        k,
                        layout,
                        4096,
                        (16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1)[
                            input_address & 15
                        ],
                        (16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1)[
                            output_address & 15
                        ],
                    ),
                    num_warps=4,
                    enable_fp_fusion=False,
                    WIDE=wide,
                    grid=(65535,),
                    warmup=False,
                    PROGRAMS=programs,
                )
            else:
                _address_kernel_L.run(
                    input_address,
                    scale_address,
                    masked_m,
                    output_scale_address,
                    output_address,
                    (
                        t,
                        k,
                        layout,
                        4096,
                        (16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1)[
                            input_address & 15
                        ],
                        (16, 1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1)[
                            output_address & 15
                        ],
                    ),
                    num_warps=4,
                    enable_fp_fusion=False,
                    WIDE=wide,
                    grid=(programs,),
                    warmup=False,
                )
    return out


def fp8_per_token_to_per_tensor_quant(x, x_scale, masked_m, output_scale):
    if x.numel() <= 8388608:
        return fp8_per_token_to_per_tensor_quant_small(
            x, x_scale, masked_m, output_scale
        )
    return fp8_per_token_to_per_tensor_quant_large(
        x, x_scale, masked_m, output_scale
    )


__all__ = ["fp8_per_token_to_per_tensor_quant"]
