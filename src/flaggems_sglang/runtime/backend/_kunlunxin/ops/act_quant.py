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
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def _load_value(X, offsets, ROUND: tl.constexpr):
    if X.dtype.element_ty == tl.float16 and (not ROUND):
        raw = tl.load(X.to(tl.pointer_type(tl.uint16)) + offsets).to(tl.uint32)
        magnitude = raw & 32767
        exponent = magnitude >> 10
        mantissa = magnitude & 1023
        normal = ((magnitude << 13) + 939524096).to(tl.float32, bitcast=True)
        subnormal = mantissa.to(tl.float32) * 5.960464477539063e-08
        special = (magnitude << 13 | 2139095040).to(tl.float32, bitcast=True)
        absolute = tl.where(
            exponent == 0, subnormal, tl.where(exponent == 31, special, normal)
        )
        bits = absolute.to(tl.uint32, bitcast=True) | (raw & 32768) << 16
        return bits.to(tl.float32, bitcast=True)
    return tl.load(X + offsets).to(tl.float32)


@triton.jit
def _encode(v, NAN_STYLE: tl.constexpr = 0):
    if NAN_STYLE == 2:
        raw = v.to(tl.uint32, bitcast=True)
        mag = raw & 2147483647
        exponent = tl.maximum(mag & 2139095040, 1015021568)
        magic = (exponent + 167772160).to(tl.float32, bitcast=True)
        absolute = mag.to(tl.float32, bitcast=True)
        rounded = (absolute + magic).to(tl.uint32, bitcast=True)
        code = rounded + (exponent >> 20) - 968
        code = tl.where(mag > 2139095040, 127, code)
        return (code | raw >> 24 & 128).to(tl.uint8)
    else:
        raw = v.to(tl.uint32, bitcast=True)
        sign = raw >> 24 & 128
        mag = raw & 2147483647
        normal = mag + 3288858623 + (mag >> 20 & 1) >> 20
        sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) & 15
        code = tl.where(tl.abs(v) >= 0.015625, normal, sub)
        if NAN_STYLE == 0:
            code = tl.where(v != v, 127, code)
        elif NAN_STYLE == 2:
            code = tl.where(mag > 2139095040, 127, code)
        return (code | sign).to(tl.uint8)


@triton.jit
def _nonround_scale_bits(value, AXIS: tl.constexpr):
    magnitude = tl.abs(value).to(tl.uint32, bitcast=True)
    maximum_bits = tl.maximum(tl.max(magnitude, axis=AXIS), 953267991)
    scale = tl.div_rn(maximum_bits.to(tl.float32, bitcast=True), 448.0)
    return tl.where(
        maximum_bits > 2139095040,
        tl.full((), 2143289344, tl.uint32),
        scale.to(tl.uint32, bitcast=True),
    )


@triton.jit
def _nonround_code(value, scale_bits):
    normalized = tl.div_rn(value, scale_bits.to(tl.float32, bitcast=True))
    code = _encode(normalized, 2)
    invalid = scale_bits > 2139095040
    return tl.where(invalid, 127, code).to(tl.uint8)


@triton.jit
def _fallback_group(
    X,
    Q,
    S,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    group,
):
    lanes = tl.arange(0, BLOCK)
    if BLOCK == GROUP:
        value = _load_value(X, group * GROUP + lanes, ROUND)
    else:
        value = _load_value(
            X, group * GROUP + tl.minimum(lanes, GROUP - 1), ROUND
        )
        value = tl.where(lanes < GROUP, value, 0.0)
    if ROUND:
        amax = tl.maximum(tl.max(tl.abs(value), axis=0), 0.0001)
        scale = tl.div_rn(amax, 448.0)
        if ROUND:
            scale_bits = scale.to(tl.uint32, bitcast=True)
            exponent = (scale_bits >> 23 & 255).to(tl.int32) - 127
            exponent = exponent + (scale_bits & 8388607 != 0).to(tl.int32)
            bits = tl.minimum(tl.maximum(exponent + 127, 1), 254) << 23
            scale = bits.to(tl.float32, bitcast=True)
        quantized = tl.minimum(
            tl.maximum(tl.div_rn(value, scale), -448.0), 448.0
        )
        code = _encode(quantized)
        tl.store(Q + group * GROUP + lanes, code, lanes < GROUP)
        tl.store(S + group, scale)
    else:
        scale_bits = _nonround_scale_bits(value, 0)
        code = _nonround_code(value, scale_bits)
        tl.store(Q + group * GROUP + lanes, code, lanes < GROUP)
        tl.store(S.to(tl.pointer_type(tl.uint32)) + group, scale_bits)


@triton.jit
def _fallback(
    X, Q, S, GROUP: tl.constexpr, BLOCK: tl.constexpr, ROUND: tl.constexpr
):
    _fallback_group(X, Q, S, GROUP, BLOCK, ROUND, tl.program_id(0))


@triton.jit
def _tile(X, Q, S, META: tl.constexpr, tile):
    GROUP: tl.constexpr = META[0]
    TILES: tl.constexpr = META[1]
    BLOCK: tl.constexpr = META[2]
    ROWS: tl.constexpr = META[3]
    ROUND: tl.constexpr = META[4]
    if TILES * ROWS * GROUP >= 2147483648:
        groups = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
    else:
        groups = tile * ROWS + tl.arange(0, ROWS)
    lanes = tl.arange(0, BLOCK)
    if BLOCK == GROUP:
        value = _load_value(X, groups[:, None] * GROUP + lanes[None, :], ROUND)
    else:
        value = _load_value(
            X,
            groups[:, None] * GROUP + tl.minimum(lanes[None, :], GROUP - 1),
            ROUND,
        )
        value = tl.where(lanes[None, :] < GROUP, value, 0.0)
    if ROUND:
        amax = tl.maximum(tl.max(tl.abs(value), axis=1), 0.0001)
        scale = tl.div_rn(amax, 448.0)
        if ROUND:
            bits = scale.to(tl.uint32, bitcast=True)
            bits = tl.minimum(tl.maximum(bits + 8388607 >> 23, 1), 254) << 23
            scale = bits.to(tl.float32, bitcast=True)
        if ROUND:
            quantized = tl.minimum(
                tl.maximum(value * tl.div_rn(1.0, scale[:, None]), -448.0),
                448.0,
            )
        else:
            quantized = tl.minimum(
                tl.maximum(tl.div_rn(value, scale[:, None]), -448.0), 448.0
            )
        code = _encode(quantized)
        if BLOCK == GROUP:
            tl.store(Q + groups[:, None] * GROUP + lanes[None, :], code)
        else:
            tl.store(
                Q + groups[:, None] * GROUP + lanes[None, :],
                code,
                lanes[None, :] < GROUP,
            )
        tl.store(S + groups, scale)
    else:
        scale_bits = _nonround_scale_bits(value, 1)
        code = _nonround_code(value, scale_bits[:, None])
        if BLOCK == GROUP:
            tl.store(Q + groups[:, None] * GROUP + lanes[None, :], code)
        else:
            tl.store(
                Q + groups[:, None] * GROUP + lanes[None, :],
                code,
                lanes[None, :] < GROUP,
            )
        tl.store(S.to(tl.pointer_type(tl.uint32)) + groups, scale_bits)


@triton.jit
def _kernel(X, Q, S, META: tl.constexpr):
    if META[1] > 2147483647:
        for tile in range(
            tl.program_id(0).to(tl.int64),
            tl.full((), META[1], tl.int64),
            tl.num_programs(0),
        ):
            _tile(X, Q, S, META, tile)
    else:
        _tile(X, Q, S, META, tl.program_id(0))


def act_quant(x, block_size=128, scale_fmt=None):
    x = x.contiguous()
    shape = x.shape
    q = torch.empty(shape, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(
        shape[:-1] + (shape[-1] // block_size,),
        device=x.device,
        dtype=torch.float32,
    )
    groups = x.numel() // block_size
    if groups:
        output = q.data_ptr()
        scales = s.data_ptr()
        assert output % 16 == 0 and scales % 16 == 0
        if (
            groups > 768
            and x.dtype == torch.bfloat16
            and scale_fmt is None
            and block_size == 128
            and groups % 16 == 0
            and groups * block_size < 2147483648
            and x.data_ptr() % 64 == 0
            and output % 64 == 0
            and scales % 64 == 0
        ):
            row_shift = min(4, ((groups - 1) // 768).bit_length())
            rows = 1 << row_shift
            _raw_kernel179.run(
                x,
                output,
                scales,
                groups,
                row_shift,
                grid=(min(triton.cdiv(groups, rows * 64), 12),),
                warmup=False,
                num_warps=1,
                enable_fp_fusion=False,
            )
            return (q, s)
        block = 1 << (block_size - 1).bit_length()
        if block != block_size or (
            x.dtype == torch.float16 and scale_fmt is None
        ):
            if groups * block_size >= 2147483648:
                _wide_address_fallback.run(
                    x,
                    output,
                    scales,
                    block_size,
                    block,
                    scale_fmt is not None,
                    groups,
                    grid=(min(groups, 65535),),
                    warmup=False,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                _address_fallback.run(
                    x,
                    output,
                    scales,
                    block_size,
                    block,
                    scale_fmt is not None,
                    grid=(groups,),
                    warmup=False,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        else:
            rows = min(64, groups & -groups)
            tiles = groups // rows
            meta = (block_size, tiles, block, rows, scale_fmt is not None)
            _address_kernel.run(
                x,
                output,
                scales,
                meta,
                grid=(65535 if tiles > 2147483647 else tiles,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (q, s)


@triton.jit(do_not_specialize=["Q", "S"])
def _address_kernel(X, Q, S, META: tl.constexpr):
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _kernel(X, output, scales, META)


@triton.jit(do_not_specialize=["Q", "S"])
def _address_fallback(
    X, Q, S, GROUP: tl.constexpr, BLOCK: tl.constexpr, ROUND: tl.constexpr
):
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _fallback(X, output, scales, GROUP, BLOCK, ROUND)


@triton.jit(do_not_specialize=["Q", "S"])
def _wide_address_fallback(
    X,
    Q,
    S,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    GROUPS: tl.constexpr,
):
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    for group in range(
        tl.program_id(0).to(tl.int64),
        tl.full((), GROUPS, tl.int64),
        tl.num_programs(0),
    ):
        _fallback_group(X, output, scales, GROUP, BLOCK, ROUND, group)


RAW179 = (
    "typedef unsigned short __attribute__((address_space(1))) gu1"
    "6;\n"
    "typedef unsigned char __attribute__((address_space(1))) gu8;"
    "\n"
    "typedef unsigned int __attribute__((address_space(1))) gu32;"
    "\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)"
    "));\n"
    "typedef int i32v __attribute__((ext_vector_type(16)));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "union fvbits179 { f32v f; u32v u; };\n"
    "union fbits179 { float f; unsigned int u; };\n"
    "namespace xpu { __attribute__((device)) float __fdiv_rn(floa"
    "t, float); }\n"
    "\n"
    "__attribute__((device,always_inline)) u32v opaque179(u32v va"
    "lue) {\n"
    '    __asm__("" : "+v"(value));\n'
    "    return value;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v signbit179(u32v v"
    "alue) {\n"
    "    return opaque179(opaque179(value) >> 31);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v negmask179(u32v v"
    "alue) {\n"
    "    return opaque179(0u - (opaque179(value) >> 31));\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v select179(u32v ma"
    "sk, u32v yes, u32v no) {\n"
    "    return no ^ ((yes ^ no) & mask);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v maxsmall179(u32v "
    "left, u32v right) {\n"
    "    u32v difference = left - right;\n"
    "    return left - (difference & negmask179(difference));\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v minsmall179(u32v "
    "left, u32v right) {\n"
    "    u32v difference = left - right;\n"
    "    return right + (difference & negmask179(difference));\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v remainder179(u32v"
    " raw, unsigned int sb) {\n"
    "    u32v magnitude = raw & 32767u;\n"
    "    u32v exponent = magnitude >> 7;\n"
    "    u32v mantissa = (magnitude & 127u) | 128u;\n"
    "    u32v denominator = (sb & 8388607u) | 8388608u;\n"
    "    u32v E = exponent - (sb >> 23);\n"
    "    u32v lower = signbit179((mantissa << 16) - denominator);"
    "\n"
    "    u32v k = E - lower;\n"
    "    u32v normal = ~negmask179(k + 6u);\n"
    "    u32v shift = reinterpret_cast<u32v>(__builtin_elementwis"
    "e_min(\n"
    "        __builtin_elementwise_max(reinterpret_cast<i32v>(E +"
    " 25u), (i32v)15),\n"
    "        reinterpret_cast<i32v>(19u + lower)));\n"
    "    u32v numerator = mantissa << shift;\n"
    "    u32v remainder = numerator - ((denominator << 3) & norma"
    "l);\n"
    "    u32v n = 8u & normal;\n"
    "    u32v take4 = ~negmask179(remainder - (denominator << 2))"
    ";\n"
    "    remainder = remainder - ((denominator << 2) & take4);\n"
    "    n = n + (4u & take4);\n"
    "    u32v take2 = ~negmask179(remainder - (denominator << 1))"
    ";\n"
    "    remainder = remainder - ((denominator << 1) & take2);\n"
    "    n = n + (2u & take2);\n"
    "    u32v take1 = ~negmask179(remainder - denominator);\n"
    "    remainder = remainder - (denominator & take1);\n"
    "    n = n + (1u & take1);\n"
    "    u32v cutoff = take1 & 1u;\n"
    "    cutoff = select179(take2, denominator >> 22, cutoff);\n"
    "    cutoff = select179(take4, denominator >> 21, cutoff);\n"
    "    cutoff = select179(normal, denominator >> 20, cutoff);\n"
    "    u32v delta = (remainder << 1) - denominator;\n"
    "    u32v up = 1u - signbit179(delta + (cutoff ^ ~take1));\n"
    "    u32v code = (((k + 6u) << 3) & normal) + n + up;\n"
    "    code = code & ~negmask179(E + 10u);\n"
    "    return code | ((raw >> 8) & 128u);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode179(u32v ra"
    "w) {\n"
    "    u32v magnitude = raw & 2147483647u;\n"
    "    u32v exponent = maxsmall179(magnitude & 2139095040u, 101"
    "5021568u);\n"
    "    fvbits179 absolute, magic, rounded;\n"
    "    absolute.u = magnitude;\n"
    "    magic.u = exponent + 167772160u;\n"
    "    rounded.f = absolute.f + magic.f;\n"
    "    u32v code = rounded.u + (exponent >> 20) - 968u;\n"
    "    return (code | ((raw >> 24) & 128u)) & 255u;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v mac179(f32v a, f3"
    "2v b, f32v c) {\n"
    '    __asm__("vmac.f.mz.rn %0{mr1}, %1, %2" : "+v"(c) : "v"(a'
    '), "v"(b));\n'
    "    return c;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v corrected179(\n"
    "    u32v raw, float scale, float reciprocal) {\n"
    "    fvbits179 value, coarse, corrected;\n"
    "    value.u = (raw & 32767u) << 16;\n"
    "    coarse.f = value.f * (f32v)reciprocal;\n"
    "    coarse.u = opaque179(coarse.u);\n"
    "    f32v residual = mac179(-coarse.f, (f32v)scale, value.f);"
    "\n"
    "    corrected.f = mac179(residual, (f32v)reciprocal, coarse."
    "f);\n"
    "    return encode179(corrected.u & 2147483647u) | ((raw >> 8"
    ") & 128u);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v nonfinite179(\n"
    "    u32v raw, unsigned int positive_nan, unsigned int negati"
    "ve_nan) {\n"
    "    u32v magnitude = raw & 32767u;\n"
    "    u32v sign = (raw >> 8) & 128u;\n"
    "    u32v infinity = ~negmask179(magnitude - 32640u);\n"
    "    u32v nan = select179(negmask179(raw << 16), (u32v)negati"
    "ve_nan, (u32v)positive_nan);\n"
    "    return select179(infinity, nan, sign);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v packpairs179(u32v"
    " a, u32v b) {\n"
    "    u32v a1 = __builtin_xpu2_vmerge_l_u(a, b);\n"
    "    u32v b1 = __builtin_xpu2_vmerge_h_u(a, b);\n"
    "    u32v a2 = __builtin_xpu2_vmerge_l_u(a1, b1);\n"
    "    u32v b2 = __builtin_xpu2_vmerge_h_u(a1, b1);\n"
    "    u32v a3 = __builtin_xpu2_vmerge_l_u(a2, b2);\n"
    "    u32v b3 = __builtin_xpu2_vmerge_h_u(a2, b2);\n"
    "    u32v a4 = __builtin_xpu2_vmerge_l_u(a3, b3);\n"
    "    u32v b4 = __builtin_xpu2_vmerge_h_u(a3, b3);\n"
    "    return (a4 & 65535u) | (b4 << 16);\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device179(\n'
    "    const gu16* X, gu8* Q, gu32* S, int groups, int pid, int"
    " np, int row_shift) {\n"
    "    unsigned int input[1024] __attribute__((aligned(64)));\n"
    "    unsigned int scales[16] __attribute__((aligned(64)));\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    int worker = pid * 64 + core;\n"
    "    int workers = np * 64;\n"
    "    int quotient = groups / workers;\n"
    "    int extra = groups - quotient * workers;\n"
    "    int prefix = worker < extra ? worker : extra;\n"
    "    int first_group = quotient * worker + prefix;\n"
    "    int end_group = first_group + quotient + (worker < extra"
    " ? 1 : 0);\n"
    "    for (int first = first_group; first < end_group;) {\n"
    "        int rows = end_group - first;\n"
    "        if (rows > 16) rows = 16;\n"
    "        __builtin_xpu_gm2lm_v3(input, X + ((long long)first "
    "<< 7), 0, rows * 256);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        for (int row = 0; row < rows; ++row) {\n"
    "            fvbits179 maximum;\n"
    "            maximum.u = (u32v)1065353216u;\n"
    "            for (int offset = 0; offset < 64; offset += 16) "
    "{\n"
    "                u32v packed = *((u32v*)(input + row * 64 + o"
    "ffset));\n"
    "                fvbits179 low, high;\n"
    "                low.u = (packed & 32767u) | 1065353216u;\n"
    "                high.u = ((packed >> 16) & 32767u) | 1065353"
    "216u;\n"
    "                maximum.f = __builtin_elementwise_max(maximu"
    "m.f, __builtin_elementwise_max(low.f, high.f));\n"
    "            }\n"
    "            fbits179 maximum_value;\n"
    "            maximum_value.f = __builtin_reduce_max(maximum.f"
    ");\n"
    "            unsigned int maximum_bits = maximum_value.u << 1"
    "6;\n"
    "            maximum_bits = maximum_bits < 953267991u ? 95326"
    "7991u : maximum_bits;\n"
    "            fbits179 scale;\n"
    "            scale.u = maximum_bits;\n"
    "            scale.f = xpu::__fdiv_rn(scale.f, 448.0f);\n"
    "            if (maximum_bits > 2139095040u) scale.u = 214328"
    "9344u;\n"
    "            scales[row] = scale.u;\n"
    "            bool finite_scale = scale.u >= 8388608u && scale"
    ".u < 2139095040u;\n"
    "            float reciprocal = 0.0f;\n"
    "            if (finite_scale) {\n"
    "                reciprocal = xpu::__fdiv_rn(1.0f, scale.f);\n"
    "            }\n"
    "            unsigned int positive_nan = 127u;\n"
    "            unsigned int negative_nan = 127u;\n"
    "            if (scale.u == 2139095040u) {\n"
    "                fbits179 positive, negative, negative_scale;"
    "\n"
    "                negative_scale.u = scale.u | 2147483648u;\n"
    "                positive.f = xpu::__fdiv_rn(scale.f, scale.f"
    ");\n"
    "                negative.f = xpu::__fdiv_rn(negative_scale.f"
    ", scale.f);\n"
    "                positive_nan = 127u | ((positive.u >> 24) & "
    "128u);\n"
    "                negative_nan = 127u | ((negative.u >> 24) & "
    "128u);\n"
    "            }\n"
    "            for (int offset = 0; offset < 64; offset += 32) "
    "{\n"
    "                u32v pairs[2];\n"
    "                for (int part = 0; part < 2; ++part) {\n"
    "                    int index = offset + part * 16;\n"
    "                    u32v low, high;\n"
    "                    if (finite_scale) {\n"
    "                        u32v packed = *((u32v*)(input + row "
    "* 64 + index));\n"
    "                        low = corrected179(packed & 65535u, "
    "scale.f, reciprocal);\n"
    "                        high = corrected179(packed >> 16, sc"
    "ale.f, reciprocal);\n"
    "                    } else {\n"
    "                        u32v packed = *((u32v*)(input + row "
    "* 64 + index));\n"
    "                        low = nonfinite179(packed & 65535u, "
    "positive_nan, negative_nan);\n"
    "                        high = nonfinite179(packed >> 16, po"
    "sitive_nan, negative_nan);\n"
    "                        u32v nanmask = negmask179(2139095040"
    "u - (u32v)scale.u);\n"
    "                        low = select179(nanmask, (u32v)127u,"
    " low);\n"
    "                        high = select179(nanmask, (u32v)127u"
    ", high);\n"
    "                    }\n"
    "                    pairs[part] = low | (high << 8);\n"
    "                }\n"
    "                *((u32v*)(input + row * 32 + offset / 2)) = "
    "packpairs179(pairs[0], pairs[1]);\n"
    "            }\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(Q + ((long long)first << 7), "
    "input, 0, rows * 128);\n"
    "        __builtin_xpu_lm2gm_v3(S + first, scales, 0, rows * "
    "4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        first += rows;\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW179, flags=("-ffp-contract=off",))
def _device179(X, Q, S, groups, pid, np, row_shift):
    pass


@triton.jit(do_not_specialize=["Q", "S"])
def _raw_kernel179(X, Q, S, GROUPS: tl.constexpr, ROW_SHIFT: tl.constexpr):
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 64)
    scale = tl.multiple_of(S.to(tl.uint64).to(tl.pointer_type(tl.uint32)), 64)
    tle.raw.call(
        _device179,
        (
            X,
            output,
            scale,
            tl.full((), GROUPS, tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
            tl.full((), ROW_SHIFT, tl.int32),
        ),
    )


__all__ = ["act_quant"]
