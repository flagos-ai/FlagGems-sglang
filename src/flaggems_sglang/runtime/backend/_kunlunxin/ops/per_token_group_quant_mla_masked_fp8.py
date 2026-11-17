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
def _encode(value):
    raw = value.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = raw & 2147483647
    exponent = tl.maximum(magnitude & 2139095040, 1015021568)
    magic = (exponent + 167772160).to(tl.float32, bitcast=True)
    rounded = (tl.abs(value) + magic).to(tl.uint32, bitcast=True)
    code = rounded + (exponent >> 20) - 968
    code = tl.where(value != value, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _quantize(X, Q, S, MASKED, EPS, META: tl.constexpr):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    GROUP: tl.constexpr = META[2]
    ALIGNED_M: tl.constexpr = META[3]
    STRIDE_B: tl.constexpr = META[4]
    STRIDE_M: tl.constexpr = META[5]
    STRIDE_K: tl.constexpr = META[6]
    ROWS: tl.constexpr = META[7]
    BLOCK: tl.constexpr = META[8]
    EXACT_SCALE: tl.constexpr = META[9]
    WIDE: tl.constexpr = META[10]
    GROUPS: tl.constexpr = K // GROUP
    flat_pid = tl.program_id(0)
    if WIDE:
        flat_pid = flat_pid.to(tl.int64)
    TILES_PER_BATCH: tl.constexpr = M * GROUPS // ROWS
    batch = flat_pid // TILES_PER_BATCH
    pid = flat_pid % TILES_PER_BATCH
    indices = pid * ROWS + tl.arange(0, ROWS)
    tokens = indices // GROUPS
    groups = indices % GROUPS
    lanes = tl.arange(0, BLOCK)
    if WIDE:
        lanes = lanes.to(tl.int64)
    columns = groups[:, None] * GROUP + lanes[None, :]
    value = tl.load(
        X + batch * STRIDE_B + tokens[:, None] * STRIDE_M + columns * STRIDE_K,
        lanes[None, :] < GROUP,
        other=0.0,
    ).to(tl.float32)
    maximum_bits = tl.max(
        value.to(tl.uint32, bitcast=True) & 2147483647, axis=1
    )
    maximum = maximum_bits.to(tl.float32, bitcast=True)
    absolute = tl.maximum(maximum, EPS)
    absolute = tl.where(maximum_bits > 2139095040, maximum, absolute)
    if EXACT_SCALE:
        scale = tl.div_rn(absolute, 448.0)
    else:
        scale = absolute * 0.0022321429569274187
    divided = tl.div_rn(value, scale[:, None])
    quantized = tl.minimum(tl.maximum(divided, -448.0), 448.0)
    code = tl.where(divided != divided, 127, _encode(quantized)).to(tl.uint8)
    tl.store(
        Q + batch * ALIGNED_M * K + tokens[:, None] * K + columns,
        code,
        lanes[None, :] < GROUP,
    )
    tl.store(
        S + batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens, scale
    )
    if pid == 0:
        tl.store(MASKED + batch, M)


@triton.jit
def _encode_bounded(value):
    raw = value.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = tl.minimum(raw & 2147483647, 1139802112)
    exponent = tl.maximum(magnitude >> 23, 121)
    magic_bits = (exponent * 8388616 + 167771192) | sign
    rounded = (
        magnitude.to(tl.float32, bitcast=True)
        + magic_bits.to(tl.float32, bitcast=True)
    ).to(tl.uint32, bitcast=True)
    return rounded.to(tl.uint8)


@triton.jit
def _quantize_padded(X, Q, S, MASKED, EPS, META: tl.constexpr):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    GROUP: tl.constexpr = META[2]
    ALIGNED_M: tl.constexpr = META[3]
    STRIDE_B: tl.constexpr = META[4]
    STRIDE_M: tl.constexpr = META[5]
    STRIDE_K: tl.constexpr = META[6]
    ROWS: tl.constexpr = META[7]
    BLOCK: tl.constexpr = META[8]
    EXACT_SCALE: tl.constexpr = META[9]
    WIDE: tl.constexpr = META[10]
    GROUPS: tl.constexpr = K // GROUP
    flat_pid = tl.program_id(0)
    if WIDE:
        flat_pid = flat_pid.to(tl.int64)
    TILES_PER_BATCH: tl.constexpr = M * GROUPS // ROWS
    batch = flat_pid // TILES_PER_BATCH
    pid = flat_pid % TILES_PER_BATCH
    indices = pid * ROWS + tl.arange(0, ROWS)
    tokens = indices // GROUPS
    groups = indices % GROUPS
    lanes = tl.arange(0, BLOCK)
    if WIDE:
        lanes = lanes.to(tl.int64)
    load_columns = groups[:, None] * GROUP + tl.minimum(
        lanes[None, :], GROUP - 1
    )
    value = tl.load(
        X
        + batch * STRIDE_B
        + tokens[:, None] * STRIDE_M
        + load_columns * STRIDE_K
    ).to(tl.float32)
    value = tl.where(lanes[None, :] < GROUP, value, 0.0)
    maximum_bits = tl.max(
        value.to(tl.uint32, bitcast=True) & 2147483647, axis=1
    )
    maximum = maximum_bits.to(tl.float32, bitcast=True)
    absolute = tl.maximum(maximum, EPS)
    absolute = tl.where(maximum_bits > 2139095040, maximum, absolute)
    if EXACT_SCALE:
        scale = tl.div_rn(absolute, 448.0)
    else:
        scale = absolute * 0.0022321429569274187
    divided = tl.div_rn(value, scale[:, None])
    quantized = tl.minimum(tl.maximum(divided, -448.0), 448.0)
    code = tl.where(divided != divided, 127, _encode(quantized)).to(tl.uint8)
    tl.store(
        Q
        + batch * M * GROUPS * BLOCK
        + indices[:, None] * BLOCK
        + lanes[None, :],
        code,
    )
    tl.store(
        S + batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens, scale
    )
    if pid == 0:
        tl.store(MASKED + batch, M)


@triton.jit
def _copy_codes(
    TEMP,
    Q,
    M: tl.constexpr,
    K: tl.constexpr,
    GROUP: tl.constexpr,
    ALIGNED: tl.constexpr,
    PADDED: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    if WIDE:
        pid = pid.to(tl.int64)
        batch = batch.to(tl.int64)
    indices = pid * 256 + tl.arange(0, 256)
    if WIDE:
        indices = indices.to(tl.int64)
    safe_indices = tl.minimum(indices, M * K - 1)
    source = (
        batch * M * (K // GROUP) * PADDED
        + (safe_indices // GROUP) * PADDED
        + safe_indices % GROUP
    )
    codes = tl.load(TEMP + source)
    tl.store(Q + batch * ALIGNED * K + indices, codes, indices < M * K)


@triton.jit
def _paired(X, Q, S, MASKED, EPS, META: tl.constexpr):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    GROUP: tl.constexpr = META[2]
    ALIGNED_M: tl.constexpr = META[3]
    STRIDE_B: tl.constexpr = META[4]
    STRIDE_M: tl.constexpr = META[5]
    ROWS: tl.constexpr = META[7]
    WIDE: tl.constexpr = META[10]
    GROUPS: tl.constexpr = K // GROUP
    flat_pid = tl.program_id(0)
    if WIDE:
        flat_pid = flat_pid.to(tl.int64)
    TILES_PER_BATCH: tl.constexpr = M * GROUPS // ROWS
    batch = flat_pid // TILES_PER_BATCH
    pid = flat_pid % TILES_PER_BATCH
    indices = pid * ROWS + tl.arange(0, ROWS)
    tokens = indices // GROUPS
    groups = indices % GROUPS
    words = tl.arange(0, GROUP // 2)
    if WIDE:
        words = words.to(tl.int64)
    columns = groups[:, None] * (GROUP // 2) + words[None, :]
    raw = tl.load(
        X.to(tl.pointer_type(tl.uint32))
        + batch * (STRIDE_B // 2)
        + tokens[:, None] * (STRIDE_M // 2)
        + columns
    )
    a = (raw << 16).to(tl.float32, bitcast=True)
    b = (raw & 4294901760).to(tl.float32, bitcast=True)
    magnitudes = tl.maximum(
        a.to(tl.uint32, bitcast=True) & 2147483647,
        b.to(tl.uint32, bitcast=True) & 2147483647,
    )
    maximum_bits = tl.max(magnitudes, axis=1)
    maximum = maximum_bits.to(tl.float32, bitcast=True)
    absolute = tl.where(
        maximum_bits > 2139095040, maximum, tl.maximum(maximum, EPS)
    )
    scale = tl.div_rn(absolute, 448.0)
    qa = tl.div_rn(a, scale[:, None])
    qb = tl.div_rn(b, scale[:, None])
    if META[11]:
        ca = _encode_bounded(qa).to(tl.uint32)
        cb = _encode_bounded(qb).to(tl.uint32)
    else:
        ca = tl.where(
            qa != qa, 127, _encode(tl.minimum(tl.maximum(qa, -448.0), 448.0))
        ).to(tl.uint32)
        cb = tl.where(
            qb != qb, 127, _encode(tl.minimum(tl.maximum(qb, -448.0), 448.0))
        ).to(tl.uint32)
    packed = (ca | cb << 8).to(tl.uint16)
    output = (
        batch * ALIGNED_M * (K // 2) + tokens[:, None] * (K // 2) + columns
    )
    tl.store(Q.to(tl.pointer_type(tl.uint16)) + output, packed)
    tl.store(
        S + batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens, scale
    )
    if pid == 0:
        tl.store(MASKED + batch, M)


@triton.jit
def _fill_masked(
    MASKED, M: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if B > 2147483648:
        pid = pid.to(tl.int64)
    indices = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(MASKED + indices, M, indices < B)


def per_token_group_quant_mla_masked_fp8(x, group_size, eps):
    if eps != eps:
        eps = 0.0
    b, m, k = x.shape
    aligned_m = (m + 255) // 256 * 256
    groups = k // group_size
    q = torch.empty((b, aligned_m, k), device=x.device, dtype=torch.uint8)
    s = torch.empty(
        (b, groups, aligned_m), device=x.device, dtype=torch.float32
    )
    masked = torch.empty((b,), device=x.device, dtype=torch.int32)
    tiles = m * groups
    if b and tiles:
        if (
            x.dtype == torch.bfloat16
            and group_size == 128
            and 0 < k <= 512
            and k % 128 == 0
            and m % 16 == 0
            and x.is_contiguous()
            and b * m * k < 2147483648
            and x.data_ptr() % 64 == 0
            and q.data_ptr() % 64 == 0
            and s.data_ptr() % 64 == 0
        ):
            tile_tokens = min(
                16,
                triton.next_power_of_2(triton.cdiv(b * m, 768)),
                1 << ((2048 // k).bit_length() - 1),
            )
            _raw_kernel188.run(
                x,
                q,
                s,
                masked,
                eps,
                b,
                m,
                k,
                aligned_m,
                1e-30 <= eps < float("inf"),
                tile_tokens,
                grid=(min(triton.cdiv(b * (m // tile_tokens), 64), 12),),
                warmup=False,
                num_warps=1,
                enable_fp_fusion=False,
            )
            return (
                q.view(torch.float8_e4m3fn),
                s.transpose(1, 2),
                masked,
                m,
                aligned_m,
            )
        rows = min(64, tiles & -tiles)
        block = 1 << (group_size - 1).bit_length()
        strides = x.stride()
        if max(strides) > 2147483647:
            strides = tuple(
                0 if size == 1 else stride
                for size, stride in zip(x.shape, strides)
            )
        wide = b * aligned_m * k > 2147483648
        if not wide and not x.is_contiguous():
            wide = (
                sum(
                    (size - 1) * stride
                    for size, stride in zip(x.shape, strides)
                )
                >= 2147483648
            )
        meta = (m, k, group_size, aligned_m, *strides, rows, block, True, wide)
        if (
            group_size >= 2
            and group_size == block
            and strides[2] == 1
            and strides[0] % 2 == 0
            and strides[1] % 2 == 0
            and x.data_ptr() % 4 == 0
        ):
            _paired.run(
                x,
                q,
                s,
                masked,
                eps,
                meta + (1e-30 <= eps < float("inf"),),
                grid=(b * tiles // rows,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif group_size != block:
            padded_block = max(128, block)
            temporary = torch.empty(
                (b, tiles, padded_block), device=x.device, dtype=torch.uint8
            )
            temporary_wide = wide or b * tiles * padded_block > 2147483648
            temporary_meta = (
                m,
                k,
                group_size,
                aligned_m,
                *strides,
                rows,
                padded_block,
                True,
                temporary_wide,
            )
            _quantize_padded.run(
                x,
                temporary,
                s,
                masked,
                eps,
                temporary_meta,
                grid=(b * tiles // rows,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _copy_codes.run(
                temporary,
                q,
                m,
                k,
                group_size,
                aligned_m,
                padded_block,
                temporary_wide,
                grid=((m * k + 255) // 256, b),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _quantize.run(
                x,
                q,
                s,
                masked,
                eps,
                meta,
                grid=(b * tiles // rows,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    elif b:
        _fill_masked.run(
            masked, m, b, 128, grid=((b + 127) // 128,), warmup=False
        )
    return q.view(torch.float8_e4m3fn), s.transpose(1, 2), masked, m, aligned_m


RAW188 = (
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
    "    u32v nanmask = negmask179(2139095040u - magnitude);\n"
    "    code = select179(nanmask, 127u, code);\n"
    "    return (code | ((raw >> 24) & 128u)) & 255u;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v finite_encode188("
    "u32v raw) {\n"
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
    "    return finite_encode188(corrected.u & 2147483647u) | ((r"
    "aw >> 8) & 128u);\n"
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
    "__attribute__((device,always_inline)) u32v encode_bounded188"
    "(u32v raw) {\n"
    "    u32v sign = (raw >> 24) & 128u;\n"
    "    u32v magnitude = minsmall179(raw & 2147483647u, 11398021"
    "12u);\n"
    "    u32v exponent = maxsmall179(magnitude >> 23, 121u);\n"
    "    fvbits179 absolute, magic, rounded;\n"
    "    absolute.u = magnitude;\n"
    "    magic.u = (exponent * 8388616u + 167771192u) | sign;\n"
    "    rounded.f = absolute.f + magic.f;\n"
    "    return rounded.u & 255u;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode_clamped188"
    "(u32v raw) {\n"
    "    u32v magnitude = raw & 2147483647u;\n"
    "    u32v clamped = minsmall179(magnitude, 1138753536u) | (ra"
    "w & 2147483648u);\n"
    "    u32v code = encode179(clamped);\n"
    "    u32v nanmask = negmask179(2139095040u - magnitude);\n"
    "    return select179(nanmask, 127u, code);\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device188(\n'
    "    const gu16* X, gu8* Q, gu32* S, gu32* MASKED,\n"
    "    float EPS, int B, int M, int K, int ALIGNED_M, int BOUND"
    "ED, int R, int pid, int np) {\n"
    "    unsigned int input[1024] __attribute__((aligned(64)));\n"
    "    unsigned int scales[64] __attribute__((aligned(64)));\n"
    "    unsigned int quotient[128] __attribute__((aligned(64)));"
    "\n"
    "    unsigned int masked_value[16] __attribute__((aligned(64)"
    "));\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    int groups = K / 128;\n"
    "    int chunk_tokens = R;\n"
    "    int micros_per_batch = M / chunk_tokens;\n"
    "    int microtiles = B * micros_per_batch;\n"
    "    int workers = np * 64;\n"
    "    int worker = pid * 64 + core;\n"
    "    int first_micro = (int)((long long)microtiles * worker /"
    " workers);\n"
    "    int end_micro = (int)((long long)microtiles * (worker + "
    "1) / workers);\n"
    "    for (int micro = first_micro; micro < end_micro;) {\n"
    "        int batch = micro / micros_per_batch;\n"
    "        int offset_micro = micro % micros_per_batch;\n"
    "        int tile_micros = 16 / chunk_tokens;\n"
    "        if (tile_micros > end_micro - micro) tile_micros = e"
    "nd_micro - micro;\n"
    "        if (tile_micros > micros_per_batch - offset_micro) t"
    "ile_micros = micros_per_batch - offset_micro;\n"
    "        int first = offset_micro * chunk_tokens;\n"
    "        int tile_tokens = tile_micros * chunk_tokens;\n"
    "        masked_value[0] = (unsigned int)M;\n"
    "        for (int token_base = 0; token_base < tile_tokens; t"
    "oken_base += chunk_tokens) {\n"
    "            int token_count = tile_tokens - token_base;\n"
    "            if (token_count > chunk_tokens) token_count = ch"
    "unk_tokens;\n"
    "            long long x_offset = ((long long)batch * M + fir"
    "st + token_base) * K;\n"
    "            __builtin_xpu_gm2lm_v3(input, X + x_offset, 0, t"
    "oken_count * K * 2);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            int rows = token_count * groups;\n"
    "            for (int row = 0; row < rows; ++row) {\n"
    "                fvbits179 maximum;\n"
    "                maximum.u = (u32v)1065353216u;\n"
    "                for (int offset = 0; offset < 64; offset += "
    "16) {\n"
    "                    u32v packed = *((u32v*)(input + row * 64"
    " + offset));\n"
    "                    fvbits179 low, high;\n"
    "                    low.u = (packed & 32767u) | 1065353216u;"
    "\n"
    "                    high.u = ((packed >> 16) & 32767u) | 106"
    "5353216u;\n"
    "                    maximum.f = __builtin_elementwise_max(ma"
    "ximum.f, __builtin_elementwise_max(low.f, high.f));\n"
    "                }\n"
    "                fbits179 maximum_value;\n"
    "                maximum_value.f = __builtin_reduce_max(maxim"
    "um.f);\n"
    "                unsigned int maximum_bits = maximum_value.u "
    "<< 16;\n"
    "                fbits179 scale;\n"
    "                scale.u = maximum_bits;\n"
    "                if (maximum_bits <= 2139095040u && scale.f <"
    " EPS) scale.f = EPS;\n"
    "                bool use_corrected = scale.u == maximum_bits"
    " && maximum_bits >= 953267991u && maximum_bits < 2139095040u"
    ";\n"
    "                scale.f = xpu::__fdiv_rn(scale.f, 448.0f);\n"
    "                int group = row % groups;\n"
    "                int token = token_base + row / groups;\n"
    "                scales[group * 16 + token] = scale.u;\n"
    "                bool finite_scale = scale.u >= 92274688u && "
    "scale.u < 2139095040u;\n"
    "                float reciprocal = 0.0f;\n"
    "                if (use_corrected) reciprocal = xpu::__fdiv_"
    "rn(1.0f, scale.f);\n"
    "                if (!finite_scale) {\n"
    "                    for (int lane = 0; lane < 64; ++lane) {\n"
    "                        unsigned int packed = input[row * 64"
    " + lane];\n"
    "                        fbits179 low, high;\n"
    "                        low.u = packed << 16;\n"
    "                        high.u = packed & 4294901760u;\n"
    "                        low.f = xpu::__fdiv_rn(low.f, scale."
    "f);\n"
    "                        high.f = xpu::__fdiv_rn(high.f, scal"
    "e.f);\n"
    "                        quotient[lane] = low.u;\n"
    "                        quotient[64 + lane] = high.u;\n"
    "                    }\n"
    "                    __builtin_xpu2_mfence(7);\n"
    "                }\n"
    "                for (int offset = 0; offset < 64; offset += "
    "32) {\n"
    "                    u32v pairs[2];\n"
    "                    for (int part = 0; part < 2; ++part) {\n"
    "                        int index = offset + part * 16;\n"
    "                        u32v low, high;\n"
    "                        if (finite_scale) {\n"
    "                            u32v packed = *((u32v*)(input + "
    "row * 64 + index));\n"
    "                            if (use_corrected) {\n"
    "                                low = corrected179(packed & "
    "65535u, scale.f, reciprocal);\n"
    "                                high = corrected179(packed >"
    "> 16, scale.f, reciprocal);\n"
    "                            } else {\n"
    "                                low = remainder179(packed & "
    "65535u, scale.u);\n"
    "                                high = remainder179(packed >"
    "> 16, scale.u);\n"
    "                            }\n"
    "                        } else if (BOUNDED) {\n"
    "                            low = encode_bounded188(*((u32v*"
    ")(quotient + index)));\n"
    "                            high = encode_bounded188(*((u32v"
    "*)(quotient + 64 + index)));\n"
    "                        } else {\n"
    "                            low = encode_clamped188(*((u32v*"
    ")(quotient + index)));\n"
    "                            high = encode_clamped188(*((u32v"
    "*)(quotient + 64 + index)));\n"
    "                        }\n"
    "                        pairs[part] = low | (high << 8);\n"
    "                    }\n"
    "                    *((u32v*)(input + row * 32 + offset / 2)"
    ") = packpairs179(pairs[0], pairs[1]);\n"
    "                }\n"
    "            }\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            long long q_offset = ((long long)batch * ALIGNED"
    "_M + first + token_base) * K;\n"
    "            __builtin_xpu_lm2gm_v3(Q + q_offset, input, 0, t"
    "oken_count * K);\n"
    "            if (token_base + token_count == tile_tokens) {\n"
    "                for (int group = 0; group < groups; ++group)"
    " {\n"
    "                    long long s_offset = ((long long)batch *"
    " groups + group) * ALIGNED_M + first;\n"
    "                    __builtin_xpu_lm2gm_v3(S + s_offset, sca"
    "les + group * 16, 0, tile_tokens * 4);\n"
    "                }\n"
    "                if (first == 0) {\n"
    "                    __builtin_xpu_lm2gm_v3(MASKED + batch, m"
    "asked_value, 0, 4);\n"
    "                }\n"
    "            }\n"
    "            __builtin_xpu2_mfence(7);\n"
    "        }\n"
    "        micro += tile_micros;\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW188, flags=("-ffp-contract=off",))
def _device188(X, Q, S, MASKED, EPS, B, M, K, ALIGNED_M, BOUNDED, R, pid, np):
    pass


@triton.jit
def _raw_kernel188(
    X,
    Q,
    S,
    MASKED,
    EPS,
    B: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    ALIGNED_M: tl.constexpr,
    BOUNDED: tl.constexpr,
    R: tl.constexpr,
):
    tle.raw.call(
        _device188,
        (
            X,
            Q,
            S.to(tl.pointer_type(tl.uint32)),
            MASKED.to(tl.pointer_type(tl.uint32)),
            EPS,
            tl.full((), B, tl.int32),
            tl.full((), M, tl.int32),
            tl.full((), K, tl.int32),
            tl.full((), ALIGNED_M, tl.int32),
            tl.full((), BOUNDED, tl.int32),
            tl.full((), R, tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


__all__ = ["per_token_group_quant_mla_masked_fp8"]
