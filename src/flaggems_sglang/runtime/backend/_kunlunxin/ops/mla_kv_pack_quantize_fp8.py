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

import math

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def _encode(value, E4: tl.constexpr, original, SCALE_SIGN: tl.constexpr):
    M: tl.constexpr = 3 if E4 else 2
    BIAS: tl.constexpr = 7 if E4 else 15
    MAX: tl.constexpr = 448.0 if E4 else 57344.0
    raw = value.to(tl.uint32, bitcast=True)
    sign = (
        original.to(tl.float32).to(tl.uint32, bitcast=True) >> 24 ^ SCALE_SIGN
    ) & 128
    mag = tl.minimum(
        raw & 2147483647,
        tl.full((), MAX, tl.float32).to(tl.uint32, bitcast=True),
    )
    exponent = tl.maximum(mag >> 23, 128 - BIAS)
    magic_bits = exponent * ((1 << 23) + (1 << M)) + (
        (23 - M << 23) - (128 - BIAS << M)
    )
    code = (
        mag.to(tl.float32, bitcast=True)
        + magic_bits.to(tl.float32, bitcast=True)
    ).to(tl.uint32, bitcast=True)
    code = tl.where(value != value, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _copy_tile(
    X,
    K_OUT,
    work,
    WIDTH: tl.constexpr,
    OUT_WIDTH: tl.constexpr,
    OFFSET: tl.constexpr,
    H: tl.constexpr,
    TS: tl.constexpr,
    HS: tl.constexpr,
    SCALE: tl.constexpr,
    SIGN: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    TILES: tl.constexpr = triton.cdiv(WIDTH, BLOCK)
    row = work // TILES
    col = work % TILES * BLOCK + tl.arange(0, BLOCK)
    token = row // H
    head = row % H
    original = tl.load(X + token * TS + head * HS + col, col < WIDTH, other=0)
    code = _encode(original.to(tl.float32) * SCALE, E4, original, SIGN)
    tl.store(K_OUT + row * OUT_WIDTH + OFFSET + col, code, col < WIDTH)


@triton.jit
def _flat_tile(
    X,
    K_OUT,
    tile,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    SIGN: tl.constexpr,
    E4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tile * BLOCK + tl.arange(0, BLOCK)
    if N % BLOCK == 0:
        original = tl.load(X + index)
    else:
        original = tl.load(X + index, index < N, other=0)
    code = _encode(original.to(tl.float32) * SCALE, E4, original, SIGN)
    if N % BLOCK == 0:
        tl.store(K_OUT + index, code)
    else:
        tl.store(K_OUT + index, code, index < N)


@triton.jit
def _stream_work(
    K,
    P,
    V,
    K_OUT,
    W,
    work,
    R: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    PE: tl.constexpr,
    D: tl.constexpr,
    KS: tl.constexpr,
    KH: tl.constexpr,
    PS: tl.constexpr,
    VS: tl.constexpr,
    VH: tl.constexpr,
    K_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
    K_SIGN: tl.constexpr,
    V_SIGN: tl.constexpr,
    E4: tl.constexpr,
    NB: tl.constexpr,
    PB: tl.constexpr,
    VB: tl.constexpr,
    VF: tl.constexpr,
    NT: tl.constexpr,
    PT: tl.constexpr,
):
    if work < NT:
        if N > 0:
            _copy_tile(
                K,
                K_OUT,
                work,
                N,
                N + PE,
                0,
                H,
                KS,
                KH,
                K_SCALE,
                K_SIGN,
                E4,
                NB,
            )
    elif work < NT + PT:
        if PE > 0:
            _copy_tile(
                P,
                K_OUT,
                work - NT,
                PE,
                N + PE,
                N,
                H,
                PS,
                0,
                K_SCALE,
                K_SIGN,
                E4,
                PB,
            )
    elif D > 0:
        if VF:
            _flat_tile(V, W, work - NT - PT, R * D, V_SCALE, V_SIGN, E4, VB)
        else:
            _copy_tile(
                V,
                W,
                work - NT - PT,
                D,
                D,
                0,
                H,
                VS,
                VH,
                V_SCALE,
                V_SIGN,
                E4,
                VB,
            )


@triton.jit
def _streams(K, P, V, K_OUT, W, META: tl.constexpr):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    PE: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    KS: tl.constexpr = META[5]
    KH: tl.constexpr = META[6]
    PS: tl.constexpr = META[7]
    VS: tl.constexpr = META[8]
    VH: tl.constexpr = META[9]
    K_SCALE: tl.constexpr = META[10]
    V_SCALE: tl.constexpr = META[11]
    K_SIGN: tl.constexpr = META[12]
    V_SIGN: tl.constexpr = META[13]
    E4: tl.constexpr = META[14]
    NB: tl.constexpr = META[15]
    PB: tl.constexpr = META[16]
    VB: tl.constexpr = META[17]
    VF: tl.constexpr = META[18]
    NT: tl.constexpr = R * triton.cdiv(N, NB)
    PT: tl.constexpr = R * triton.cdiv(PE, PB)
    VT: tl.constexpr = triton.cdiv(R * D, VB) if VF else R * triton.cdiv(D, VB)
    TOTAL: tl.constexpr = NT + PT + VT
    WIDE: tl.constexpr = (
        (R // H - 1) * KS + (H - 1) * KH + N - 1 > 2147483647
        or (R // H - 1) * PS + PE - 1 > 2147483647
        or (R // H - 1) * VS + (H - 1) * VH + D - 1 > 2147483647
        or (R * (N + PE) + NB + PB - 1 > 2147483647)
        or (R * D + VB - 1 > 2147483647)
        or (TOTAL > 2147483647)
        or (KS > 2147483647)
        or (KH > 2147483647)
        or (PS > 2147483647)
        or (VS > 2147483647)
        or (VH > 2147483647)
        or (H > 2147483647)
        or (N + PE > 2147483647)
        or (D > 2147483647)
    )
    start = tl.program_id(0)
    if WIDE or TOTAL > 1073741823:
        start = start.to(tl.int64)
    if TOTAL <= 1073741823:
        _stream_work(
            K,
            P,
            V,
            K_OUT,
            W,
            start,
            R,
            H,
            N,
            PE,
            D,
            KS,
            KH,
            PS,
            VS,
            VH,
            K_SCALE,
            V_SCALE,
            K_SIGN,
            V_SIGN,
            E4,
            NB,
            PB,
            VB,
            VF,
            NT,
            PT,
        )
    else:
        for work in range(start, TOTAL, tl.num_programs(0)):
            _stream_work(
                K,
                P,
                V,
                K_OUT,
                W,
                work,
                R,
                H,
                N,
                PE,
                D,
                KS,
                KH,
                PS,
                VS,
                VH,
                K_SCALE,
                V_SCALE,
                K_SIGN,
                V_SIGN,
                E4,
                NB,
                PB,
                VB,
                VF,
                NT,
                PT,
            )


@triton.jit(do_not_specialize=["O_ADDR", "W_ADDR"])
def _address_streams(K, P, V, O_ADDR, W_ADDR, META: tl.constexpr):
    output = tl.multiple_of(
        O_ADDR.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16
    )
    values = tl.multiple_of(
        W_ADDR.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16
    )
    _streams(K, P, V, output, values, META)


@triton.jit(do_not_specialize=["O_ADDR", "W_ADDR"])
def _segmented_selected(K, P, V, O_ADDR, W_ADDR, META: tl.constexpr):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    N: tl.constexpr = META[2]
    PE: tl.constexpr = META[3]
    D: tl.constexpr = META[4]
    PS: tl.constexpr = META[7]
    KSCALE: tl.constexpr = META[10]
    VSCALE: tl.constexpr = META[11]
    KSIGN: tl.constexpr = META[12]
    VSIGN: tl.constexpr = META[13]
    E4: tl.constexpr = META[14]
    BLOCK: tl.constexpr = 4096
    SEG: tl.constexpr = N if N <= PE and N <= D else PE if PE <= D else D
    KT: tl.constexpr = triton.cdiv(R * N, BLOCK)
    PT: tl.constexpr = triton.cdiv(R * PE, BLOCK)
    output = tl.multiple_of(
        O_ADDR.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16
    )
    values = tl.multiple_of(
        W_ADDR.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16
    )
    work = tl.program_id(0)
    is_k = work < KT
    is_p = (work >= KT) & (work < KT + PT)
    is_v = work >= KT + PT
    tile = tl.where(is_k, work, tl.where(is_p, work - KT, work - KT - PT))
    segment = tl.arange(0, BLOCK // SEG)[:, None] * SEG
    lane = tl.arange(0, SEG)[None, :]
    index = tile * BLOCK + segment + lane
    k_base = tl.where(is_k, tile, 0) * BLOCK + segment
    p_base = tl.where(is_p, tile, 0) * BLOCK + segment
    v_base = tl.where(is_v, tile, 0) * BLOCK + segment
    p_source = p_base // PE // H * PS + p_base % PE
    k_dest = k_base // N * (N + PE) + k_base % N
    p_dest = p_base // PE * (N + PE) + N + p_base % PE
    source = tl.where(
        is_k, K + k_base, tl.where(is_p, P + p_source, V + v_base)
    )
    destination = tl.where(
        is_k, output + k_dest, tl.where(is_p, output + p_dest, values + v_base)
    )
    limit = tl.where(is_k, R * N, tl.where(is_p, R * PE, R * D))
    if R * N % BLOCK == 0 and R * PE % BLOCK == 0 and (R * D % BLOCK == 0):
        original = tl.load(source + lane)
    else:
        original = tl.load(source + lane, index < limit, other=0)
    scale = tl.where(
        is_v, tl.full((), VSCALE, tl.float32), tl.full((), KSCALE, tl.float32)
    )
    sign = tl.where(is_v, VSIGN, KSIGN).to(tl.uint8)
    code = _encode(original.to(tl.float32) * scale, E4, original, 0) ^ sign
    if R * N % BLOCK == 0 and R * PE % BLOCK == 0 and (R * D % BLOCK == 0):
        tl.store(destination + lane, code)
    else:
        tl.store(destination + lane, code, index < limit)


def mla_kv_pack_quantize_fp8(
    k_nope, k_pe, v, k_scale_inv, v_scale_inv, fp8_dtype
):
    s, heads, nope = k_nope.shape
    rope, width = (k_pe.shape[-1], v.shape[-1])
    k_out = torch.empty(
        (s, heads, nope + rope), device=k_nope.device, dtype=fp8_dtype
    )
    v_out = torch.empty((s, heads, width), device=v.device, dtype=fp8_dtype)
    rows = s * heads
    if rows == 0 or not (nope or rope or width):
        return (k_out, v_out)
    k_address = k_out.data_ptr()
    v_address = v_out.data_ptr()
    if (
        k_nope.dtype == torch.bfloat16
        and k_pe.dtype == torch.bfloat16
        and (v.dtype == torch.bfloat16)
        and (s > 0)
        and (heads > 0)
        and all(
            (
                0 < size <= 512 and size % 64 == 0
                for size in (nope, rope, width)
            )
        )
        and k_nope.is_contiguous()
        and v.is_contiguous()
        and (k_pe.stride(-1) == 1)
        and (0 <= k_pe.stride(0) <= 9223372036854775807)
        and (k_pe.stride(0) % 32 == 0)
        and all(
            (
                address % 64 == 0
                for address in (
                    k_nope.data_ptr(),
                    k_pe.data_ptr(),
                    v.data_ptr(),
                    k_address,
                    v_address,
                )
            )
        )
        and (
            rows * max(2 * nope, 2 * width, nope + rope) <= 9223372036854775807
        )
        and (2 * ((s - 1) * k_pe.stride(0) + rope) <= 9223372036854775807)
    ):
        raw_meta = (
            rows,
            heads,
            nope,
            rope,
            width,
            k_pe.stride(0),
            k_scale_inv,
            v_scale_inv,
            128 if math.copysign(1.0, k_scale_inv) < 0 else 0,
            128 if math.copysign(1.0, v_scale_inv) < 0 else 0,
            fp8_dtype == torch.float8_e4m3fn,
        )
        target = (rows + 767) // 768
        job_heads = min(heads, 1 << (target - 1).bit_length())
        head_jobs = (heads + job_heads - 1) // job_heads
        _raw_kernel185.run(
            k_nope,
            k_pe,
            v,
            k_address,
            v_address,
            raw_meta,
            job_heads,
            head_jobs,
            grid=(min(triton.cdiv(s * head_jobs, 64), 12),),
            warmup=False,
            num_warps=1,
            enable_fp_fusion=False,
        )
        return (k_out, v_out)
    nb = min(1024, max(128, 1 << max(nope - 1, 0).bit_length()))
    pb = min(1024, max(128, 1 << max(rope - 1, 0).bit_length()))
    vf = v.is_contiguous()
    vb = (
        4096
        if vf
        else min(1024, max(128, 1 << max(width - 1, 0).bit_length()))
    )
    total = rows * ((nope + nb - 1) // nb + (rope + pb - 1) // pb)
    total += (
        (rows * width + vb - 1) // vb
        if vf
        else rows * ((width + vb - 1) // vb)
    )
    meta = (
        rows,
        heads,
        nope,
        rope,
        width,
        k_nope.stride(0),
        k_nope.stride(1),
        k_pe.stride(0),
        v.stride(0),
        v.stride(1),
        k_scale_inv,
        v_scale_inv,
        128 if math.copysign(1.0, k_scale_inv) < 0 else 0,
        128 if math.copysign(1.0, v_scale_inv) < 0 else 0,
        fp8_dtype == torch.float8_e4m3fn,
        nb,
        pb,
        vb,
        vf,
    )
    if total:
        assert k_address % 16 == 0 and v_address % 16 == 0
        if (
            nope > 0
            and rope > 0
            and (width > 0)
            and (nope <= 4096)
            and (rope <= 4096)
            and (width <= 4096)
            and (nope & nope - 1 == 0)
            and (rope & rope - 1 == 0)
            and (width & width - 1 == 0)
            and ((rows + 4096) * max(nope + rope, width) <= 2147483647)
            and (
                (s - 1 + (4095 + rope - 1) // rope) * k_pe.stride(0)
                + rope
                + 4095
                <= 2147483647
            )
            and k_nope.is_contiguous()
            and v.is_contiguous()
            and (min(nope, rope, width) >= 64)
        ):
            count = (
                (rows * nope + 4095) // 4096
                + (rows * rope + 4095) // 4096
                + (rows * width + 4095) // 4096
            )
            _segmented_selected[count,](
                k_nope,
                k_pe,
                v,
                k_address,
                v_address,
                meta,
                num_warps=4,
                enable_fp_fusion=False,
                groups_per_cluster=4096 // min(nope, rope, width),
            )
        else:
            _address_streams[min(total, 1073741823),](
                k_nope,
                k_pe,
                v,
                k_address,
                v_address,
                meta,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (k_out, v_out)


RAW185 = (
    "typedef unsigned short __attribute__((address_space(1))) gu1"
    "6;\n"
    "typedef unsigned char __attribute__((address_space(1))) gu8;"
    "\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)"
    "));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "union fvbits185 { f32v f; u32v u; };\n"
    "\n"
    "__attribute__((device,always_inline)) u32v opaque185(u32v va"
    "lue) {\n"
    '    __asm__("" : "+v"(value));\n'
    "    return value;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v negmask185(u32v v"
    "alue) {\n"
    "    return opaque185(0u - (opaque185(value) >> 31));\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v minimum185(u32v l"
    "eft, u32v right) {\n"
    "    u32v difference = left - right;\n"
    "    return right + (difference & negmask185(difference));\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v maximum185(u32v l"
    "eft, u32v right) {\n"
    "    u32v difference = left - right;\n"
    "    return left - (difference & negmask185(difference));\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode185(\n"
    "    u32v original, float scale, unsigned int scale_sign,\n"
    "    unsigned int max_bits, unsigned int min_exponent,\n"
    "    unsigned int exponent_factor, unsigned int magic_offset)"
    " {\n"
    "    fvbits185 input, product, magnitude, magic, rounded;\n"
    "    input.u = original;\n"
    "    product.f = input.f * scale;\n"
    "    u32v abs_bits = product.u & 0x7fffffffu;\n"
    "    fvbits185 absolute_value, magnitude_limit;\n"
    "    absolute_value.u = abs_bits;\n"
    "    magnitude_limit.u = (u32v)max_bits;\n"
    "    magnitude.f = __builtin_elementwise_min(absolute_value.f"
    ", magnitude_limit.f);\n"
    "    fvbits185 exponent_floor, bounded_exponent;\n"
    "    exponent_floor.u = (u32v)(min_exponent << 23);\n"
    "    bounded_exponent.f = __builtin_elementwise_max(magnitude"
    ".f, exponent_floor.f);\n"
    "    u32v exponent = bounded_exponent.u >> 23;\n"
    "    magic.u = exponent * exponent_factor + magic_offset;\n"
    "    rounded.f = magnitude.f + magic.f;\n"
    "    u32v nan_mask = negmask185((u32v)0x7f800000u - abs_bits)"
    ";\n"
    "    u32v code = rounded.u ^ ((rounded.u ^ 127u) & nan_mask);"
    "\n"
    "    u32v sign = ((original >> 24) ^ scale_sign) & 128u;\n"
    "    return (code | sign) & 255u;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode_pair185(\n"
    "    const unsigned int* input, float scale, unsigned int sca"
    "le_sign,\n"
    "    unsigned int max_bits, unsigned int min_exponent,\n"
    "    unsigned int exponent_factor, unsigned int magic_offset)"
    " {\n"
    "    u32v packed = *((const u32v*)input);\n"
    "    u32v low = encode185(packed << 16, scale, scale_sign,\n"
    "                        max_bits, min_exponent, exponent_fac"
    "tor, magic_offset);\n"
    "    u32v high = encode185(packed & 0xffff0000u, scale, scale"
    "_sign,\n"
    "                         max_bits, min_exponent, exponent_fa"
    "ctor, magic_offset);\n"
    "    return low | (high << 8);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v pack64_bytes185(u"
    "32v first, u32v second) {\n"
    "    u32v a1 = __builtin_xpu2_vmerge_l_u(first, second);\n"
    "    u32v b1 = __builtin_xpu2_vmerge_h_u(first, second);\n"
    "    u32v a2 = __builtin_xpu2_vmerge_l_u(a1, b1);\n"
    "    u32v b2 = __builtin_xpu2_vmerge_h_u(a1, b1);\n"
    "    u32v a3 = __builtin_xpu2_vmerge_l_u(a2, b2);\n"
    "    u32v b3 = __builtin_xpu2_vmerge_h_u(a2, b2);\n"
    "    u32v a4 = __builtin_xpu2_vmerge_l_u(a3, b3);\n"
    "    u32v b4 = __builtin_xpu2_vmerge_h_u(a3, b3);\n"
    "    return a4 | (b4 << 16);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) void encode_row185(\n"
    "    const unsigned int* input, unsigned int* output, int wid"
    "th,\n"
    "    float scale, unsigned int scale_sign, unsigned int max_b"
    "its,\n"
    "    unsigned int min_exponent, unsigned int exponent_factor,"
    "\n"
    "    unsigned int magic_offset) {\n"
    "    for (int column = 0; column < width; column += 64) {\n"
    "        u32v first = encode_pair185(input + column / 2, scal"
    "e, scale_sign,\n"
    "                                   max_bits, min_exponent, e"
    "xponent_factor, magic_offset);\n"
    "        u32v second = encode_pair185(input + column / 2 + 16"
    ", scale, scale_sign,\n"
    "                                    max_bits, min_exponent, "
    "exponent_factor, magic_offset);\n"
    "        *((u32v*)(output + column / 4)) = pack64_bytes185(fi"
    "rst, second);\n"
    "    }\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device185(\n'
    "    const gu16* K, const gu16* P, const gu16* V, gu8* O, gu8"
    "* W,\n"
    "    long long tokens, long long heads, int N, int PE, int D,"
    " long long PS,\n"
    "    float k_scale, float v_scale, int k_sign, int v_sign, in"
    "t E4,\n"
    "    int pid, int np, long long job_heads, long long head_job"
    "s) {\n"
    "    unsigned int workspace[1024] __attribute__((aligned(64))"
    ");\n"
    "    unsigned int position[256] __attribute__((aligned(64)));"
    "\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    unsigned int mantissa = E4 ? 3u : 2u;\n"
    "    unsigned int min_exponent = E4 ? 121u : 113u;\n"
    "    unsigned int max_bits = E4 ? 0x43e00000u : 0x47600000u;\n"
    "    unsigned int exponent_factor = (1u << 23) + (1u << manti"
    "ssa);\n"
    "    unsigned int magic_offset = ((23u - mantissa) << 23) - ("
    "min_exponent << mantissa);\n"
    "    for (long long job = pid * 64 + core; job < tokens * hea"
    "d_jobs; job += np * 64) {\n"
    "        long long token = job;\n"
    "        long long first_head = 0;\n"
    "        if (head_jobs > 1) {\n"
    "            token = job / head_jobs;\n"
    "            first_head = (job - token * head_jobs) * job_hea"
    "ds;\n"
    "        }\n"
    "        long long end_head = first_head + job_heads < heads "
    "? first_head + job_heads : heads;\n"
    "        long long first_row = token * heads + first_head;\n"
    "        int first_count = end_head - first_head < 4 ? (int)("
    "end_head - first_head) : 4;\n"
    "        __builtin_xpu_gm2lm_v3(position, P + token * PS, 0, "
    "PE * 2);\n"
    "        __builtin_xpu_gm2lm_v3(workspace, K + first_row * N,"
    " 0, first_count * N * 2);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        encode_row185(position, position, PE, k_scale, k_sig"
    "n,\n"
    "                      max_bits, min_exponent, exponent_facto"
    "r, magic_offset);\n"
    "        for (long long head = first_head; head < end_head; h"
    "ead += 4) {\n"
    "            int count = end_head - head < 4 ? (int)(end_head"
    " - head) : 4;\n"
    "            long long row = token * heads + head;\n"
    "            if (head != first_head) {\n"
    "                __builtin_xpu_gm2lm_v3(workspace, K + row * "
    "N, 0, count * N * 2);\n"
    "                __builtin_xpu2_mfence(7);\n"
    "            }\n"
    "            if (PE <= N) {\n"
    "                for (int local = 0; local < count; ++local) "
    "{\n"
    "                    encode_row185(workspace + local * N / 2,"
    "\n"
    "                                  workspace + local * (N + P"
    "E) / 4, N, k_scale, k_sign,\n"
    "                                  max_bits, min_exponent, ex"
    "ponent_factor, magic_offset);\n"
    "                }\n"
    "            } else {\n"
    "                encode_row185(workspace, workspace, count * "
    "N, k_scale, k_sign,\n"
    "                              max_bits, min_exponent, expone"
    "nt_factor, magic_offset);\n"
    "            }\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            for (int local = count - 1; local >= 0; --local)"
    " {\n"
    "                if (PE > N && local != 0) {\n"
    "                    for (int column = N - 64; column >= 0; c"
    "olumn -= 64) {\n"
    "                        u32v block = *((const u32v*)(workspa"
    "ce + (local * N + column) / 4));\n"
    "                        *((u32v*)(workspace + (local * (N + "
    "PE) + column) / 4)) = block;\n"
    "                    }\n"
    "                }\n"
    "                for (int column = 0; column < PE; column += "
    "64) {\n"
    "                    *((u32v*)(workspace + (local * (N + PE) "
    "+ N + column) / 4)) =\n"
    "                        *((const u32v*)(position + column / "
    "4));\n"
    "                }\n"
    "            }\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            __builtin_xpu_lm2gm_v3(O + row * (N + PE), works"
    "pace, 0, count * (N + PE));\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            __builtin_xpu_gm2lm_v3(workspace, V + row * D, 0"
    ", count * D * 2);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            encode_row185(workspace, workspace, count * D, v"
    "_scale, v_sign,\n"
    "                          max_bits, min_exponent, exponent_f"
    "actor, magic_offset);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            __builtin_xpu_lm2gm_v3(W + row * D, workspace, 0"
    ", count * D);\n"
    "            __builtin_xpu2_mfence(7);\n"
    "        }\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW185, flags=("-ffp-contract=off",))
def _device185(
    K,
    P,
    V,
    O,
    W,
    tokens,
    heads,
    N,
    PE,
    D,
    PS,
    k_scale,
    v_scale,
    k_sign,
    v_sign,
    E4,
    pid,
    np,
    job_heads,
    head_jobs,
):
    pass


@triton.jit(do_not_specialize=["O_ADDR", "W_ADDR"])
def _raw_kernel185(
    K,
    P,
    V,
    O_ADDR,
    W_ADDR,
    META: tl.constexpr,
    JOB_HEADS: tl.constexpr,
    HEAD_JOBS: tl.constexpr,
):
    output = tl.multiple_of(
        O_ADDR.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 64
    )
    values = tl.multiple_of(
        W_ADDR.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 64
    )
    tle.raw.call(
        _device185,
        (
            K,
            P,
            V,
            output,
            values,
            tl.full((), META[0] // META[1], tl.int64),
            tl.full((), META[1], tl.int64),
            tl.full((), META[2], tl.int32),
            tl.full((), META[3], tl.int32),
            tl.full((), META[4], tl.int32),
            tl.full((), META[5], tl.int64),
            tl.full((), META[6], tl.float32),
            tl.full((), META[7], tl.float32),
            tl.full((), META[8], tl.int32),
            tl.full((), META[9], tl.int32),
            tl.full((), META[10], tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
            tl.full((), JOB_HEADS, tl.int64),
            tl.full((), HEAD_JOBS, tl.int64),
        ),
    )


__all__ = ["mla_kv_pack_quantize_fp8"]
