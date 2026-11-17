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
import triton.experimental.tle as tle
import triton.language as tl


@tr.jit
def _encode(v):
    bits = v.to(tl.uint32, bitcast=True)
    mag = bits & 2147483647
    normal = (mag + 67633151 + (mag >> 20 & 1) >> 20) - 1024
    subnormal = (tl.abs(v) + 16384.0).to(tl.int32, bitcast=True) - 1182793728
    code = tl.minimum(tl.where(mag < 1015021568, subnormal, normal), 126)
    code = tl.where(v != v, 127, code)
    return (code | bits >> 24 & 128).to(tl.uint8)


@tr.jit
def _encode_short(v):
    bits = v.to(tl.uint32, bitcast=True)
    mag = bits & 2147483647
    normal = mag + 67633151 + (mag >> 20 & 1) >> 20
    subnormal = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True)
    code = tl.where(mag < 1015021568, subnormal, tl.minimum(normal, 1150))
    code = tl.where(v != v, 127, code)
    return (code | bits >> 24 & 128).to(tl.uint8)


@tr.jit
def _quant(x, s, SHORT: tl.constexpr):
    if SHORT:
        return _encode_short(x / s)
    else:
        return _encode(x / s)


@tr.jit
def _pack_addr(
    X, R, output_ptr, M: tl.constexpr, BR: tl.constexpr, P: tl.constexpr
):
    X = X.to(tl.pointer_type(tl.bfloat16))
    R = R.to(tl.pointer_type(tl.bfloat16))
    output_ptr = output_ptr.to(tl.pointer_type(tl.uint8))
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
    FAST: tl.constexpr = (
        N % BR == 0
        and BR == 16
        and (P == 4)
        and (XS == D)
        and (NG == 4)
        and (G == 128)
        and (DR == 64)
        and (RS == DR)
        and (RC == 1)
    )
    groups = tl.program_id(0) * BR * NG + tl.arange(0, BG)
    rows = groups // NG
    tiles = groups % NG
    c = tl.arange(0, BC)
    mask = (
        (groups[:, None] < N * NG)
        & (c[None, :] < G // P)
        & (tl.arange(0, BG)[:, None] < BR * NG)
    )
    if FAST:
        rc = tl.arange(0, DR // NG)
        rope = tl.load(
            R.to(tl.pointer_type(tl.int16))
            + rows[:, None] * RS
            + tiles[:, None] * (DR // NG)
            + rc[None, :],
            offset_state_policy="continuous",
        )
        tl.store(
            (output_ptr + END).to(tl.pointer_type(tl.int16))
            + rows[:, None] * (WIDTH // 2)
            + tiles[:, None] * (DR // NG)
            + rc[None, :],
            rope,
            offset_state_policy="continuous",
        )
    if P == 4:
        offsets = (
            rows[:, None] * (XS // 4) + tiles[:, None] * (G // 4) + c[None, :]
        )
        if N % BR == 0 and BR == 16 and (NG == 4) and (G == 128) and (XS == D):
            raw = tl.load(
                X.to(tl.pointer_type(tl.uint64))
                + groups[:, None] * (G // 4)
                + c[None, :],
                offset_state_policy="continuous",
            )
        else:
            raw = tl.load(X.to(tl.pointer_type(tl.uint64)) + offsets, mask, 0)
        lo = raw.to(tl.int32)
        hi = (raw >> 32).to(tl.int32)
        a = (lo << 16).to(tl.float32, bitcast=True)
        b = (lo & -65536).to(tl.float32, bitcast=True)
        e = (hi << 16).to(tl.float32, bitcast=True)
        f = (hi & -65536).to(tl.float32, bitcast=True)
        mag = tl.maximum(
            a.to(tl.int32, bitcast=True) & 2147483647,
            b.to(tl.int32, bitcast=True) & 2147483647,
        )
        mag = tl.maximum(
            mag,
            tl.maximum(
                e.to(tl.int32, bitcast=True) & 2147483647,
                f.to(tl.int32, bitcast=True) & 2147483647,
            ),
        )
        scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
        word = (
            _quant(a, scale[:, None], FAST).to(tl.uint32)
            | _quant(b, scale[:, None], FAST).to(tl.uint32) << 8
        )
        word = (
            word
            | _quant(e, scale[:, None], FAST).to(tl.uint32) << 16
            | _quant(f, scale[:, None], FAST).to(tl.uint32) << 24
        )
        if N % BR == 0 and BR == 16 and (NG == 4) and (G == 128):
            tl.store(
                output_ptr.to(tl.pointer_type(tl.uint32))
                + rows[:, None] * (WIDTH // 4)
                + tiles[:, None] * (G // 4)
                + c[None, :],
                word,
                offset_state_policy="continuous",
            )
        else:
            tl.store(
                output_ptr.to(tl.pointer_type(tl.uint32))
                + rows[:, None] * (WIDTH // 4)
                + tiles[:, None] * (G // 4)
                + c[None, :],
                word,
                mask,
            )
    else:
        x = tl.load(
            X + rows[:, None] * XS + (tiles[:, None] * G + c[None, :]) * XC,
            mask,
            0,
        ).to(tl.float32)
        mag = x.to(tl.int32, bitcast=True) & 2147483647
        scale = tl.max(mag, 1).to(tl.float32, bitcast=True) / 448.0
        tl.store(
            output_ptr
            + rows[:, None] * WIDTH
            + tiles[:, None] * G
            + c[None, :],
            _quant(x, scale[:, None], FAST),
            mask,
        )
    if FAST:
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rows * (WIDTH // 4)
            + tiles,
            scale,
        )
    else:
        tl.store(
            (output_ptr + D).to(tl.pointer_type(tl.float32))
            + rows * (WIDTH // 4)
            + tiles,
            scale,
            (groups < N * NG) & (tl.arange(0, BG) < BR * NG),
        )
    if not FAST:
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


def _baseline(k_nope, k_rope, group_size=128):
    n, d = k_nope.shape
    dr = k_rope.shape[1]
    width = d + d // group_size * 4 + dr * 2
    output = k_nope.new_empty((n, width), dtype=torch.float8_e4m3fn)
    if n:
        br = 4 if n < 64 else 16
        p = (
            4
            if k_nope.stride(1) == 1
            and k_nope.stride(0) % 4 == 0
            and (group_size % 4 == 0)
            and (width % 4 == 0)
            and (k_nope.data_ptr() % 8 == 0)
            else 1
        )
        meta = (n, d, dr, group_size, *k_nope.stride(), *k_rope.stride())
        _pack_addr.run(
            k_nope.data_ptr(),
            k_rope.data_ptr(),
            output.data_ptr(),
            meta,
            br,
            p,
            grid=(tr.cdiv(n, br),),
            warmup=False,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
            groups_per_cluster=64,
        )
    return output


def quantize_k_cache_fast(k_nope, k_rope, group_size=128):
    n, d = k_nope.shape
    if not (
        n > 0
        and n < 1048576
        and (d == 512)
        and (group_size == 128)
        and (tuple(k_nope.stride()) == (512, 1))
        and (tuple(k_rope.shape) == (n, 64))
        and (tuple(k_rope.stride()) == (64, 1))
        and (k_nope.data_ptr() % 64 == 0)
        and (k_rope.data_ptr() % 64 == 0)
    ):
        return _baseline(k_nope, k_rope, group_size)
    output = k_nope.new_empty((n, 656), dtype=torch.float8_e4m3fn)
    _raw137_2.run(
        k_nope.data_ptr(),
        k_rope.data_ptr(),
        output.data_ptr(),
        n,
        grid=(min(tr.cdiv(n, 64), 12),),
        warmup=False,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return output


RAW137_2 = (
    "typedef unsigned short __attribute__((address_space(1))) gu1"
    "6;\n"
    "typedef unsigned char __attribute__((address_space(1))) gu8;"
    "\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)"
    "));\n"
    "typedef unsigned short u16v __attribute__((ext_vector_type(1"
    "6)));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "union vb { u32v u; f32v f; };\n"
    "union sb { unsigned int u; float f; };\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode123(f32v va"
    "lue) {\n"
    "    vb bits, rounded;\n"
    "    bits.f = value;\n"
    "    u32v original = bits.u & 0x7fffffffu;\n"
    "    u32v mag = original;\n"
    "    u32v normal_mag = __builtin_elementwise_max(mag, (u32v)0"
    "x3c800000u);\n"
    "    u32v normal = ((normal_mag + 0x7ffffu + ((normal_mag >> "
    "20) & 1u)) >> 20) - 960u;\n"
    "    rounded.u = mag;\n"
    "    rounded.f = rounded.f + 16384.0f;\n"
    "    u32v sub = rounded.u - 1182793728u;\n"
    "    u32v code = __builtin_elementwise_min(normal, sub);\n"
    "    u32v diff = 0x7f800000u - original;\n"
    '    __asm__("" : "+v"(diff));\n'
    "    u32v nanmask = 0u - (diff >> 31);\n"
    '    __asm__("" : "+v"(nanmask));\n'
    "    return ((code & 127u) | (nanmask & 127u)) | ((bits.u >> "
    "24) & 128u);\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _device137_2(gu16* X'
    ", gu16* R, gu8* O, int N, int B, int pid, int np) {\n"
    "    unsigned int input[256] __attribute__((aligned(64)));\n"
    "    unsigned int rope[32] __attribute__((aligned(64)));\n"
    "    unsigned int output[192] __attribute__((aligned(64)));\n"
    "    unsigned int packing[256] __attribute__((aligned(64)));\n"
    "    float division[32] __attribute__((aligned(64)));\n"
    "    unsigned int maxima[16] __attribute__((aligned(64)));\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    for (int first = (pid * 64 + core) * B; first < N; first"
    " += np * 64 * B) {\n"
    "        int count = N - first < B ? N - first : B;\n"
    "        __builtin_xpu_gm2lm_v3(input, X + first * 512, 0, co"
    "unt * 1024);\n"
    "        __builtin_xpu_gm2lm_v3(rope, R + first * 64, 0, coun"
    "t * 128);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        for (int row = 0; row < count; ++row) {\n"
    "            unsigned char* target = (unsigned char*)(output "
    "+ row * 192);\n"
    "            for (int group = 0; group < 4; ++group) {\n"
    "                u32v acc = (u32v)0u;\n"
    "                for (int offset = 0; offset < 64; offset += "
    "16) {\n"
    "                    u32v packed = *((u32v*)(input + row * 25"
    "6 + group * 64 + offset));\n"
    "                    u32v a = (packed << 16) & 0x7fffffffu;\n"
    "                    u32v b = packed & 0x7fff0000u;\n"
    "                    acc = __builtin_elementwise_max(a, acc);"
    "\n"
    "                    acc = __builtin_elementwise_max(b, acc);"
    "\n"
    "                }\n"
    "                *((u32v*)maxima) = acc;\n"
    "                __builtin_xpu2_mfence(7);\n"
    "                unsigned int maximum = 0;\n"
    "                __attribute__((opencl_unroll_hint(16))) for "
    "(int lane = 0; lane < 16; ++lane) maximum = maxima[lane] > m"
    "aximum ? maxima[lane] : maximum;\n"
    "                sb scale;\n"
    "                scale.u = maximum;\n"
    "                scale.f = scale.f / 448.0f;\n"
    "                *((float*)(target + 512 + group * 4)) = scal"
    "e.f;\n"
    "                for (int offset = 0; offset < 64; offset += "
    "16) {\n"
    "                    u32v packed = *((u32v*)(input + row * 25"
    "6 + group * 64 + offset));\n"
    "                    vb a, b;\n"
    "                    a.u = packed << 16;\n"
    "                    b.u = packed & 0xffff0000u;\n"
    "                    u32v qa = encode123(a.f / (f32v)scale.f)"
    ";\n"
    "                    u32v qb = encode123(b.f / (f32v)scale.f)"
    ";\n"
    "                    *((u32v*)(packing + group * 64 + offset)"
    ") = qa | (qb << 8);\n"
    "                }\n"
    "            }\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            __attribute__((opencl_unroll_hint(8))) for (int "
    "word = 0; word < 128; ++word) ((unsigned int*)target)[word] "
    "= packing[word * 2] | (packing[word * 2 + 1] << 16);\n"
    "            __attribute__((opencl_unroll_hint(8))) for (int "
    "word = 0; word < 32; ++word) ((unsigned int*)(target + 528))"
    "[word] = rope[row * 32 + word];\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            __builtin_xpu_lm2gm_v3(O + (first + row) * 656, "
    "target, 0, 656);\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW137_2, flags=("-ffp-contract=off",))
def _device137_2(X, R, output_ptr, N, B, pid, np):
    pass


@tr.jit
def _raw137_2(X, R, output_ptr, N: tl.constexpr):
    tle.raw.call(
        _device137_2,
        (
            X.to(tl.pointer_type(tl.bfloat16)),
            R.to(tl.pointer_type(tl.bfloat16)),
            output_ptr.to(tl.pointer_type(tl.uint8)),
            tl.full((), N, tl.int32),
            tl.full((), 1, tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


__all__ = ["quantize_k_cache_fast"]
