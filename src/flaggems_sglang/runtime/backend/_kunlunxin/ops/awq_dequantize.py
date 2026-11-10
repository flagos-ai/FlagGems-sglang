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
def _pairs(
    Q,
    S,
    Z,
    Output,
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    FULL: tl.constexpr,
):
    r = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 >= 2147483648
        else tl.program_id(0)
    )
    p = (
        START
        + (
            tl.program_id(1).to(tl.int64)
            if K * W * 8 >= 2147483648
            else tl.program_id(1)
        )
        * B
        + tl.arange(0, B)
    )
    sh = (p & 3) * 4
    if FULL:
        q = tl.load(Q + r * W + p // 4)
        z = tl.load(Z + r // G * W + p // 4)
        sp = tl.load(S.to(tl.pointer_type(tl.uint32)) + r // G * W * 4 + p)
    else:
        q = tl.load(Q + r * W + p // 4, p < W * 4, 0)
        z = tl.load(Z + r // G * W + p // 4, p < W * 4, 0)
        sp = tl.load(
            S.to(tl.pointer_type(tl.uint32)) + r // G * W * 4 + p, p < W * 4, 0
        )
    s0 = (sp & 65535).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    s1 = (sp >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    v0 = (((q >> sh & 15) - (z >> sh & 15)).to(tl.float32) * s0).to(tl.float16)
    v1 = (((q >> sh + 16 & 15) - (z >> sh + 16 & 15)).to(tl.float32) * s1).to(
        tl.float16
    )
    out = (
        v0.to(tl.uint16, bitcast=True).to(tl.uint32)
        | v1.to(tl.uint16, bitcast=True).to(tl.uint32) << 16
    )
    if FULL:
        tl.store(Output.to(tl.pointer_type(tl.uint32)) + r * W * 4 + p, out)
    else:
        tl.store(
            Output.to(tl.pointer_type(tl.uint32)) + r * W * 4 + p,
            out,
            p < W * 4,
        )


@triton.jit
def _reuse(
    Q,
    S,
    Z,
    Output,
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    R: tl.constexpr,
):
    base_row = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 >= 2147483648
        else tl.program_id(0)
    ) * R
    rows = base_row + tl.arange(0, R)
    n = (
        START * 2
        + (
            tl.program_id(1).to(tl.int64)
            if K * W * 8 >= 2147483648
            else tl.program_id(1)
        )
        * B
        * 2
        + tl.arange(0, B * 2)
    )
    sh = (n & 1) * 16 + (n & 6) * 2
    q = tl.load(Q + rows[:, None] * W + n[None, :] // 8)
    z = tl.load(Z + base_row // G * W + n // 8)
    s = tl.load(S + base_row // G * W * 8 + n)
    d = (q >> sh[None, :] & 15) - (z >> sh & 15)[None, :]
    v = d.to(tl.float16) * s[None, :]
    tl.store(Output + rows[:, None] * W * 8 + n[None, :], v)


@triton.jit
def _reuse_decoded_zero(
    Q,
    S,
    Z,
    Output,
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    R: tl.constexpr,
):
    base_row = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 >= 2147483648
        else tl.program_id(0)
    ) * R
    rows = base_row + tl.arange(0, R)
    n = (
        START * 2
        + (
            tl.program_id(1).to(tl.int64)
            if K * W * 8 >= 2147483648
            else tl.program_id(1)
        )
        * B
        * 2
        + tl.arange(0, B * 2)
    )
    sh = (n & 1) * 16 + (n & 6) * 2
    q = tl.load(Q + rows[:, None] * W + n[None, :] // 8)
    z = tl.load(Z + base_row // G * W * 8 + n)
    s = tl.load(S + base_row // G * W * 8 + n)
    d = ((q >> sh[None, :] & 15).to(tl.float16) - z[None, :]).to(tl.float16)
    v = d.to(tl.float16) * s[None, :]
    tl.store(Output + rows[:, None] * W * 8 + n[None, :], v)


@triton.jit
def _expand_zero(Z, D, W: tl.constexpr, B: tl.constexpr):
    group = tl.program_id(0).to(tl.int64)
    n = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    sh = (n & 1) * 16 + (n & 6) * 2
    if W * 8 % B == 0:
        z = tl.load(Z + group * W + n // 8)
        tl.store(D + group * W * 8 + n, (z >> sh & 15).to(tl.float16))
    else:
        z = tl.load(Z + group * W + n // 8, n < W * 8, 0)
        tl.store(
            D + group * W * 8 + n, (z >> sh & 15).to(tl.float16), n < W * 8
        )


@triton.jit
def _column_word_reuse(
    Q,
    S,
    Z,
    Output,
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    R: tl.constexpr,
):
    words = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 >= 2147483648
        else tl.program_id(0)
    ) * 64 + tl.arange(0, 64)
    rows = (
        tl.program_id(1).to(tl.int64)
        if K * W * 8 >= 2147483648
        else tl.program_id(1)
    ) * R + tl.arange(0, R)
    q = tl.load(Q + words[:, None] * K + rows[None, :])
    z = tl.load(Z + rows[None, :] // G * W + words[:, None])
    s0 = tl.load(S + rows[None, :] // G * W * 8 + words[:, None] * 8 + 0)
    s1 = tl.load(S + rows[None, :] // G * W * 8 + words[:, None] * 8 + 1)
    s2 = tl.load(S + rows[None, :] // G * W * 8 + words[:, None] * 8 + 2)
    s3 = tl.load(S + rows[None, :] // G * W * 8 + words[:, None] * 8 + 3)
    s4 = tl.load(S + rows[None, :] // G * W * 8 + words[:, None] * 8 + 4)
    s5 = tl.load(S + rows[None, :] // G * W * 8 + words[:, None] * 8 + 5)
    s6 = tl.load(S + rows[None, :] // G * W * 8 + words[:, None] * 8 + 6)
    s7 = tl.load(S + rows[None, :] // G * W * 8 + words[:, None] * 8 + 7)
    d0 = (q >> 0 & 15) - (z >> 0 & 15)
    v0 = d0.to(tl.float16) * s0
    tl.store(Output + (words[:, None] * 8 + 0) * K + rows[None, :], v0)
    d1 = (q >> 16 & 15) - (z >> 16 & 15)
    v1 = d1.to(tl.float16) * s1
    tl.store(Output + (words[:, None] * 8 + 1) * K + rows[None, :], v1)
    d2 = (q >> 4 & 15) - (z >> 4 & 15)
    v2 = d2.to(tl.float16) * s2
    tl.store(Output + (words[:, None] * 8 + 2) * K + rows[None, :], v2)
    d3 = (q >> 20 & 15) - (z >> 20 & 15)
    v3 = d3.to(tl.float16) * s3
    tl.store(Output + (words[:, None] * 8 + 3) * K + rows[None, :], v3)
    d4 = (q >> 8 & 15) - (z >> 8 & 15)
    v4 = d4.to(tl.float16) * s4
    tl.store(Output + (words[:, None] * 8 + 4) * K + rows[None, :], v4)
    d5 = (q >> 24 & 15) - (z >> 24 & 15)
    v5 = d5.to(tl.float16) * s5
    tl.store(Output + (words[:, None] * 8 + 5) * K + rows[None, :], v5)
    d6 = (q >> 12 & 15) - (z >> 12 & 15)
    v6 = d6.to(tl.float16) * s6
    tl.store(Output + (words[:, None] * 8 + 6) * K + rows[None, :], v6)
    d7 = (q >> 28 & 15) - (z >> 28 & 15)
    v7 = d7.to(tl.float16) * s7
    tl.store(Output + (words[:, None] * 8 + 7) * K + rows[None, :], v7)


@triton.jit
def _column_major(
    Q,
    S,
    Z,
    Output,
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    R: tl.constexpr,
):
    n = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 >= 2147483648
        else tl.program_id(0)
    ) * 64 + tl.arange(0, 64)
    base_row = (
        tl.program_id(1).to(tl.int64)
        if K * W * 8 >= 2147483648
        else tl.program_id(1)
    ) * R
    rows = base_row + tl.arange(0, R)
    sh = (n & 1) * 16 + (n & 6) * 2
    q = tl.load(Q + n[:, None] // 8 * K + rows[None, :])
    z = tl.load(Z + rows[None, :] // G * W + n[:, None] // 8)
    s = tl.load(S + rows[None, :] // G * W * 8 + n[:, None])
    d = (q >> sh[:, None] & 15) - (z >> sh[:, None] & 15)
    v = d.to(tl.float16) * s
    tl.store(Output + n[:, None] * K + rows[None, :], v)


RAW161 = (
    "\n"
    "typedef unsigned int __attribute__((address_space(1))) gu32;"
    "\n"
    "typedef __fp16 __attribute__((address_space(1))) gf16;\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)"
    "));\n"
    "typedef int i32v __attribute__((ext_vector_type(16)));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned short h32v __attribute__((ext_vector_type(3"
    "2)));\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_column161(\n'
    "    gu32* Q, gf16* S, gu32* Z, gf16* O,\n"
    "    int K, int W, int G, int pid, int np) {\n"
    "    unsigned int q_buffer[128] __attribute__((aligned(64)));"
    "\n"
    "    unsigned int z_buffer[256] __attribute__((aligned(64)));"
    "\n"
    "    unsigned int scale_rows[256] __attribute__((aligned(64))"
    ");\n"
    "    __fp16 output_buffer[1024] __attribute__((aligned(64)));"
    "\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    int blocks = W / 64;\n"
    "    int cores = np * 64;\n"
    "    int jobs32 = blocks * (K / 32);\n"
    "    int max_rows = ((jobs32 + cores - 1) / cores) * 32;\n"
    "    int rows_per_job = 32;\n"
    "    int jobs64 = blocks * (K / 64);\n"
    "    int max_rows64 = ((jobs64 + cores - 1) / cores) * 64;\n"
    "    if (max_rows64 == max_rows) rows_per_job = 64;\n"
    "    int jobs128 = blocks * (K / 128);\n"
    "    int max_rows128 = ((jobs128 + cores - 1) / cores) * 128;"
    "\n"
    "    if (max_rows128 == max_rows) rows_per_job = 128;\n"
    "    int jobs16 = blocks * (K / 16);\n"
    "    int max_rows16 = ((jobs16 + cores - 1) / cores) * 16;\n"
    "    if (max_rows16 * 16 <= max_rows * 15) rows_per_job = 16;"
    "\n"
    "    int jobs = blocks * (K / rows_per_job);\n"
    "    for (int job = pid * 64 + core; job < jobs; job += np * "
    "64) {\n"
    "        int word = (job % blocks) * 64;\n"
    "        int first_row = (job / blocks) * rows_per_job;\n"
    "        int group = first_row / G;\n"
    "        __builtin_xpu_gm2lm_v3(z_buffer, Z + (long long)grou"
    "p * W + word, 0, 256);\n"
    "        __builtin_xpu_gm2lm_v3(scale_rows, S + ((long long)g"
    "roup * W + word) * 8, 0, 1024);\n"
    "        __builtin_xpu_gm2lm_v3(q_buffer, Q + (long long)firs"
    "t_row * W + word, 0, 256);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        for (int zero_offset = 0; zero_offset < 64; zero_off"
    "set += 16) {\n"
    "            u32v packed_zero = *((u32v*)(z_buffer + zero_off"
    "set));\n"
    "            *((u32v*)(z_buffer + 0 * 64 + zero_offset)) = (("
    "packed_zero >> 0) & 0x000f000fu) | 0x64006400u;\n"
    "            *((u32v*)(z_buffer + 1 * 64 + zero_offset)) = (("
    "packed_zero >> 4) & 0x000f000fu) | 0x64006400u;\n"
    "            *((u32v*)(z_buffer + 2 * 64 + zero_offset)) = (("
    "packed_zero >> 8) & 0x000f000fu) | 0x64006400u;\n"
    "            *((u32v*)(z_buffer + 3 * 64 + zero_offset)) = (("
    "packed_zero >> 12) & 0x000f000fu) | 0x64006400u;\n"
    "        }\n"
    "        for (int local_row = 0; local_row < rows_per_job; ++"
    "local_row) {\n"
    "            int slot = local_row & 1;\n"
    "            unsigned int* qb = q_buffer + slot * 64;\n"
    "            __fp16* out = output_buffer + slot * 512;\n"
    "            int row = first_row + local_row;\n"
    "            if (local_row + 1 < rows_per_job) {\n"
    "                __builtin_xpu_gm2lm_v3(q_buffer + (slot ^ 1)"
    " * 64,\n"
    "                                      Q + ((long long)row + "
    "1) * W + word, 0, 256);\n"
    "            }\n"
    "            for (int offset = 0; offset < 64; offset += 16) "
    "{\n"
    "                u32v q = *((u32v*)(qb + offset));\n"
    "                u32v packed0 = __builtin_xpu2_vvor_u_mr1((q "
    ">> 0) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                h32v d0 = __builtin_xpu3_vvsub_hf_mr1_rn((h3"
    "2v)packed0, *((h32v*)(z_buffer + 0 * 64 + offset)));\n"
    "                u32v h0 = (u32v)d0;\n"
    "                u32v packed1 = __builtin_xpu2_vvor_u_mr1((q "
    ">> 4) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                h32v d1 = __builtin_xpu3_vvsub_hf_mr1_rn((h3"
    "2v)packed1, *((h32v*)(z_buffer + 1 * 64 + offset)));\n"
    "                u32v h1 = (u32v)d1;\n"
    "                u32v packed2 = __builtin_xpu2_vvor_u_mr1((q "
    ">> 8) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                h32v d2 = __builtin_xpu3_vvsub_hf_mr1_rn((h3"
    "2v)packed2, *((h32v*)(z_buffer + 2 * 64 + offset)));\n"
    "                u32v h2 = (u32v)d2;\n"
    "                u32v packed3 = __builtin_xpu2_vvor_u_mr1((q "
    ">> 12) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                h32v d3 = __builtin_xpu3_vvsub_hf_mr1_rn((h3"
    "2v)packed3, *((h32v*)(z_buffer + 3 * 64 + offset)));\n"
    "                u32v h3 = (u32v)d3;\n"
    "                u32v a0 = __builtin_xpu2_vmerge_l_u(h0, h2);"
    "\n"
    "                u32v a1 = __builtin_xpu2_vmerge_h_u(h0, h2);"
    "\n"
    "                u32v a2 = __builtin_xpu2_vmerge_l_u(h1, h3);"
    "\n"
    "                u32v a3 = __builtin_xpu2_vmerge_h_u(h1, h3);"
    "\n"
    "                u32v o0 = __builtin_xpu2_vmerge_l_u(a0, a2);"
    "\n"
    "                u32v o1 = __builtin_xpu2_vmerge_h_u(a0, a2);"
    "\n"
    "                u32v o2 = __builtin_xpu2_vmerge_l_u(a1, a3);"
    "\n"
    "                u32v o3 = __builtin_xpu2_vmerge_h_u(a1, a3);"
    "\n"
    "                *((h32v*)(out + offset * 8 + 0)) = __builtin"
    "_xpu3_vvmul_hf_mr1_rn((h32v)o0, *((h32v*)(scale_rows + offse"
    "t * 4 + 0)));\n"
    "                *((h32v*)(out + offset * 8 + 32)) = __builti"
    "n_xpu3_vvmul_hf_mr1_rn((h32v)o1, *((h32v*)(scale_rows + offs"
    "et * 4 + 16)));\n"
    "                *((h32v*)(out + offset * 8 + 64)) = __builti"
    "n_xpu3_vvmul_hf_mr1_rn((h32v)o2, *((h32v*)(scale_rows + offs"
    "et * 4 + 32)));\n"
    "                *((h32v*)(out + offset * 8 + 96)) = __builti"
    "n_xpu3_vvmul_hf_mr1_rn((h32v)o3, *((h32v*)(scale_rows + offs"
    "et * 4 + 48)));\n"
    "            }\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            __builtin_xpu_lm2gm_v3(O + (long long)row * W * "
    "8 + (long long)word * 8, out, 0, 1024);\n"
    "        }\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW161, flags=("-ffp-contract=off",))
def _device_column161(Q, S, Z, Output, K, W, G, pid, np):
    pass


@triton.jit
def _raw_column_entry161(
    Q, S, Z, Output, K: tl.constexpr, W: tl.constexpr, G: tl.constexpr
):
    tle.raw.call(
        _device_column161,
        (
            Q,
            S,
            Z,
            Output,
            tl.full((), K, tl.int32),
            tl.full((), W, tl.int32),
            tl.full((), G, tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


RAW161_WIDE = (
    "typedef unsigned int __attribute__((address_space(1))) gu32;"
    "\n"
    "typedef __fp16 __attribute__((address_space(1))) gf16;\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)"
    "));\n"
    "typedef int i32v __attribute__((ext_vector_type(16)));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned short h32v __attribute__((ext_vector_type(3"
    "2)));\n"
    "\n"
    'extern "C" __attribute__((device)) void _device_column161_wi'
    "de(\n"
    "    gu32* Q, gf16* S, gu32* Z, gf16* O,\n"
    "    int K, int W, int G, int pid, int np) {\n"
    "    unsigned int q_buffer[256] __attribute__((aligned(64)));"
    "\n"
    "    unsigned int z_buffer[128] __attribute__((aligned(64)));"
    "\n"
    "    unsigned int scale_rows[512] __attribute__((aligned(64))"
    ");\n"
    "    __fp16 output_buffer[2048] __attribute__((aligned(64)));"
    "\n"
    "    int core = __builtin_xpu_core_id();\n"
    "    int blocks = W / 128;\n"
    "    int jobs = blocks * (K / 32);\n"
    "    for (int job = pid * 64 + core; job < jobs; job += np * "
    "64) {\n"
    "        int word = (job % blocks) * 128;\n"
    "        int first_row = (job / blocks) * 32;\n"
    "        int group = first_row / G;\n"
    "        __builtin_xpu_gm2lm_v3(z_buffer, Z + (long long)grou"
    "p * W + word, 0, 512);\n"
    "        __builtin_xpu_gm2lm_v3(scale_rows, S + ((long long)g"
    "roup * W + word) * 8, 0, 2048);\n"
    "        __builtin_xpu_gm2lm_v3(q_buffer, Q + (long long)firs"
    "t_row * W + word, 0, 512);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        for (int local_row = 0; local_row < 32; ++local_row)"
    " {\n"
    "            int slot = local_row & 1;\n"
    "            unsigned int* qb = q_buffer + slot * 128;\n"
    "            __fp16* out = output_buffer + slot * 1024;\n"
    "            int row = first_row + local_row;\n"
    "            if (local_row + 1 < 32) {\n"
    "                __builtin_xpu_gm2lm_v3(q_buffer + (slot ^ 1)"
    " * 128,\n"
    "                                      Q + ((long long)row + "
    "1) * W + word, 0, 512);\n"
    "            }\n"
    "            for (int offset = 0; offset < 128; offset += 16)"
    " {\n"
    "                u32v q = *((u32v*)(qb + offset));\n"
    "                u32v z = *((u32v*)(z_buffer + offset));\n"
    "                u32v packed0 = __builtin_xpu2_vvor_u_mr1((q "
    ">> 0) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                u32v zpacked0 = __builtin_xpu2_vvor_u_mr1((z"
    " >> 0) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                h32v d0 = __builtin_xpu3_vvsub_hf_mr1_rn((h3"
    "2v)packed0, (h32v)zpacked0);\n"
    "                u32v h0 = (u32v)d0;\n"
    "                u32v packed1 = __builtin_xpu2_vvor_u_mr1((q "
    ">> 4) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                u32v zpacked1 = __builtin_xpu2_vvor_u_mr1((z"
    " >> 4) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                h32v d1 = __builtin_xpu3_vvsub_hf_mr1_rn((h3"
    "2v)packed1, (h32v)zpacked1);\n"
    "                u32v h1 = (u32v)d1;\n"
    "                u32v packed2 = __builtin_xpu2_vvor_u_mr1((q "
    ">> 8) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                u32v zpacked2 = __builtin_xpu2_vvor_u_mr1((z"
    " >> 8) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                h32v d2 = __builtin_xpu3_vvsub_hf_mr1_rn((h3"
    "2v)packed2, (h32v)zpacked2);\n"
    "                u32v h2 = (u32v)d2;\n"
    "                u32v packed3 = __builtin_xpu2_vvor_u_mr1((q "
    ">> 12) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                u32v zpacked3 = __builtin_xpu2_vvor_u_mr1((z"
    " >> 12) & 0x000f000fu, (u32v)0x64006400u);\n"
    "                h32v d3 = __builtin_xpu3_vvsub_hf_mr1_rn((h3"
    "2v)packed3, (h32v)zpacked3);\n"
    "                u32v h3 = (u32v)d3;\n"
    "                u32v a0 = __builtin_xpu2_vmerge_l_u(h0, h2);"
    "\n"
    "                u32v a1 = __builtin_xpu2_vmerge_h_u(h0, h2);"
    "\n"
    "                u32v a2 = __builtin_xpu2_vmerge_l_u(h1, h3);"
    "\n"
    "                u32v a3 = __builtin_xpu2_vmerge_h_u(h1, h3);"
    "\n"
    "                u32v o0 = __builtin_xpu2_vmerge_l_u(a0, a2);"
    "\n"
    "                u32v o1 = __builtin_xpu2_vmerge_h_u(a0, a2);"
    "\n"
    "                u32v o2 = __builtin_xpu2_vmerge_l_u(a1, a3);"
    "\n"
    "                u32v o3 = __builtin_xpu2_vmerge_h_u(a1, a3);"
    "\n"
    "                *((h32v*)(out + offset * 8 + 0)) = __builtin"
    "_xpu3_vvmul_hf_mr1_rn((h32v)o0, *((h32v*)(scale_rows + offse"
    "t * 4 + 0)));\n"
    "                *((h32v*)(out + offset * 8 + 32)) = __builti"
    "n_xpu3_vvmul_hf_mr1_rn((h32v)o1, *((h32v*)(scale_rows + offs"
    "et * 4 + 16)));\n"
    "                *((h32v*)(out + offset * 8 + 64)) = __builti"
    "n_xpu3_vvmul_hf_mr1_rn((h32v)o2, *((h32v*)(scale_rows + offs"
    "et * 4 + 32)));\n"
    "                *((h32v*)(out + offset * 8 + 96)) = __builti"
    "n_xpu3_vvmul_hf_mr1_rn((h32v)o3, *((h32v*)(scale_rows + offs"
    "et * 4 + 48)));\n"
    "            }\n"
    "            __builtin_xpu2_mfence(7);\n"
    "            __builtin_xpu_lm2gm_v3(O + (long long)row * W * "
    "8 + (long long)word * 8, out, 0, 2048);\n"
    "        }\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW161_WIDE, flags=("-ffp-contract=off",))
def _device_column161_wide(Q, S, Z, Output, K, W, G, pid, np):
    pass


@triton.jit
def _raw_column_entry161_wide(
    Q, S, Z, Output, K: tl.constexpr, W: tl.constexpr, G: tl.constexpr
):
    tle.raw.call(
        _device_column161_wide,
        (
            Q,
            S,
            Z,
            Output,
            tl.full((), K, tl.int32),
            tl.full((), W, tl.int32),
            tl.full((), G, tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


def awq_dequantize(qweight, scales, zeros):
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    zeros = zeros.contiguous()
    if scales.data_ptr() % 4:
        scales = scales.clone()
    k, w = qweight.shape
    g = k // scales.shape[0]
    if (
        k % 128 == 0
        and g % 128 == 0
        and (w % 64 == 0)
        and (k * w * 8 >= 2097152)
    ):
        row_major_out = scales.new_empty((k, w * 8))
        if w % 128 == 0:
            _raw_column_entry161_wide.run(
                qweight,
                scales,
                zeros,
                row_major_out,
                k,
                w,
                g,
                grid=(12,),
                warmup=False,
            )
        else:
            _raw_column_entry161.run(
                qweight,
                scales,
                zeros,
                row_major_out,
                k,
                w,
                g,
                grid=(12,),
                warmup=False,
            )
        return row_major_out
    if (
        k % 128 == 0
        and g % 128 == 0
        and (w % 8 == 0)
        and (k * w * 8 >= (2097152 if w % 64 == 0 else 16777216))
    ):
        transposed_q = qweight.t().contiguous()
        transposed_out = scales.new_empty((w * 8, k))
        if w % 64 == 0:
            _raw_column_entry161.run(
                transposed_q,
                scales,
                zeros,
                transposed_out,
                k,
                w,
                g,
                grid=(12,),
                warmup=False,
            )
        else:
            _column_major.run(
                transposed_q,
                scales,
                zeros,
                transposed_out,
                k,
                w,
                g,
                128,
                grid=(w * 8 // 64, k // 128),
                warmup=False,
                groups_per_cluster=64,
                isCloseMemoryAsync=False,
                unroll_num=16,
            )
        return transposed_out.t()
    rows = (
        64
        if k % 64 == 0 and g % 64 == 0
        else 32 if k % 32 == 0 and g % 32 == 0 else 1
    )
    out = scales.new_empty((k, w * 8))
    start = 0
    remain = w * 4
    if remain < 4096 and (not (rows > 1 and remain >= 64)):
        size = max(
            64, (0 if (remain) == 0 else 1 << ((remain) - 1).bit_length())
        )
        _pairs.run(
            qweight,
            scales,
            zeros,
            out,
            k,
            w,
            g,
            size,
            0,
            size == remain,
            grid=(k, 1),
            warmup=False,
        )
        return out
    use_predecode = rows > 1 and k * w * 8 >= 16777216
    if use_predecode:
        decoded_zeros = torch.empty(
            (k // g, w * 8), dtype=torch.float16, device=qweight.device
        )
        zero_block = min(
            2048,
            max(64, (0 if (w * 8) == 0 else 1 << ((w * 8) - 1).bit_length())),
        )
        _expand_zero.run(
            zeros,
            decoded_zeros,
            w,
            zero_block,
            grid=(k // g, (((w * 8) + (zero_block) - 1) // (zero_block))),
            warmup=False,
        )
    while remain >= 64:
        size = min(256 if rows > 1 else 16384, 1 << remain.bit_length() - 1)
        count = remain // size
        if rows > 1:
            if use_predecode:
                _reuse_decoded_zero.run(
                    qweight,
                    scales,
                    decoded_zeros,
                    out,
                    k,
                    w,
                    g,
                    size,
                    start,
                    rows,
                    grid=(k // rows, count),
                    warmup=False,
                    groups_per_cluster=rows,
                    isCloseMemoryAsync=False,
                )
            else:
                _reuse.run(
                    qweight,
                    scales,
                    zeros,
                    out,
                    k,
                    w,
                    g,
                    size,
                    start,
                    rows,
                    grid=(k // rows, count),
                    warmup=False,
                    groups_per_cluster=rows,
                    isCloseMemoryAsync=False,
                )
        else:
            _pairs.run(
                qweight,
                scales,
                zeros,
                out,
                k,
                w,
                g,
                size,
                start,
                True,
                grid=(k, count),
                warmup=False,
            )
        start += size * count
        remain -= size * count
    if remain:
        _pairs.run(
            qweight,
            scales,
            zeros,
            out,
            k,
            w,
            g,
            64,
            start,
            False,
            grid=(k, 1),
            warmup=False,
        )
    return out


__all__ = ["awq_dequantize"]
