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
def _encode(v):
    magnitude = tl.minimum(v, 448.0)
    bits = magnitude.to(tl.int32, bitcast=True)
    bias = tl.maximum(bits & 2139095040, 1015021568) + 167772160
    biased = magnitude + bias.to(tl.float32, bitcast=True)
    code = biased.to(tl.int32, bitcast=True) - bias + (bias >> 20) - 1128
    code = tl.where(v != v, 127, code)
    return code.to(tl.uint8)


@triton.jit
def _activate(
    X,
    A,
    P,
    NT,
    D: tl.constexpr,
    T: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    B: tl.constexpr,
):
    n = tl.load(NT)
    lane = tl.arange(0, B)
    C: tl.constexpr = triton.cdiv(D, B)
    maximum = tl.full((B,), 0, tl.float32)
    for tile in range(tl.program_id(0), n * C, tl.num_programs(0)):
        row = tile // C
        col = tile % C * B + lane
        if D % B == 0:
            gate = tl.load(X + row * SX + col * CX).to(tl.float32)
            up = tl.load(X + row * SX + (col + D) * CX).to(tl.float32)
        else:
            safe = tl.minimum(col, D - 1)
            gate = tl.load(X + row * SX + safe * CX).to(tl.float32)
            up = tl.load(X + row * SX + (safe + D) * CX).to(tl.float32)
        value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
        value = tl.where(col < D, value, 0.0)
        maximum = tl.maximum(maximum, tl.abs(value))
        tl.store(A + row * D + col, value, col < D)
    tl.store(P + tl.program_id(0), tl.max(maximum, 0))


@triton.jit
def _scale_row2048(P, S, K: tl.constexpr, B: tl.constexpr):
    lane = tl.arange(0, B)
    maximum = tl.load(P + tl.minimum(lane, K - 1))
    maximum = tl.where(lane < K, maximum, 0.0)
    scale = tl.maximum(tl.max(maximum, 0), 1e-10) / 448.0
    tl.store(S, scale)


@triton.jit
def _quant(
    A,
    Q,
    S,
    NT,
    D: tl.constexpr,
    T: tl.constexpr,
    SQ: tl.constexpr,
    CQ: tl.constexpr,
    B: tl.constexpr,
):
    n = tl.load(NT)
    scale = tl.load(S)
    lane = tl.arange(0, B)
    C: tl.constexpr = triton.cdiv(D, B)
    for tile in range(tl.program_id(0), n * C, tl.num_programs(0)):
        row = tile // C
        col = tile % C * B + lane
        if D % B == 0:
            value = tl.load(A + row * D + col)
        else:
            safe = tl.minimum(col, D - 1)
            value = tl.load(A + row * D + safe)
        quantized = value / scale
        sign = quantized.to(tl.int32, bitcast=True) >> 24 & 128
        code = _encode(tl.abs(quantized)) | sign.to(tl.uint8)
        tl.store(Q + row * SQ + col * CQ, code, col < D)


def _row2048_entry(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    rows = input.shape[0]
    width = intermediate_size
    result = output
    result_bytes = result.view(torch.uint8)
    result_scale = scale
    activation = torch.empty(
        (rows, width), device=input.device, dtype=torch.float32
    )
    block = min(2048, width and 1 << (width - 1).bit_length())
    programs = min(1024, rows * ((width + block - 1) // block))
    partial = torch.empty(
        (programs,), device=input.device, dtype=torch.float32
    )
    _activate.run(
        input,
        activation,
        partial,
        num_tokens_tensor,
        width,
        rows,
        input.stride(0),
        input.stride(1),
        block,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    _scale_row2048.run(
        partial,
        result_scale,
        programs,
        programs and 1 << (programs - 1).bit_length(),
        num_warps=4,
        enable_fp_fusion=False,
        grid=(1,),
        warmup=False,
    )
    _quant.run(
        activation,
        result_bytes,
        result_scale,
        num_tokens_tensor,
        width,
        rows,
        result_bytes.stride(0),
        result_bytes.stride(1),
        block,
        num_warps=4,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    return (result, result_scale)


@triton.jit
def _activate_rows(
    X,
    A,
    P,
    NT,
    D: tl.constexpr,
    T: tl.constexpr,
    SX: tl.constexpr,
    CX: tl.constexpr,
    B: tl.constexpr,
):
    n = tl.load(NT)
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    lane = tl.arange(0, B)
    safe_rows = tl.minimum(rows, n - 1)
    input_policy: tl.constexpr = (
        "continuous" if T % 64 == 0 and D % B == 0 and (CX == 1) else ""
    )
    activation_policy: tl.constexpr = (
        "continuous" if T % 64 == 0 and D % B == 0 else ""
    )
    maximum = tl.full((64, B), 0.0, tl.float32)
    for tile in range(triton.cdiv(D, B)):
        cols = tile * B + lane
        if D % B == 0:
            gate = tl.load(
                X + safe_rows[:, None] * SX + cols[None, :] * CX,
                offset_state_policy=input_policy,
            ).to(tl.float32)
            up = tl.load(
                X + safe_rows[:, None] * SX + (cols[None, :] + D) * CX,
                offset_state_policy=input_policy,
            ).to(tl.float32)
        else:
            safe_cols = tl.minimum(cols, D - 1)
            gate = tl.load(
                X + safe_rows[:, None] * SX + safe_cols[None, :] * CX
            ).to(tl.float32)
            up = tl.load(
                X + safe_rows[:, None] * SX + (safe_cols[None, :] + D) * CX
            ).to(tl.float32)
        valid = (rows[:, None] < n) & (cols[None, :] < D)
        value = gate * (1.0 / (1.0 + tl.exp(-gate))) * up
        value = tl.where(valid, value, 0.0)
        maximum = tl.maximum(maximum, tl.abs(value))
        tl.store(
            A + rows[:, None] * D + cols[None, :],
            value,
            valid,
            offset_state_policy=activation_policy,
        )
    tl.store(P + rows, tl.max(maximum, axis=1))


@triton.jit
def _scale_live_rows(P, S, NT, K: tl.constexpr, B: tl.constexpr):
    n = tl.load(NT)
    lane = tl.arange(0, B)
    maximum = tl.load(P + lane, lane < n, 0.0)
    scale = tl.maximum(tl.max(maximum, 0), 1e-10) / 448.0
    tl.store(S, scale)


@triton.jit
def _quant_rows(
    A,
    Q,
    S,
    NT,
    D: tl.constexpr,
    T: tl.constexpr,
    SQ: tl.constexpr,
    CQ: tl.constexpr,
    B: tl.constexpr,
):
    n = tl.load(NT)
    scale = tl.load(S)
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    lane = tl.arange(0, B)
    safe_rows = tl.minimum(rows, n - 1)
    activation_policy: tl.constexpr = (
        "continuous" if T % 64 == 0 and D % B == 0 else ""
    )
    output_policy: tl.constexpr = (
        "continuous" if T % 64 == 0 and D % B == 0 and (CQ == 1) else ""
    )
    for tile in range(triton.cdiv(D, B)):
        cols = tile * B + lane
        if D % B == 0:
            value = tl.load(
                A + safe_rows[:, None] * D + cols[None, :],
                offset_state_policy=activation_policy,
            )
        else:
            safe_cols = tl.minimum(cols, D - 1)
            value = tl.load(A + safe_rows[:, None] * D + safe_cols[None, :])
        quantized = value / scale
        sign = quantized.to(tl.int32, bitcast=True) >> 24 & 128
        code = _encode(tl.abs(quantized)) | sign.to(tl.uint8)
        valid = (rows[:, None] < n) & (cols[None, :] < D)
        tl.store(
            Q + rows[:, None] * SQ + cols[None, :] * CQ,
            code,
            valid,
            offset_state_policy=output_policy,
        )


def _rows64_entry(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
    require_raw=False,
):
    rows = input.shape[0]
    width = intermediate_size
    block = min(128, width and 1 << (width - 1).bit_length())
    programs = (rows + 64 - 1) // 64
    count = programs * 64
    activation = torch.empty(
        (rows, width), device=input.device, dtype=torch.float32
    )
    result_bytes = output.view(torch.uint8)
    activation_aligned = (
        input.dtype == torch.bfloat16
        and input.data_ptr() % 64 == 0
        and (input.stride(0) % 32 == 0)
        and (activation.data_ptr() % 64 == 0)
        and (rows <= 2147483647)
    )
    raw_aligned = (
        activation.data_ptr() % 64 == 0
        and result_bytes.data_ptr() % 64 == 0
        and (result_bytes.stride(0) % 64 == 0)
        and (rows <= 2147483647)
    )
    fused_raw = activation_aligned and raw_aligned
    partial_count = (
        count and 1 << (count - 1).bit_length() if fused_raw else count
    )
    partial = torch.empty(
        (partial_count,), device=input.device, dtype=torch.float32
    )
    if require_raw and (not (activation_aligned and raw_aligned)):
        return _row2048_entry(
            input,
            output,
            scale,
            num_tokens_tensor,
            expected_num_tokens,
            intermediate_size,
        )
    if fused_raw:
        _activate_neutral207.run(
            input,
            activation,
            partial,
            num_tokens_tensor,
            width,
            input.stride(0),
            count,
            partial_count,
            num_warps=1,
            enable_fp_fusion=False,
            grid=(programs,),
            warmup=False,
        )
        _quant_neutral207.run(
            activation,
            result_bytes,
            partial,
            scale,
            num_tokens_tensor,
            width,
            result_bytes.stride(0),
            partial_count,
            num_warps=1,
            enable_fp_fusion=False,
            grid=(programs,),
            warmup=False,
            OVERLAP=rows * width >= 2097152,
        )
        return (output, scale)
    if activation_aligned:
        _activate_raw207.run(
            input,
            activation,
            partial,
            num_tokens_tensor,
            width,
            input.stride(0),
            num_warps=1,
            enable_fp_fusion=False,
            grid=(programs,),
            warmup=False,
        )
    else:
        _activate_rows.run(
            input,
            activation,
            partial,
            num_tokens_tensor,
            width,
            rows,
            input.stride(0),
            input.stride(1),
            block,
            num_warps=4,
            enable_fp_fusion=False,
            groups_per_cluster=64,
            grid=(programs,),
            warmup=False,
        )
    _scale_live_rows.run(
        partial,
        scale,
        num_tokens_tensor,
        count,
        count and 1 << (count - 1).bit_length(),
        num_warps=4,
        enable_fp_fusion=False,
        grid=(1,),
        warmup=False,
    )
    if raw_aligned:
        _raw_quant207.run(
            activation,
            result_bytes,
            scale,
            num_tokens_tensor,
            width,
            result_bytes.stride(0),
            num_warps=1,
            enable_fp_fusion=False,
            grid=(programs,),
            warmup=False,
        )
    else:
        _quant_rows.run(
            activation,
            result_bytes,
            scale,
            num_tokens_tensor,
            width,
            rows,
            result_bytes.stride(0),
            result_bytes.stride(1),
            block,
            num_warps=4,
            enable_fp_fusion=False,
            groups_per_cluster=64,
            grid=(programs,),
            warmup=False,
        )
    return (output, scale)


def silu_mul_dynamic_tensorwise_quant(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    rows, width = (input.shape[0], intermediate_size)
    continuous_rows = (
        rows % 64 == 0
        and width % 128 == 0
        and (input.stride(1) == 1)
        and (output.stride(1) == 1)
    )
    enough_row_parallelism = width <= 1024 or rows >= 512
    if continuous_rows and enough_row_parallelism:
        return _rows64_entry(
            input,
            output,
            scale,
            num_tokens_tensor,
            expected_num_tokens,
            intermediate_size,
        )
    if continuous_rows and rows >= 256 and (width <= 2048):
        return _rows64_entry(
            input,
            output,
            scale,
            num_tokens_tensor,
            expected_num_tokens,
            intermediate_size,
            require_raw=True,
        )
    if continuous_rows and rows < 256 and (1024 < width <= 2048):
        return _chunks128_entry(
            input,
            output,
            scale,
            num_tokens_tensor,
            expected_num_tokens,
            intermediate_size,
        )
    return _row2048_entry(
        input,
        output,
        scale,
        num_tokens_tensor,
        expected_num_tokens,
        intermediate_size,
    )


RAW207 = (
    "\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef unsigned char __attribute__((address_space(1))) gu8;\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "union bits207 { f32v f; u32v u; };\n"
    "typedef bool b32v __attribute__((ext_vector_type(32)));\n"
    'extern "C" __attribute__((device)) b32v nan_compare207(int, u32v)'
    ' __asm__("llvm.xpu.svle_s_mr1");\n'
    'extern "C" __attribute__((device)) u32v nan_select207(b32v, unsig'
    'ned int, u32v, u32v) __asm__("llvm.xpu.svadd_u_mh");\n'
    "union scalar_bits207 { float f; unsigned int u; };\n"
    "namespace xpu { __attribute__((device)) float __fdiv_rn(float, fl"
    "oat); }\n"
    "\n"
    "__attribute__((device,always_inline)) u32v opaque207(u32v value) "
    "{\n"
    '    __asm__("" : "+v"(value));\n'
    "    return value;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v mac207(f32v a, f32v b,"
    " f32v c) {\n"
    '    __asm__("vmac.f.mz.rn %0{mr1}, %1, %2" : "+v"(c) : "v"(a), "v'
    '"(b));\n'
    "    return c;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode207(\n"
    "    f32v values, float scale, float reciprocal, bool finite_scale"
    ") {\n"
    "    bits207 quantized, magnitude, limit, exponent, floor, bias, r"
    "ounded;\n"
    "    if (finite_scale) {\n"
    "        bits207 value, coarse;\n"
    "        value.f = values;\n"
    "        coarse.f = values * (f32v)reciprocal;\n"
    "        coarse.u = opaque207(coarse.u);\n"
    "        f32v residual = mac207(-coarse.f, (f32v)scale, values);\n"
    "        quantized.f = mac207(residual, (f32v)reciprocal, coarse.f"
    ");\n"
    "        quantized.u = (quantized.u & 0x7fffffffu) | (value.u & 0x"
    "80000000u);\n"
    "    } else {\n"
    "        quantized.f = values / (f32v)scale;\n"
    "    }\n"
    "    u32v absolute_bits = quantized.u & 0x7fffffffu;\n"
    "    magnitude.u = absolute_bits;\n"
    "    limit.u = (u32v)0x43e00000u;\n"
    "    magnitude.f = __builtin_elementwise_min(magnitude.f, limit.f)"
    ";\n"
    "    exponent.u = magnitude.u & 0x7f800000u;\n"
    "    floor.u = (u32v)1015021568u;\n"
    "    exponent.f = __builtin_elementwise_max(exponent.f, floor.f);\n"
    "    bias.u = exponent.u + 167772160u;\n"
    "    rounded.f = magnitude.f + bias.f;\n"
    "    u32v code = rounded.u - bias.u + (bias.u >> 20) - 1128u;\n"
    "    b32v nan_mask = nan_compare207(0x7f800001, absolute_bits);\n"
    "    code = nan_select207(nan_mask, 127u, (u32v)0u, code);\n"
    "    return code | ((quantized.u >> 24) & 128u);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v pack207(u32v a, u32v b"
    ", u32v c, u32v d) {\n"
    "    u32v a1 = __builtin_xpu2_vmerge_l_u(a, c);\n"
    "    u32v b1 = __builtin_xpu2_vmerge_h_u(a, c);\n"
    "    u32v c1 = __builtin_xpu2_vmerge_l_u(b, d);\n"
    "    u32v d1 = __builtin_xpu2_vmerge_h_u(b, d);\n"
    "    u32v a2 = __builtin_xpu2_vmerge_l_u(a1, c1);\n"
    "    u32v b2 = __builtin_xpu2_vmerge_h_u(a1, c1);\n"
    "    u32v c2 = __builtin_xpu2_vmerge_l_u(b1, d1);\n"
    "    u32v d2 = __builtin_xpu2_vmerge_h_u(b1, d1);\n"
    "    u32v a3 = __builtin_xpu2_vmerge_l_u(a2, c2);\n"
    "    u32v b3 = __builtin_xpu2_vmerge_h_u(a2, c2);\n"
    "    u32v c3 = __builtin_xpu2_vmerge_l_u(b2, d2);\n"
    "    u32v d3 = __builtin_xpu2_vmerge_h_u(b2, d2);\n"
    "    u32v a4 = __builtin_xpu2_vmerge_l_u(a3, c3);\n"
    "    u32v b4 = __builtin_xpu2_vmerge_h_u(a3, c3);\n"
    "    u32v c4 = __builtin_xpu2_vmerge_l_u(b3, d3);\n"
    "    u32v d4 = __builtin_xpu2_vmerge_h_u(b3, d3);\n"
    "    return a4 | (b4 << 8) | (c4 << 16) | (d4 << 24);\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _raw207(\n'
    "    const gf32* A, gu8* Q, float scale, int n, long long D, long "
    "long SQ, int pid) {\n"
    "    float input[512] __attribute__((aligned(64)));\n"
    "    unsigned int output[128] __attribute__((aligned(64)));\n"
    "    long long row = (long long)pid * 64 + __builtin_xpu_core_id()"
    ";\n"
    "    if (row >= n) return;\n"
    "\n"
    "    scalar_bits207 scale_bits;\n"
    "    scale_bits.f = scale;\n"
    "    bool finite_scale = scale_bits.u >= 0x00800000u && scale_bits"
    ".u < 0x7f800000u;\n"
    "    float reciprocal = 0.0f;\n"
    "    if (finite_scale) reciprocal = xpu::__fdiv_rn(1.0f, scale);\n"
    "    for (long long column = 0; column < D; column += 512) {\n"
    "        int count = (D - column < 512) ? (int)(D - column) : 512;"
    "\n"
    "        __builtin_xpu_gm2lm_v3(input, A + row * D + column, 0, co"
    "unt * 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        for (int offset = 0; offset < count; offset += 64) {\n"
    "            u32v c0 = encode207(*((f32v*)(input + offset)), scale"
    ", reciprocal, finite_scale);\n"
    "            u32v c1 = encode207(*((f32v*)(input + offset + 16)), "
    "scale, reciprocal, finite_scale);\n"
    "            u32v c2 = encode207(*((f32v*)(input + offset + 32)), "
    "scale, reciprocal, finite_scale);\n"
    "            u32v c3 = encode207(*((f32v*)(input + offset + 48)), "
    "scale, reciprocal, finite_scale);\n"
    "            *((u32v*)(output + offset / 4)) = pack207(c0, c1, c2,"
    " c3);\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(Q + row * SQ + column, output, 0, "
    "count);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _raw207_overlap(\n'
    "    const gf32* A, gu8* Q, float scale, int n, long long D, long "
    "long SQ, int pid) {\n"
    "    float input0[512] __attribute__((aligned(64)));\n"
    "    float input1[512] __attribute__((aligned(64)));\n"
    "    float* input = input0;\n"
    "    float* spare_input = input1;\n"
    "    unsigned int output0[128] __attribute__((aligned(64)));\n"
    "    unsigned int output1[128] __attribute__((aligned(64)));\n"
    "    unsigned int* output = output0;\n"
    "    unsigned int* spare_output = output1;\n"
    "    long long row = (long long)pid * 64 + __builtin_xpu_core_id()"
    ";\n"
    "    if (row >= n) return;\n"
    "\n"
    "    scalar_bits207 scale_bits;\n"
    "    scale_bits.f = scale;\n"
    "    bool finite_scale = scale_bits.u >= 0x00800000u && scale_bits"
    ".u < 0x7f800000u;\n"
    "    float reciprocal = 0.0f;\n"
    "    if (finite_scale) reciprocal = xpu::__fdiv_rn(1.0f, scale);\n"
    "    int first_count = (D < 512) ? (int)D : 512;\n"
    "    __builtin_xpu_gm2lm_v3(input, A + row * D, 0, first_count * 4"
    ");\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (long long column = 0; column < D; column += 512) {\n"
    "        int count = (D - column < 512) ? (int)(D - column) : 512;"
    "\n"
    "        if (column + 512 < D) {\n"
    "            int next_count = (D - column - 512 < 512) ? (int)(D -"
    " column - 512) : 512;\n"
    "            __builtin_xpu_gm2lm_v3(spare_input, A + row * D + col"
    "umn + 512, 0, next_count * 4);\n"
    "        }\n"
    "        for (int offset = 0; offset < count; offset += 64) {\n"
    "            u32v c0 = encode207(*((f32v*)(input + offset)), scale"
    ", reciprocal, finite_scale);\n"
    "            u32v c1 = encode207(*((f32v*)(input + offset + 16)), "
    "scale, reciprocal, finite_scale);\n"
    "            u32v c2 = encode207(*((f32v*)(input + offset + 32)), "
    "scale, reciprocal, finite_scale);\n"
    "            u32v c3 = encode207(*((f32v*)(input + offset + 48)), "
    "scale, reciprocal, finite_scale);\n"
    "            *((u32v*)(output + offset / 4)) = pack207(c0, c1, c2,"
    " c3);\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(Q + row * SQ + column, output, 0, "
    "count);\n"
    "        float* previous_input = input;\n"
    "        input = spare_input;\n"
    "        spare_input = previous_input;\n"
    "        unsigned int* previous_output = output;\n"
    "        output = spare_output;\n"
    "        spare_output = previous_output;\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect("xpu3", source=RAW207, flags=("-ffp-contract=off",))
def _raw207(A, Q, scale, n, D, SQ, pid):
    pass


@tle.raw.dialect("xpu3", source=RAW207, flags=("-ffp-contract=off",))
def _raw207_overlap(A, Q, scale, n, D, SQ, pid):
    pass


@triton.jit
def _raw_quant207(A, Q, S, NT, D: tl.constexpr, SQ: tl.constexpr):
    scale = tl.load(S)
    n = tl.load(NT)
    tle.raw.call(
        _raw207,
        (
            A,
            Q,
            scale,
            n,
            tl.full((), D, tl.int64),
            tl.full((), SQ, tl.int64),
            tl.program_id(0),
        ),
    )


RAW_ACTIVATION207 = (
    "\n"
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "union bits_activate207 { f32v f; u32v u; };\n"
    "\n"
    "typedef bool b32v __attribute__((ext_vector_type(32)));\n"
    'extern "C" __attribute__((device)) b32v compare_inf207(float, f32'
    'v) __asm__("llvm.xpu.svneq_f_mr1");\n'
    'extern "C" __attribute__((device)) f32v masked_mul207(b32v, f32v,'
    ' f32v) __asm__("llvm.xpu.vvmul_f_mz_rn");\n'
    "\n"
    "__attribute__((device,always_inline)) f32v reciprocal_seed207(f32"
    "v denominator) {\n"
    "    bits_activate207 input, estimate;\n"
    "    input.f = denominator;\n"
    "    estimate.u = (u32v)0x5f3759dfu - (input.u >> 1);\n"
    "    b32v finite_mask = compare_inf207(__builtin_inff(), denominat"
    "or);\n"
    "    return masked_mul207(finite_mask, estimate.f, estimate.f);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v mac_activate207(f32v a"
    ", f32v b, f32v c) {\n"
    '    __asm__("vmac.f.mz.rn %0{mr1}, %1, %2" : "+v"(c) : "v"(a), "v'
    '"(b));\n'
    "    return c;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v swiglu207(u32v gate_bi"
    "ts, u32v up_bits) {\n"
    "    bits_activate207 gate, up;\n"
    "    gate.u = gate_bits;\n"
    "    up.u = up_bits;\n"
    "    f32v exponential = __builtin_elementwise_exp2(-gate.f);\n"
    "    f32v denominator = (f32v)1.0f + exponential;\n"
    "    f32v reciprocal = reciprocal_seed207(denominator);\n"
    "    f32v safe_denominator = __builtin_elementwise_min(denominator"
    ", (f32v)0x1.fffffep127f);\n"
    "    f32v residual = mac_activate207(-safe_denominator, reciprocal"
    ", (f32v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    residual = mac_activate207(-safe_denominator, reciprocal, (f3"
    "2v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    residual = mac_activate207(-safe_denominator, reciprocal, (f3"
    "2v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    return gate.f * reciprocal * up.f;\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _raw_activate207(\n'
    "    const gu16* X, gf32* A, gf32* P, int n, long long D, long lon"
    "g SX, int pid) {\n"
    "    unsigned int gate0[64] __attribute__((aligned(64)));\n"
    "    unsigned int gate1[64] __attribute__((aligned(64)));\n"
    "    unsigned int up0[64] __attribute__((aligned(64)));\n"
    "    unsigned int up1[64] __attribute__((aligned(64)));\n"
    "    unsigned int* gate_input = gate0;\n"
    "    unsigned int* gate_spare = gate1;\n"
    "    unsigned int* up_input = up0;\n"
    "    unsigned int* up_spare = up1;\n"
    "    float output[128] __attribute__((aligned(64)));\n"
    "    float partial[1] __attribute__((aligned(64)));\n"
    "    f32v maximum = (f32v)0.0f;\n"
    "    long long row = (long long)pid * 64 + __builtin_xpu_core_id()"
    ";\n"
    "    if (row >= n) return;\n"
    "    __builtin_xpu_gm2lm_v3(gate_input, X + row * SX, 0, 256);\n"
    "    __builtin_xpu_gm2lm_v3(up_input, X + row * SX + D, 0, 256);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (long long column = 0; column < D; column += 128) {\n"
    "        if (column + 128 < D) {\n"
    "            __builtin_xpu_gm2lm_v3(gate_spare, X + row * SX + col"
    "umn + 128, 0, 256);\n"
    "            __builtin_xpu_gm2lm_v3(up_spare, X + row * SX + D + c"
    "olumn + 128, 0, 256);\n"
    "        }\n"
    "        for (int offset = 0; offset < 128; offset += 32) {\n"
    "            u32v gate = *((u32v*)(gate_input + offset / 2));\n"
    "            u32v up = *((u32v*)(up_input + offset / 2));\n"
    "            bits_activate207 even, odd, absolute_even, absolute_o"
    "dd;\n"
    "            even.f = swiglu207(gate << 16, up << 16);\n"
    "            odd.f = swiglu207(gate & 0xffff0000u, up & 0xffff0000"
    "u);\n"
    "            *((u32v*)(output + offset)) = __builtin_xpu2_vmerge_l"
    "_u(even.u, odd.u);\n"
    "            *((u32v*)(output + offset + 16)) = __builtin_xpu2_vme"
    "rge_h_u(even.u, odd.u);\n"
    "            absolute_even.u = even.u & 0x7fffffffu;\n"
    "            absolute_odd.u = odd.u & 0x7fffffffu;\n"
    "            maximum = __builtin_elementwise_max(maximum,\n"
    "                       __builtin_elementwise_max(absolute_even.f,"
    " absolute_odd.f));\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(A + row * D + column, output, 0, 5"
    "12);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        unsigned int* previous_gate = gate_input;\n"
    "        gate_input = gate_spare;\n"
    "        gate_spare = previous_gate;\n"
    "        unsigned int* previous_up = up_input;\n"
    "        up_input = up_spare;\n"
    "        up_spare = previous_up;\n"
    "    }\n"
    "    partial[0] = __builtin_reduce_max(maximum);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(P + row, partial, 0, 4);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect(
    "xpu3", source=RAW_ACTIVATION207, flags=("-ffp-contract=off",)
)
def _raw_activate207(X, A, P, n, D, SX, pid):
    pass


@triton.jit
def _activate_raw207(X, A, P, NT, D: tl.constexpr, SX: tl.constexpr):
    n = tl.load(NT)
    tle.raw.call(
        _raw_activate207,
        (
            X.to(tl.pointer_type(tl.uint16)),
            A,
            P,
            n,
            tl.full((), D, tl.int64),
            tl.full((), SX, tl.int64),
            tl.program_id(0),
        ),
    )


RAW_ACTIVATION_NEUTRAL207 = (
    "\n"
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "union bits_activate207 { f32v f; u32v u; };\n"
    "\n"
    "typedef bool b32v __attribute__((ext_vector_type(32)));\n"
    'extern "C" __attribute__((device)) b32v compare_inf207(float, f32'
    'v) __asm__("llvm.xpu.svneq_f_mr1");\n'
    'extern "C" __attribute__((device)) f32v masked_mul207(b32v, f32v,'
    ' f32v) __asm__("llvm.xpu.vvmul_f_mz_rn");\n'
    "\n"
    "__attribute__((device,always_inline)) f32v reciprocal_seed207(f32"
    "v denominator) {\n"
    "    bits_activate207 input, estimate;\n"
    "    input.f = denominator;\n"
    "    estimate.u = (u32v)0x5f3759dfu - (input.u >> 1);\n"
    "    b32v finite_mask = compare_inf207(__builtin_inff(), denominat"
    "or);\n"
    "    return masked_mul207(finite_mask, estimate.f, estimate.f);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v mac_activate207(f32v a"
    ", f32v b, f32v c) {\n"
    '    __asm__("vmac.f.mz.rn %0{mr1}, %1, %2" : "+v"(c) : "v"(a), "v'
    '"(b));\n'
    "    return c;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v swiglu207(u32v gate_bi"
    "ts, u32v up_bits) {\n"
    "    bits_activate207 gate, up;\n"
    "    gate.u = gate_bits;\n"
    "    up.u = up_bits;\n"
    "    f32v exponential = __builtin_elementwise_exp2(-gate.f);\n"
    "    f32v denominator = (f32v)1.0f + exponential;\n"
    "    f32v reciprocal = reciprocal_seed207(denominator);\n"
    "    f32v safe_denominator = __builtin_elementwise_min(denominator"
    ", (f32v)0x1.fffffep127f);\n"
    "    f32v residual = mac_activate207(-safe_denominator, reciprocal"
    ", (f32v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    residual = mac_activate207(-safe_denominator, reciprocal, (f3"
    "2v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    residual = mac_activate207(-safe_denominator, reciprocal, (f3"
    "2v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    return gate.f * reciprocal * up.f;\n"
    "}\n"
    "\n"
    'extern "C" __attribute__((device)) void _raw_activate_neutral207('
    "\n"
    "    const gu16* X, gf32* A, gf32* P, int n, long long D, long lon"
    "g SX, int pid, long long count, long long padded) {\n"
    "    unsigned int gate0[64] __attribute__((aligned(64)));\n"
    "    unsigned int gate1[64] __attribute__((aligned(64)));\n"
    "    unsigned int up0[64] __attribute__((aligned(64)));\n"
    "    unsigned int up1[64] __attribute__((aligned(64)));\n"
    "    unsigned int* gate_input = gate0;\n"
    "    unsigned int* gate_spare = gate1;\n"
    "    unsigned int* up_input = up0;\n"
    "    unsigned int* up_spare = up1;\n"
    "    float output[128] __attribute__((aligned(64)));\n"
    "    float partial[1] __attribute__((aligned(64)));\n"
    "    f32v maximum = (f32v)0.0f;\n"
    "    long long row = (long long)pid * 64 + __builtin_xpu_core_id()"
    ";\n"
    "    partial[0] = 0.0f;\n"
    "    if (row + count < padded) {\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(P + row + count, partial, 0, 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "    if (row >= n) {\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(P + row, partial, 0, 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        return;\n"
    "    }\n"
    "    __builtin_xpu_gm2lm_v3(gate_input, X + row * SX, 0, 256);\n"
    "    __builtin_xpu_gm2lm_v3(up_input, X + row * SX + D, 0, 256);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (long long column = 0; column < D; column += 128) {\n"
    "        if (column + 128 < D) {\n"
    "            __builtin_xpu_gm2lm_v3(gate_spare, X + row * SX + col"
    "umn + 128, 0, 256);\n"
    "            __builtin_xpu_gm2lm_v3(up_spare, X + row * SX + D + c"
    "olumn + 128, 0, 256);\n"
    "        }\n"
    "        for (int offset = 0; offset < 128; offset += 32) {\n"
    "            u32v gate = *((u32v*)(gate_input + offset / 2));\n"
    "            u32v up = *((u32v*)(up_input + offset / 2));\n"
    "            bits_activate207 even, odd, absolute_even, absolute_o"
    "dd;\n"
    "            even.f = swiglu207(gate << 16, up << 16);\n"
    "            odd.f = swiglu207(gate & 0xffff0000u, up & 0xffff0000"
    "u);\n"
    "            *((u32v*)(output + offset)) = __builtin_xpu2_vmerge_l"
    "_u(even.u, odd.u);\n"
    "            *((u32v*)(output + offset + 16)) = __builtin_xpu2_vme"
    "rge_h_u(even.u, odd.u);\n"
    "            absolute_even.u = even.u & 0x7fffffffu;\n"
    "            absolute_odd.u = odd.u & 0x7fffffffu;\n"
    "            maximum = __builtin_elementwise_max(maximum,\n"
    "                       __builtin_elementwise_max(absolute_even.f,"
    " absolute_odd.f));\n"
    "        }\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(A + row * D + column, output, 0, 5"
    "12);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        unsigned int* previous_gate = gate_input;\n"
    "        gate_input = gate_spare;\n"
    "        gate_spare = previous_gate;\n"
    "        unsigned int* previous_up = up_input;\n"
    "        up_input = up_spare;\n"
    "        up_spare = previous_up;\n"
    "    }\n"
    "    partial[0] = __builtin_reduce_max(maximum);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(P + row, partial, 0, 4);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect(
    "xpu3", source=RAW_ACTIVATION_NEUTRAL207, flags=("-ffp-contract=off",)
)
def _raw_activate_neutral207(X, A, P, n, D, SX, pid, count, padded):
    pass


@triton.jit
def _activate_neutral207(
    X,
    A,
    P,
    NT,
    D: tl.constexpr,
    SX: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    n = tl.load(NT)
    tle.raw.call(
        _raw_activate_neutral207,
        (
            X.to(tl.pointer_type(tl.uint16)),
            A,
            P,
            n,
            tl.full((), D, tl.int64),
            tl.full((), SX, tl.int64),
            tl.program_id(0),
            tl.full((), K, tl.int64),
            tl.full((), B, tl.int64),
        ),
    )


@triton.jit
def _quant_neutral207(
    A,
    Q,
    P,
    S,
    NT,
    D: tl.constexpr,
    SQ: tl.constexpr,
    B: tl.constexpr,
    OVERLAP: tl.constexpr,
):
    n = tl.load(NT)
    lane = tl.arange(0, B)
    maximum = tl.load(P + lane)
    scale = tl.maximum(tl.max(maximum, 0), 1e-10) / 448.0
    if tl.program_id(0) == 0:
        tl.store(S, scale)
    if OVERLAP:
        tle.raw.call(
            _raw207_overlap,
            (
                A,
                Q,
                scale,
                n,
                tl.full((), D, tl.int64),
                tl.full((), SQ, tl.int64),
                tl.program_id(0),
            ),
        )
    else:
        tle.raw.call(
            _raw207,
            (
                A,
                Q,
                scale,
                n,
                tl.full((), D, tl.int64),
                tl.full((), SQ, tl.int64),
                tl.program_id(0),
            ),
        )


RAW_ACTIVATION_CHUNK207 = (
    "\n"
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "union bits_activate207 { f32v f; u32v u; };\n"
    "\n"
    "typedef bool b32v __attribute__((ext_vector_type(32)));\n"
    'extern "C" __attribute__((device)) b32v compare_inf207(float, f32'
    'v) __asm__("llvm.xpu.svneq_f_mr1");\n'
    'extern "C" __attribute__((device)) f32v masked_mul207(b32v, f32v,'
    ' f32v) __asm__("llvm.xpu.vvmul_f_mz_rn");\n'
    "\n"
    "__attribute__((device,always_inline)) f32v reciprocal_seed207(f32"
    "v denominator) {\n"
    "    bits_activate207 input, estimate;\n"
    "    input.f = denominator;\n"
    "    estimate.u = (u32v)0x5f3759dfu - (input.u >> 1);\n"
    "    b32v finite_mask = compare_inf207(__builtin_inff(), denominat"
    "or);\n"
    "    return masked_mul207(finite_mask, estimate.f, estimate.f);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v mac_activate207(f32v a"
    ", f32v b, f32v c) {\n"
    '    __asm__("vmac.f.mz.rn %0{mr1}, %1, %2" : "+v"(c) : "v"(a), "v'
    '"(b));\n'
    "    return c;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v swiglu207(u32v gate_bi"
    "ts, u32v up_bits) {\n"
    "    bits_activate207 gate, up;\n"
    "    gate.u = gate_bits;\n"
    "    up.u = up_bits;\n"
    "    f32v exponential = __builtin_elementwise_exp2(-gate.f);\n"
    "    f32v denominator = (f32v)1.0f + exponential;\n"
    "    f32v reciprocal = reciprocal_seed207(denominator);\n"
    "    f32v safe_denominator = __builtin_elementwise_min(denominator"
    ", (f32v)0x1.fffffep127f);\n"
    "    f32v residual = mac_activate207(-safe_denominator, reciprocal"
    ", (f32v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    residual = mac_activate207(-safe_denominator, reciprocal, (f3"
    "2v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    residual = mac_activate207(-safe_denominator, reciprocal, (f3"
    "2v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    return gate.f * reciprocal * up.f;\n"
    "}\n"
    "\n"
    "\n"
    'extern "C" __attribute__((device)) void _raw_activate_chunk207(\n'
    "    const gu16* X, gf32* A, gf32* P, int n, long long D, long lon"
    "g SX, int pid) {\n"
    "    unsigned int gate_input[64] __attribute__((aligned(64)));\n"
    "    unsigned int up_input[64] __attribute__((aligned(64)));\n"
    "    float output[128] __attribute__((aligned(64)));\n"
    "    float partial[1] __attribute__((aligned(64)));\n"
    "    f32v maximum = (f32v)0.0f;\n"
    "    long long chunks = D / 128;\n"
    "    long long item = (long long)pid * 64 + __builtin_xpu_core_id("
    ");\n"
    "    long long live_items = (long long)n * chunks;\n"
    "    if (item >= live_items) return;\n"
    "    long long row = item / chunks;\n"
    "    long long column = (item % chunks) * 128;\n"
    "    __builtin_xpu_gm2lm_v3(gate_input, X + row * SX + column, 0, "
    "256);\n"
    "    __builtin_xpu_gm2lm_v3(up_input, X + row * SX + D + column, 0"
    ", 256);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (int offset = 0; offset < 128; offset += 32) {\n"
    "        u32v gate = *((u32v*)(gate_input + offset / 2));\n"
    "        u32v up = *((u32v*)(up_input + offset / 2));\n"
    "        bits_activate207 even, odd, absolute_even, absolute_odd;\n"
    "        even.f = swiglu207(gate << 16, up << 16);\n"
    "        odd.f = swiglu207(gate & 0xffff0000u, up & 0xffff0000u);\n"
    "        *((u32v*)(output + offset)) = __builtin_xpu2_vmerge_l_u(e"
    "ven.u, odd.u);\n"
    "        *((u32v*)(output + offset + 16)) = __builtin_xpu2_vmerge_"
    "h_u(even.u, odd.u);\n"
    "        absolute_even.u = even.u & 0x7fffffffu;\n"
    "        absolute_odd.u = odd.u & 0x7fffffffu;\n"
    "        maximum = __builtin_elementwise_max(maximum,\n"
    "                   __builtin_elementwise_max(absolute_even.f, abs"
    "olute_odd.f));\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(A + row * D + column, output, 0, 512);"
    "\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    partial[0] = __builtin_reduce_max(maximum);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(P + item, partial, 0, 4);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)
RAW_QUANT_CHUNK207 = (
    "\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef unsigned char __attribute__((address_space(1))) gu8;\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "union bits207 { f32v f; u32v u; };\n"
    "typedef bool b32v __attribute__((ext_vector_type(32)));\n"
    'extern "C" __attribute__((device)) b32v nan_compare207(int, u32v)'
    ' __asm__("llvm.xpu.svle_s_mr1");\n'
    'extern "C" __attribute__((device)) u32v nan_select207(b32v, unsig'
    'ned int, u32v, u32v) __asm__("llvm.xpu.svadd_u_mh");\n'
    "union scalar_bits207 { float f; unsigned int u; };\n"
    "namespace xpu { __attribute__((device)) float __fdiv_rn(float, fl"
    "oat); }\n"
    "\n"
    "__attribute__((device,always_inline)) u32v opaque207(u32v value) "
    "{\n"
    '    __asm__("" : "+v"(value));\n'
    "    return value;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v mac207(f32v a, f32v b,"
    " f32v c) {\n"
    '    __asm__("vmac.f.mz.rn %0{mr1}, %1, %2" : "+v"(c) : "v"(a), "v'
    '"(b));\n'
    "    return c;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v encode207(\n"
    "    f32v values, float scale, float reciprocal, bool finite_scale"
    ") {\n"
    "    bits207 quantized, magnitude, limit, exponent, floor, bias, r"
    "ounded;\n"
    "    if (finite_scale) {\n"
    "        bits207 value, coarse;\n"
    "        value.f = values;\n"
    "        coarse.f = values * (f32v)reciprocal;\n"
    "        coarse.u = opaque207(coarse.u);\n"
    "        f32v residual = mac207(-coarse.f, (f32v)scale, values);\n"
    "        quantized.f = mac207(residual, (f32v)reciprocal, coarse.f"
    ");\n"
    "        quantized.u = (quantized.u & 0x7fffffffu) | (value.u & 0x"
    "80000000u);\n"
    "    } else {\n"
    "        quantized.f = values / (f32v)scale;\n"
    "    }\n"
    "    u32v absolute_bits = quantized.u & 0x7fffffffu;\n"
    "    magnitude.u = absolute_bits;\n"
    "    limit.u = (u32v)0x43e00000u;\n"
    "    magnitude.f = __builtin_elementwise_min(magnitude.f, limit.f)"
    ";\n"
    "    exponent.u = magnitude.u & 0x7f800000u;\n"
    "    floor.u = (u32v)1015021568u;\n"
    "    exponent.f = __builtin_elementwise_max(exponent.f, floor.f);\n"
    "    bias.u = exponent.u + 167772160u;\n"
    "    rounded.f = magnitude.f + bias.f;\n"
    "    u32v code = rounded.u - bias.u + (bias.u >> 20) - 1128u;\n"
    "    b32v nan_mask = nan_compare207(0x7f800001, absolute_bits);\n"
    "    code = nan_select207(nan_mask, 127u, (u32v)0u, code);\n"
    "    return code | ((quantized.u >> 24) & 128u);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) u32v pack207(u32v a, u32v b"
    ", u32v c, u32v d) {\n"
    "    u32v a1 = __builtin_xpu2_vmerge_l_u(a, c);\n"
    "    u32v b1 = __builtin_xpu2_vmerge_h_u(a, c);\n"
    "    u32v c1 = __builtin_xpu2_vmerge_l_u(b, d);\n"
    "    u32v d1 = __builtin_xpu2_vmerge_h_u(b, d);\n"
    "    u32v a2 = __builtin_xpu2_vmerge_l_u(a1, c1);\n"
    "    u32v b2 = __builtin_xpu2_vmerge_h_u(a1, c1);\n"
    "    u32v c2 = __builtin_xpu2_vmerge_l_u(b1, d1);\n"
    "    u32v d2 = __builtin_xpu2_vmerge_h_u(b1, d1);\n"
    "    u32v a3 = __builtin_xpu2_vmerge_l_u(a2, c2);\n"
    "    u32v b3 = __builtin_xpu2_vmerge_h_u(a2, c2);\n"
    "    u32v c3 = __builtin_xpu2_vmerge_l_u(b2, d2);\n"
    "    u32v d3 = __builtin_xpu2_vmerge_h_u(b2, d2);\n"
    "    u32v a4 = __builtin_xpu2_vmerge_l_u(a3, c3);\n"
    "    u32v b4 = __builtin_xpu2_vmerge_h_u(a3, c3);\n"
    "    u32v c4 = __builtin_xpu2_vmerge_l_u(b3, d3);\n"
    "    u32v d4 = __builtin_xpu2_vmerge_h_u(b3, d3);\n"
    "    return a4 | (b4 << 8) | (c4 << 16) | (d4 << 24);\n"
    "}\n"
    "\n"
    "\n"
    'extern "C" __attribute__((device)) void _raw_quant_chunk207(\n'
    "    const gf32* A, gu8* Q, float scale, int n, long long D, long "
    "long SQ, int pid) {\n"
    "    float input[128] __attribute__((aligned(64)));\n"
    "    unsigned int output[32] __attribute__((aligned(64)));\n"
    "    long long chunks = D / 128;\n"
    "    long long item = (long long)pid * 64 + __builtin_xpu_core_id("
    ");\n"
    "    long long live_items = (long long)n * chunks;\n"
    "    if (item >= live_items) return;\n"
    "\n"
    "    scalar_bits207 scale_bits;\n"
    "    scale_bits.f = scale;\n"
    "    bool finite_scale = scale_bits.u >= 0x00800000u && scale_bits"
    ".u < 0x7f800000u;\n"
    "    float reciprocal = 0.0f;\n"
    "    if (finite_scale) reciprocal = xpu::__fdiv_rn(1.0f, scale);\n"
    "    long long row = item / chunks;\n"
    "    long long column = (item % chunks) * 128;\n"
    "    __builtin_xpu_gm2lm_v3(input, A + row * D + column, 0, 512);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (int offset = 0; offset < 128; offset += 64) {\n"
    "        u32v c0 = encode207(*((f32v*)(input + offset)), scale, re"
    "ciprocal, finite_scale);\n"
    "        u32v c1 = encode207(*((f32v*)(input + offset + 16)), scal"
    "e, reciprocal, finite_scale);\n"
    "        u32v c2 = encode207(*((f32v*)(input + offset + 32)), scal"
    "e, reciprocal, finite_scale);\n"
    "        u32v c3 = encode207(*((f32v*)(input + offset + 48)), scal"
    "e, reciprocal, finite_scale);\n"
    "        *((u32v*)(output + offset / 4)) = pack207(c0, c1, c2, c3)"
    ";\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(Q + row * SQ + column, output, 0, 128)"
    ";\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect(
    "xpu3", source=RAW_ACTIVATION_CHUNK207, flags=("-ffp-contract=off",)
)
def _raw_activate_chunk207(X, A, P, n, D, SX, pid):
    pass


@tle.raw.dialect(
    "xpu3", source=RAW_QUANT_CHUNK207, flags=("-ffp-contract=off",)
)
def _raw_quant_chunk207(A, Q, scale, n, D, SQ, pid):
    pass


@triton.jit
def _activate_chunk207(X, A, P, NT, D: tl.constexpr, SX: tl.constexpr):
    n = tl.load(NT)
    tle.raw.call(
        _raw_activate_chunk207,
        (
            X.to(tl.pointer_type(tl.uint16)),
            A,
            P,
            n,
            tl.full((), D, tl.int64),
            tl.full((), SX, tl.int64),
            tl.program_id(0),
        ),
    )


@triton.jit
def _scale_chunk207(P, S, NT, C: tl.constexpr, B: tl.constexpr):
    n = tl.load(NT).to(tl.int64)
    live_items = n * tl.full((), C, tl.int64)
    lane = tl.arange(0, B)
    maximum = tl.load(P + lane, lane.to(tl.int64) < live_items, 0.0)
    scale = tl.maximum(tl.max(maximum, 0), 1e-10) / 448.0
    tl.store(S, scale)


@triton.jit
def _quant_chunk207(A, Q, S, NT, D: tl.constexpr, SQ: tl.constexpr):
    scale = tl.load(S)
    n = tl.load(NT)
    tle.raw.call(
        _raw_quant_chunk207,
        (
            A,
            Q,
            scale,
            n,
            tl.full((), D, tl.int64),
            tl.full((), SQ, tl.int64),
            tl.program_id(0),
        ),
    )


def _chunks128_entry(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    rows, width = (input.shape[0], intermediate_size)
    result_bytes = output.view(torch.uint8)
    qualified = (
        input.dtype == torch.bfloat16
        and input.data_ptr() % 64 == 0
        and (input.stride(0) % 32 == 0)
        and (result_bytes.data_ptr() % 64 == 0)
        and (result_bytes.stride(0) % 64 == 0)
        and (rows <= 2147483647)
    )
    if not qualified:
        return _row2048_entry(
            input,
            output,
            scale,
            num_tokens_tensor,
            expected_num_tokens,
            intermediate_size,
        )
    activation = torch.empty(
        (rows, width), device=input.device, dtype=torch.float32
    )
    if activation.data_ptr() % 64 != 0:
        return _row2048_entry(
            input,
            output,
            scale,
            num_tokens_tensor,
            expected_num_tokens,
            intermediate_size,
        )
    chunks = width // 128
    items = rows * chunks
    programs = (items + 64 - 1) // 64
    padded = items and 1 << (items - 1).bit_length()
    partial = torch.empty((padded,), device=input.device, dtype=torch.float32)
    _activate_chunk_neutral207.run(
        input,
        activation,
        partial,
        num_tokens_tensor,
        width,
        input.stride(0),
        items,
        padded,
        num_warps=1,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    _quant_chunk_neutral207.run(
        activation,
        result_bytes,
        partial,
        scale,
        num_tokens_tensor,
        width,
        result_bytes.stride(0),
        padded,
        num_warps=1,
        enable_fp_fusion=False,
        grid=(programs,),
        warmup=False,
    )
    return (output, scale)


RAW_ACTIVATION_CHUNK_NEUTRAL207 = (
    "\n"
    "typedef unsigned short __attribute__((address_space(1))) gu16;\n"
    "typedef float __attribute__((address_space(1))) gf32;\n"
    "typedef float f32v __attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v __attribute__((ext_vector_type(16)));\n"
    "union bits_activate207 { f32v f; u32v u; };\n"
    "\n"
    "typedef bool b32v __attribute__((ext_vector_type(32)));\n"
    'extern "C" __attribute__((device)) b32v compare_inf207(float, f32'
    'v) __asm__("llvm.xpu.svneq_f_mr1");\n'
    'extern "C" __attribute__((device)) f32v masked_mul207(b32v, f32v,'
    ' f32v) __asm__("llvm.xpu.vvmul_f_mz_rn");\n'
    "\n"
    "__attribute__((device,always_inline)) f32v reciprocal_seed207(f32"
    "v denominator) {\n"
    "    bits_activate207 input, estimate;\n"
    "    input.f = denominator;\n"
    "    estimate.u = (u32v)0x5f3759dfu - (input.u >> 1);\n"
    "    b32v finite_mask = compare_inf207(__builtin_inff(), denominat"
    "or);\n"
    "    return masked_mul207(finite_mask, estimate.f, estimate.f);\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v mac_activate207(f32v a"
    ", f32v b, f32v c) {\n"
    '    __asm__("vmac.f.mz.rn %0{mr1}, %1, %2" : "+v"(c) : "v"(a), "v'
    '"(b));\n'
    "    return c;\n"
    "}\n"
    "\n"
    "__attribute__((device,always_inline)) f32v swiglu207(u32v gate_bi"
    "ts, u32v up_bits) {\n"
    "    bits_activate207 gate, up;\n"
    "    gate.u = gate_bits;\n"
    "    up.u = up_bits;\n"
    "    f32v exponential = __builtin_elementwise_exp2(-gate.f);\n"
    "    f32v denominator = (f32v)1.0f + exponential;\n"
    "    f32v reciprocal = reciprocal_seed207(denominator);\n"
    "    f32v safe_denominator = __builtin_elementwise_min(denominator"
    ", (f32v)0x1.fffffep127f);\n"
    "    f32v residual = mac_activate207(-safe_denominator, reciprocal"
    ", (f32v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    residual = mac_activate207(-safe_denominator, reciprocal, (f3"
    "2v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    residual = mac_activate207(-safe_denominator, reciprocal, (f3"
    "2v)1.0f);\n"
    "    reciprocal = mac_activate207(residual, reciprocal, reciprocal"
    ");\n"
    "    return gate.f * reciprocal * up.f;\n"
    "}\n"
    "\n"
    "\n"
    'extern "C" __attribute__((device)) void _raw_activate_chunk_neutr'
    "al207(\n"
    "    const gu16* X, gf32* A, gf32* P, int n, long long D, long lon"
    "g SX, int pid, long long count, long long padded) {\n"
    "    unsigned int gate_input[64] __attribute__((aligned(64)));\n"
    "    unsigned int up_input[64] __attribute__((aligned(64)));\n"
    "    float output[128] __attribute__((aligned(64)));\n"
    "    float partial[1] __attribute__((aligned(64)));\n"
    "    f32v maximum = (f32v)0.0f;\n"
    "    long long chunks = D / 128;\n"
    "    long long item = (long long)pid * 64 + __builtin_xpu_core_id("
    ");\n"
    "    long long live_items = (long long)n * chunks;\n"
    "    partial[0] = 0.0f;\n"
    "    if (item + count < padded) {\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(P + item + count, partial, 0, 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "    }\n"
    "    if (item >= live_items) {\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        __builtin_xpu_lm2gm_v3(P + item, partial, 0, 4);\n"
    "        __builtin_xpu2_mfence(7);\n"
    "        return;\n"
    "    }\n"
    "    long long row = item / chunks;\n"
    "    long long column = (item % chunks) * 128;\n"
    "    __builtin_xpu_gm2lm_v3(gate_input, X + row * SX + column, 0, "
    "256);\n"
    "    __builtin_xpu_gm2lm_v3(up_input, X + row * SX + D + column, 0"
    ", 256);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    for (int offset = 0; offset < 128; offset += 32) {\n"
    "        u32v gate = *((u32v*)(gate_input + offset / 2));\n"
    "        u32v up = *((u32v*)(up_input + offset / 2));\n"
    "        bits_activate207 even, odd, absolute_even, absolute_odd;\n"
    "        even.f = swiglu207(gate << 16, up << 16);\n"
    "        odd.f = swiglu207(gate & 0xffff0000u, up & 0xffff0000u);\n"
    "        *((u32v*)(output + offset)) = __builtin_xpu2_vmerge_l_u(e"
    "ven.u, odd.u);\n"
    "        *((u32v*)(output + offset + 16)) = __builtin_xpu2_vmerge_"
    "h_u(even.u, odd.u);\n"
    "        absolute_even.u = even.u & 0x7fffffffu;\n"
    "        absolute_odd.u = odd.u & 0x7fffffffu;\n"
    "        maximum = __builtin_elementwise_max(maximum,\n"
    "                   __builtin_elementwise_max(absolute_even.f, abs"
    "olute_odd.f));\n"
    "    }\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(A + row * D + column, output, 0, 512);"
    "\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    partial[0] = __builtin_reduce_max(maximum);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "    __builtin_xpu_lm2gm_v3(P + item, partial, 0, 4);\n"
    "    __builtin_xpu2_mfence(7);\n"
    "}\n"
)


@tle.raw.dialect(
    "xpu3",
    source=RAW_ACTIVATION_CHUNK_NEUTRAL207,
    flags=("-ffp-contract=off",),
)
def _raw_activate_chunk_neutral207(X, A, P, n, D, SX, pid, count, padded):
    pass


@triton.jit
def _activate_chunk_neutral207(
    X,
    A,
    P,
    NT,
    D: tl.constexpr,
    SX: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    n = tl.load(NT)
    tle.raw.call(
        _raw_activate_chunk_neutral207,
        (
            X.to(tl.pointer_type(tl.uint16)),
            A,
            P,
            n,
            tl.full((), D, tl.int64),
            tl.full((), SX, tl.int64),
            tl.program_id(0),
            tl.full((), K, tl.int64),
            tl.full((), B, tl.int64),
        ),
    )


@triton.jit
def _quant_chunk_neutral207(
    A, Q, P, S, NT, D: tl.constexpr, SQ: tl.constexpr, B: tl.constexpr
):
    n = tl.load(NT)
    lane = tl.arange(0, B)
    maximum = tl.load(P + lane)
    scale = tl.maximum(tl.max(maximum, 0), 1e-10) / 448.0
    if tl.program_id(0) == 0:
        tl.store(S, scale)
    tle.raw.call(
        _raw_quant_chunk207,
        (
            A,
            Q,
            scale,
            n,
            tl.full((), D, tl.int64),
            tl.full((), SQ, tl.int64),
            tl.program_id(0),
        ),
    )


__all__ = ["silu_mul_dynamic_tensorwise_quant"]
