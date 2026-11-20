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
import triton.experimental.tle as tle
import triton.language as tl


@triton.jit
def _fused_qk_complex_rope_strided_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_out_ptr,
    k_out_ptr,
    NUM_PAIRS: tl.constexpr,
    SHAPE: tl.constexpr,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    FREQ_STRIDES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    program_id = tl.program_id(0)
    if USE_INT64:
        program_id = program_id.to(tl.int64)
    x = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dim_offsets = x % (SHAPE[-1] // 2)
    head_index = x // (SHAPE[-1] // 2) % SHAPE[-2]
    token_index = x // (SHAPE[-1] // 2 * SHAPE[-2])
    q_offsets = head_index * Q_STRIDES[-2] + 2 * dim_offsets * Q_STRIDES[-1]
    k_offsets = head_index * K_STRIDES[-2] + 2 * dim_offsets * K_STRIDES[-1]
    freq_offset = dim_offsets * FREQ_STRIDES[-2]
    for axis in tl.static_range(len(SHAPE) - 3, -1, -1):
        coord = token_index % SHAPE[axis]
        token_index = token_index // SHAPE[axis]
        q_offsets += coord * Q_STRIDES[axis]
        k_offsets += coord * K_STRIDES[axis]
        freq_offset += coord * FREQ_STRIDES[axis]
    q_real = tl.load(q_ptr + q_offsets, x < NUM_PAIRS, 0).to(tl.float32)
    q_imag = tl.load(q_ptr + q_offsets + Q_STRIDES[-1], x < NUM_PAIRS, 0).to(
        tl.float32
    )
    k_real = tl.load(k_ptr + k_offsets, x < NUM_PAIRS, 0).to(tl.float32)
    k_imag = tl.load(k_ptr + k_offsets + K_STRIDES[-1], x < NUM_PAIRS, 0).to(
        tl.float32
    )
    freq_real = tl.load(freqs_ptr + freq_offset, x < NUM_PAIRS, 0)
    freq_imag = tl.load(
        freqs_ptr + freq_offset + FREQ_STRIDES[-1], x < NUM_PAIRS, 0
    )
    q_out_real = tl.fma(q_real, freq_real, 0.0) - tl.fma(
        q_imag, freq_imag, 0.0
    )
    q_out_imag = tl.fma(q_imag, freq_real, 0.0) + tl.fma(
        q_real, freq_imag, 0.0
    )
    k_out_real = tl.fma(k_real, freq_real, 0.0) - tl.fma(
        k_imag, freq_imag, 0.0
    )
    k_out_imag = tl.fma(k_imag, freq_real, 0.0) + tl.fma(
        k_real, freq_imag, 0.0
    )
    tl.store(q_out_ptr + 2 * x, q_out_real, x < NUM_PAIRS)
    tl.store(q_out_ptr + 2 * x + 1, q_out_imag, x < NUM_PAIRS)
    tl.store(k_out_ptr + 2 * x, k_out_real, x < NUM_PAIRS)
    tl.store(k_out_ptr + 2 * x + 1, k_out_imag, x < NUM_PAIRS)


@triton.jit
def _fused_qk_complex_rope_to_bf16_bits(value):
    return (
        value.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    )


@triton.jit
def _fused_qk_complex_rope_packed_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_out_ptr,
    k_out_ptr,
    SHAPE: tl.constexpr,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    FREQ_STRIDES: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    IS_BF16: tl.constexpr,
    FAST: tl.constexpr,
):
    program_id = tl.program_id(0)
    row = program_id * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    head_index = row % SHAPE[-2]
    token_index = row // SHAPE[-2]
    q_base = head_index * (Q_STRIDES[-2] // 2)
    k_base = head_index * (K_STRIDES[-2] // 2)
    freq_base = tl.full((BLOCK_ROWS,), 0, tl.int32)
    for axis in tl.static_range(len(SHAPE) - 3, -1, -1):
        coord = token_index % SHAPE[axis]
        token_index = token_index // SHAPE[axis]
        q_base += coord * (Q_STRIDES[axis] // 2)
        k_base += coord * (K_STRIDES[axis] // 2)
        freq_base += coord * FREQ_STRIDES[axis]
    if FAST:
        q_base = row * (SHAPE[-1] // 2)
        k_base = row * (SHAPE[-1] // 2)
    pair_index = tl.arange(0, BLOCK_PAIRS)
    mask = (row[:, None] < NUM_ROWS) & (pair_index[None, :] < SHAPE[-1] // 2)
    if FAST and NUM_ROWS % BLOCK_ROWS == 0 and (BLOCK_PAIRS == SHAPE[-1] // 2):
        mask = tl.full((BLOCK_ROWS, BLOCK_PAIRS), True, tl.int1)
    q_bits = tl.load(
        q_ptr.to(tl.pointer_type(tl.uint32))
        + q_base[:, None]
        + pair_index[None, :],
        mask,
        0,
    )
    k_bits = tl.load(
        k_ptr.to(tl.pointer_type(tl.uint32))
        + k_base[:, None]
        + pair_index[None, :],
        mask,
        0,
    )
    if IS_BF16:
        q_real = ((q_bits & 65535) << 16).to(tl.float32, bitcast=True)
        q_imag = (q_bits & 4294901760).to(tl.float32, bitcast=True)
        k_real = ((k_bits & 65535) << 16).to(tl.float32, bitcast=True)
        k_imag = (k_bits & 4294901760).to(tl.float32, bitcast=True)
    else:
        q_real = (
            q_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        )
        q_imag = (
            (q_bits >> 16)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        k_real = (
            k_bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
        )
        k_imag = (
            (k_bits >> 16)
            .to(tl.uint16)
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
    if SHAPE[-2] % BLOCK_ROWS == 0:
        freq_token = program_id * BLOCK_ROWS // SHAPE[-2]
        freq_base_offset = 0
        for axis in tl.static_range(len(SHAPE) - 3, -1, -1):
            coord = freq_token % SHAPE[axis]
            freq_token = freq_token // SHAPE[axis]
            freq_base_offset += coord * FREQ_STRIDES[axis]
        fbits = tl.load(
            freqs_ptr.to(tl.pointer_type(tl.uint64))
            + freq_base_offset // 2
            + pair_index * (FREQ_STRIDES[-2] // 2),
            pair_index < SHAPE[-1] // 2,
            0,
        )
        freq_real = fbits.to(tl.uint32).to(tl.float32, bitcast=True)[None, :]
        freq_imag = (
            (fbits >> 32).to(tl.uint32).to(tl.float32, bitcast=True)[None, :]
        )
    else:
        fbits = tl.load(
            freqs_ptr.to(tl.pointer_type(tl.uint64))
            + freq_base[:, None] // 2
            + pair_index[None, :] * (FREQ_STRIDES[-2] // 2),
            mask,
            0,
        )
        freq_real = fbits.to(tl.uint32).to(tl.float32, bitcast=True)
        freq_imag = (fbits >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    q_out_real = tl.fma(q_real, freq_real, 0.0) - tl.fma(
        q_imag, freq_imag, 0.0
    )
    q_out_imag = tl.fma(q_imag, freq_real, 0.0) + tl.fma(
        q_real, freq_imag, 0.0
    )
    k_out_real = tl.fma(k_real, freq_real, 0.0) - tl.fma(
        k_imag, freq_imag, 0.0
    )
    k_out_imag = tl.fma(k_imag, freq_real, 0.0) + tl.fma(
        k_real, freq_imag, 0.0
    )
    if IS_BF16:
        q_out_values = (
            _fused_qk_complex_rope_to_bf16_bits(q_out_real)
            | _fused_qk_complex_rope_to_bf16_bits(q_out_imag) << 16
        )
        k_out_values = (
            _fused_qk_complex_rope_to_bf16_bits(k_out_real)
            | _fused_qk_complex_rope_to_bf16_bits(k_out_imag) << 16
        )
    else:
        q_out_values = (
            q_out_real.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
            | q_out_imag.to(tl.float16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
            << 16
        )
        k_out_values = (
            k_out_real.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
            | k_out_imag.to(tl.float16)
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
            << 16
        )
    tl.store(
        q_out_ptr.to(tl.pointer_type(tl.uint32))
        + row[:, None] * (SHAPE[-1] // 2)
        + pair_index[None, :],
        q_out_values,
        mask,
    )
    tl.store(
        k_out_ptr.to(tl.pointer_type(tl.uint32))
        + row[:, None] * (SHAPE[-1] // 2)
        + pair_index[None, :],
        k_out_values,
        mask,
    )


@triton.jit
def _fused_qk_complex_rope_tokens_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_out_ptr,
    k_out_ptr,
    NUM_TOKENS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    NUM_CHANNEL_PAIRS: tl.constexpr,
):
    program_id = tl.program_id(0)
    row = program_id * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    pair_index = tl.arange(0, NUM_CHANNEL_PAIRS)
    position = row[:, None] * NUM_CHANNEL_PAIRS + pair_index[None, :]
    NUM_ROWS: tl.constexpr = (
        NUM_TOKENS * NUM_HEADS * HEAD_DIM // (2 * NUM_CHANNEL_PAIRS)
    )
    if NUM_ROWS % BLOCK_TOKENS == 0:
        mask = tl.full((BLOCK_TOKENS, NUM_CHANNEL_PAIRS), True, tl.int1)
    else:
        mask = tl.broadcast_to(
            row[:, None] < NUM_ROWS, (BLOCK_TOKENS, NUM_CHANNEL_PAIRS)
        )
    q_bits = tl.load(q_ptr.to(tl.pointer_type(tl.uint32)) + position, mask, 0)
    k_bits = tl.load(k_ptr.to(tl.pointer_type(tl.uint32)) + position, mask, 0)
    q_real = ((q_bits & 65535) << 16).to(tl.float32, bitcast=True)
    q_imag = (q_bits & 4294901760).to(tl.float32, bitcast=True)
    k_real = ((k_bits & 65535) << 16).to(tl.float32, bitcast=True)
    k_imag = (k_bits & 4294901760).to(tl.float32, bitcast=True)
    fpos = position // (NUM_HEADS * HEAD_DIM // 2) * (
        HEAD_DIM // 2
    ) + position % (HEAD_DIM // 2)
    fbits = tl.load(freqs_ptr.to(tl.pointer_type(tl.uint64)) + fpos, mask, 0)
    freq_real = fbits.to(tl.uint32).to(tl.float32, bitcast=True)
    freq_imag = (fbits >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
    q_out_real = tl.fma(q_real, freq_real, 0.0) - tl.fma(
        q_imag, freq_imag, 0.0
    )
    q_out_imag = tl.fma(q_imag, freq_real, 0.0) + tl.fma(
        q_real, freq_imag, 0.0
    )
    k_out_real = tl.fma(k_real, freq_real, 0.0) - tl.fma(
        k_imag, freq_imag, 0.0
    )
    k_out_imag = tl.fma(k_imag, freq_real, 0.0) + tl.fma(
        k_real, freq_imag, 0.0
    )
    q_out_values = (
        _fused_qk_complex_rope_to_bf16_bits(q_out_real)
        | _fused_qk_complex_rope_to_bf16_bits(q_out_imag) << 16
    )
    k_out_values = (
        _fused_qk_complex_rope_to_bf16_bits(k_out_real)
        | _fused_qk_complex_rope_to_bf16_bits(k_out_imag) << 16
    )
    tl.store(
        q_out_ptr.to(tl.pointer_type(tl.uint32)) + position, q_out_values, mask
    )
    tl.store(
        k_out_ptr.to(tl.pointer_type(tl.uint32)) + position, k_out_values, mask
    )


def _fused_qk_complex_rope_compute_general(
    q,
    k,
    freqs_cis,
):
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    num_elements = q.numel() // 2
    if num_elements:
        freqs_real = torch.view_as_real(freqs_cis)
        use_int64 = q.numel() + 64 * q.shape[-1] > 2147483647
        if not use_int64 and (
            not (
                q.is_contiguous()
                and k.is_contiguous()
                and freqs_cis.is_contiguous()
            )
        ):
            use_int64 = (
                max(
                    (
                        sum(
                            (
                                (rotation_sign - 1) * stride
                                for rotation_sign, stride in zip(
                                    x.shape, x.stride()
                                )
                            )
                        )
                        for x in (q, k, freqs_real)
                    )
                )
                > 2147483647
            )
        num_heads, head_dim = q.shape[-2:]
        if (
            not use_int64
            and q.dtype == torch.bfloat16
            and q.is_contiguous()
            and k.is_contiguous()
            and freqs_cis.is_contiguous()
            and (q.storage_offset() % 2 == k.storage_offset() % 2 == 0)
            and (num_heads & num_heads - 1 == 0)
            and (head_dim & head_dim - 1 == 0)
            and (num_heads * head_dim <= 2048)
            and (q.numel() // (num_heads * head_dim) >= 1024)
        ):
            num_tokens = q.numel() // (num_heads * head_dim)
            num_channel_pairs = min(256, num_heads * head_dim // 2)
            _fused_qk_complex_rope_tokens_kernel.run(
                q,
                k,
                freqs_real,
                q_out,
                k_out,
                num_tokens,
                num_heads,
                head_dim,
                64,
                num_channel_pairs,
                grid=((num_elements // num_channel_pairs + 64 - 1) // 64,),
                warmup=False,
                groups_per_cluster=64,
                isAutoCoreTiling=False,
                num_warps=4,
                enable_fp_fusion=False,
                isCloseDtypeConvert=q.dtype == torch.bfloat16,
            )
            return (q_out, k_out)
        packed = (
            q.dtype == k.dtype
            and q.dtype in (torch.bfloat16, torch.float16)
            and not use_int64
            and q.numel() >= 2048
            and (q.stride(-1) == k.stride(-1) == 1)
            and all(
                (
                    rotation_sign % 2 == 0
                    for rotation_sign in q.stride()[:-1] + k.stride()[:-1]
                )
            )
            and (q.storage_offset() % 2 == k.storage_offset() % 2 == 0)
        )
        if packed:
            block_rows = 64 if q.shape[-1] & q.shape[-1] - 1 == 0 else 4
            num_rows = q.numel() // q.shape[-1]
            block_pairs = triton.next_power_of_2(q.shape[-1] // 2)
            if block_rows * block_pairs <= 4096:
                _fused_qk_complex_rope_packed_kernel[
                    (num_rows + block_rows - 1) // block_rows,
                ](
                    q,
                    k,
                    freqs_real,
                    q_out,
                    k_out,
                    tuple(q.shape),
                    q.stride(),
                    k.stride(),
                    freqs_real.stride(),
                    num_rows,
                    block_rows,
                    block_pairs,
                    q.dtype == torch.bfloat16,
                    q.is_contiguous() and k.is_contiguous(),
                    groups_per_cluster=64 if block_rows == 64 else 1,
                    num_warps=4,
                    enable_fp_fusion=False,
                    isCloseDtypeConvert=q.dtype == torch.bfloat16,
                )
                return (q_out, k_out)
        _fused_qk_complex_rope_strided_kernel[
            (num_elements + 512 - 1) // 512,
        ](
            q,
            k,
            freqs_real,
            q_out,
            k_out,
            num_elements,
            tuple(q.shape),
            q.stride(),
            k.stride(),
            freqs_real.stride(),
            512,
            use_int64,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (q_out, k_out)


_RAW_ROWS_SOURCE = (
    "typedef unsigned short "
    "__attribute__((address_space(1))) gu16;\n"
    "typedef float "
    "__attribute__((address_space(1))) gf32;\n"
    "typedef float f32v "
    "__attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v "
    "__attribute__((ext_vector_type(16)));\nunion "
    "bits200 { f32v f; u32v u; };\n\n"
    "__attribute__((device,always_inline)) u32v "
    "encode200(f32v value) {\n    bits200 b;\n    "
    "b.f = value;\n    u32v original = b.u;\n    "
    "u32v word = (original + 32768u) >> 16;\n    "
    'return word;\n}\n\nextern "C" '
    "__attribute__((device)) void "
    "_device_rope200(\n    gu16* Q, gu16* K, gf32*"
    " F, gu16* OQ, gu16* OK,\n    int T, int H, "
    "int COPY, int pid, int np) {\n    unsigned "
    "int qb[256] __attribute__((aligned(64)));"
    "\n    unsigned int kb[256] "
    "__attribute__((aligned(64)));\n    float "
    "fb[128] __attribute__((aligned(64)));\n    "
    "float fc[64] __attribute__((aligned(64)));"
    "\n    float fs[64] "
    "__attribute__((aligned(64)));\n    const int "
    "core = __builtin_xpu_core_id();\n    const "
    "int groups = H / 4;\n    const int blocks = T"
    " * groups;\n    for (int block = pid * 64 + "
    "core; block < blocks; block += np * 64) {"
    "\n        const int offset = block * 512;"
    "\n        __builtin_xpu_gm2lm_v3(qb, Q + "
    "offset, 0, 1024);\n        "
    "__builtin_xpu_gm2lm_v3(kb, K + offset, 0, "
    "1024);\n        if (!COPY) {\n            "
    "const int token = block / groups;"
    "\n            __builtin_xpu_gm2lm_v3(fb, F + "
    "token * 128, 0, 512);\n        }\n        "
    "__builtin_xpu2_mfence(7);\n        if (!COPY)"
    " {\n            #pragma clang loop "
    "vectorize(disable)\n            for (int pair"
    " = 0; pair < 64; ++pair) {\n                "
    "fc[pair] = fb[pair * 2];\n                "
    "fs[pair] = fb[pair * 2 + 1];\n            }"
    "\n            for (int head = 0; head < 4; "
    "++head) {\n                for (int pair = 0;"
    " pair < 64; pair += 16) {"
    "\n                    const int index = head "
    "* 64 + pair;\n                    f32v c = "
    "*((f32v*)(fc + pair));\n                    "
    "f32v s = *((f32v*)(fs + pair));"
    "\n                    u32v q = *((u32v*)(qb +"
    " index));\n                    bits200 qr, "
    "qi;\n                    qr.u = q << 16;"
    "\n                    qi.u = q & 0xffff0000u;"
    "\n                    f32v qa = qr.f * c;"
    "\n                    f32v qbmul = qi.f * s;"
    "\n                    f32v qc = qi.f * c;"
    "\n                    f32v qd = qr.f * s;"
    "\n                    *((u32v*)(qb + index)) "
    "= encode200(qa - qbmul) | (encode200(qc + "
    "qd) << 16);\n                    u32v k = "
    "*((u32v*)(kb + index));\n                    "
    "bits200 kr, ki;\n                    kr.u = k"
    " << 16;\n                    ki.u = k & "
    "0xffff0000u;\n                    f32v ka = "
    "kr.f * c;\n                    f32v kbmul = "
    "ki.f * s;\n                    f32v kc = ki.f"
    " * c;\n                    f32v kd = kr.f * "
    "s;\n                    *((u32v*)(kb + "
    "index)) = encode200(ka - kbmul) | "
    "(encode200(kc + kd) << 16);\n                "
    "}\n            }\n        }\n        "
    "__builtin_xpu2_mfence(7);\n        "
    "__builtin_xpu_lm2gm_v3(OQ + offset, qb, 0, "
    "1024);\n        __builtin_xpu_lm2gm_v3(OK + "
    "offset, kb, 0, 1024);\n        "
    "__builtin_xpu2_mfence(7);\n    }\n}\n"
)


@tle.raw.dialect("xpu3", source=_RAW_ROWS_SOURCE, flags=("-ffp-contract=off",))
def _device_rope200(
    Q,
    K,
    F,
    OQ,
    OK,
    T,
    H,
    COPY,
    pid,
    np,
):
    pass


@triton.jit(
    do_not_specialize=[
        "q_ptr",
        "k_ptr",
        "freqs_address",
        "q_out_ptr",
        "k_out_ptr",
    ]
)
def _fused_qk_complex_rope_raw_rows_kernel(
    q_ptr,
    k_ptr,
    freqs_address: "u64",  # noqa: F821
    q_out_ptr,
    k_out_ptr,
    METADATA: tl.constexpr,
):
    NUM_TOKENS: tl.constexpr = METADATA[0]
    NUM_HEADS: tl.constexpr = METADATA[1]
    COPY: tl.constexpr = METADATA[2]
    tle.raw.call(
        _device_rope200,
        (
            q_ptr,
            k_ptr,
            freqs_address.to(tl.pointer_type(tl.float32)),
            q_out_ptr,
            k_out_ptr,
            tl.full((), NUM_TOKENS, tl.int32),
            tl.full((), NUM_HEADS, tl.int32),
            tl.full((), COPY, tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


def _fused_qk_complex_rope_launch_raw(
    q,
    k,
    freqs_cis,
    copy=False,
):
    num_heads, head_dim = q.shape[-2:]
    num_tokens = q.numel() // (num_heads * head_dim)
    q_storage_address = q.untyped_storage().data_ptr()
    k_storage_address = k.untyped_storage().data_ptr()
    freq_storage_address = freqs_cis.untyped_storage().data_ptr()
    inplace = (
        q_storage_address != k_storage_address
        and q_storage_address != freq_storage_address
        and k_storage_address != freq_storage_address
    )
    q_out = (
        q if inplace else torch.empty(q.shape, dtype=q.dtype, device=q.device)
    )
    k_out = (
        k if inplace else torch.empty(k.shape, dtype=k.dtype, device=k.device)
    )
    freqs_address = freqs_cis.data_ptr()
    num_programs = min(12, (num_tokens * (num_heads // 4) + 64 - 1) // 64)
    _fused_qk_complex_rope_raw_rows_kernel.run(
        q,
        k,
        freqs_address,
        q_out,
        k_out,
        (num_tokens, num_heads, int(copy)),
        grid=(num_programs,),
        warmup=False,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return (q_out, k_out)


def _fused_qk_complex_rope_can_use_raw(
    q,
    k,
    freqs_cis,
):
    return (
        q.ndim >= 3
        and q.dtype == k.dtype == torch.bfloat16
        and (q.shape[-1] == 128)
        and (q.shape[-2] % 4 == 0)
        and q.is_contiguous()
        and k.is_contiguous()
        and freqs_cis.is_contiguous()
        and (q.numel() >= 2097152 or (q.shape[-2] == 16 and q.numel() >= 2048))
        and (q.numel() + 1024 < 2147483648)
        and ((q.data_ptr() | k.data_ptr() | freqs_cis.data_ptr()) & 63 == 0)
    )


def _fused_qk_complex_rope_dispatch_raw(
    q,
    k,
    freqs_cis,
):
    if _fused_qk_complex_rope_can_use_raw(q, k, freqs_cis):
        return _fused_qk_complex_rope_launch_raw(q, k, freqs_cis)
    return _fused_qk_complex_rope_compute_general(q, k, freqs_cis)


_RAW_TOKEN_SOURCE = (
    "typedef unsigned short "
    "__attribute__((address_space(1))) gu16;\n"
    "typedef float "
    "__attribute__((address_space(1))) gf32;\n"
    "typedef float f32v "
    "__attribute__((ext_vector_type(16)));\n"
    "typedef unsigned int u32v "
    "__attribute__((ext_vector_type(16)));\n"
    "typedef int i32v "
    "__attribute__((ext_vector_type(16)));\nunion "
    "bits200 { f32v f; u32v u; };\n\n"
    "__attribute__((device,always_inline)) u32v "
    "encode200_token(f32v value) {\n    bits200 b;"
    "\n    b.f = value;\n    u32v original = b.u;"
    "\n    u32v word = (original + 32768u) >> 16;"
    '\n    return word;\n}\n\n\nextern "C" '
    "__attribute__((device)) void "
    "_device_rope200_token(\n    gu16* Q, gu16* K,"
    " gf32* F, gu16* OQ, gu16* OK,\n    int T, int"
    " H, int COPY, int pid, int np) {\n    "
    "unsigned int buffer[1024] "
    "__attribute__((aligned(64)));\n    float "
    "fb[128] __attribute__((aligned(64)));\n    "
    "float fc[64] __attribute__((aligned(64)));"
    "\n    float fs[64] "
    "__attribute__((aligned(64)));\n    const int "
    "core = __builtin_xpu_core_id();\n    const "
    "int groups = H / 16;\n    const int "
    "first=pid*64+core;\n    if (!COPY && "
    "first<T*groups) {\n        __builtin_xpu_gm2l"
    "m_v3(fb,F+(first/groups)*128,0,512);"
    "\n        __builtin_xpu2_mfence(7);\n        "
    "const i32v offsets=(i32v){0,8,16,24,32,40,48"
    ",56,64,72,80,88,96,104,112,120};\n        for"
    " (int pair=0;pair<64;pair+=16) {"
    "\n            bits200 c,s;\n            c.u=re"
    "interpret_cast<u32v>(__builtin_xpu2_vgather_"
    "mask16_mr1((const unsigned "
    "int*)(fb+pair*2),offsets));\n            s.u="
    "reinterpret_cast<u32v>(__builtin_xpu2_vgathe"
    "r_mask16_mr1((const unsigned "
    "int*)(fb+pair*2),offsets+4));\n            "
    "*((f32v*)(fc+pair))=c.f;\n            "
    "*((f32v*)(fs+pair))=s.f;\n        }\n    }"
    "\n    for (int block=first; block<T*groups; "
    "block+=np*64) {\n        const int "
    "offset=block*2048;\n        #pragma clang "
    "loop unroll(disable)\n        for (int "
    "which=0;which<2;++which) {\n            gu16*"
    " src=which?K:Q;\n            gu16* "
    "dst=which?OK:OQ;\n            __builtin_xpu_g"
    "m2lm_v3(buffer,src+offset,0,4096);"
    "\n            __builtin_xpu2_mfence(7);"
    "\n            if (!COPY && which==0 && "
    "block+np*64<T*groups) {\n                __bu"
    "iltin_xpu_gm2lm_v3(fb,F+((block+np*64)/group"
    "s)*128,0,512);\n            }\n            if "
    "(!COPY) {\n                #pragma clang loop"
    " unroll_count(2)\n                for (int "
    "head=0;head<16;++head) {"
    "\n                    for (int "
    "pair=0;pair<64;pair+=16) {"
    "\n                        const int "
    "index=head*64+pair;\n                        "
    "f32v c=*((f32v*)(fc+pair));"
    "\n                        f32v "
    "s=*((f32v*)(fs+pair));"
    "\n                        u32v "
    "q=*((u32v*)(buffer+index));"
    "\n                        bits200 qr,qi;"
    "\n                        qr.u=q<<16;"
    "\n                        qi.u=q&0xffff0000u;"
    "\n                        f32v a=qr.f*c;"
    "\n                        f32v b=qi.f*s;"
    "\n                        f32v cprod=qi.f*c;"
    "\n                        f32v d=qr.f*s;"
    "\n                        *((u32v*)(buffer+in"
    "dex))=encode200_token(a-b)|(encode200_token("
    "cprod+d)<<16);\n                    }"
    "\n                }\n            }"
    "\n            __builtin_xpu2_mfence(7);"
    "\n            __builtin_xpu_lm2gm_v3(dst+offs"
    "et,buffer,0,4096);\n            if (!COPY && "
    "which==1 && block+np*64<T*groups) {"
    "\n                const i32v offsets=(i32v){0"
    ",8,16,24,32,40,48,56,64,72,80,88,96,104,112,"
    "120};\n                for (int "
    "pair=0;pair<64;pair+=16) {"
    "\n                    bits200 c,s;"
    "\n                    c.u=reinterpret_cast<u3"
    "2v>(__builtin_xpu2_vgather_mask16_mr1((const"
    " unsigned int*)(fb+pair*2),offsets));"
    "\n                    s.u=reinterpret_cast<u3"
    "2v>(__builtin_xpu2_vgather_mask16_mr1((const"
    " unsigned int*)(fb+pair*2),offsets+4));"
    "\n                    "
    "*((f32v*)(fc+pair))=c.f;"
    "\n                    "
    "*((f32v*)(fs+pair))=s.f;\n                }"
    "\n            }\n            "
    "__builtin_xpu2_mfence(7);\n        }\n    }\n}\n"
)


@tle.raw.dialect(
    "xpu3", source=_RAW_TOKEN_SOURCE, flags=("-ffp-contract=off",)
)
def _device_rope200_token(
    Q,
    K,
    F,
    OQ,
    OK,
    T,
    H,
    COPY,
    pid,
    np,
):
    pass


@triton.jit(
    do_not_specialize=[
        "q_ptr",
        "k_ptr",
        "freqs_address",
        "q_out_ptr",
        "k_out_ptr",
    ]
)
def _fused_qk_complex_rope_raw_tokens_kernel(
    q_ptr,
    k_ptr,
    freqs_address: "u64",  # noqa: F821
    q_out_ptr,
    k_out_ptr,
    METADATA: tl.constexpr,
):
    NUM_TOKENS: tl.constexpr = METADATA[0]
    NUM_HEADS: tl.constexpr = METADATA[1]
    COPY: tl.constexpr = METADATA[2]
    tle.raw.call(
        _device_rope200_token,
        (
            q_ptr,
            k_ptr,
            freqs_address.to(tl.pointer_type(tl.float32)),
            q_out_ptr,
            k_out_ptr,
            tl.full((), NUM_TOKENS, tl.int32),
            tl.full((), NUM_HEADS, tl.int32),
            tl.full((), COPY, tl.int32),
            tl.program_id(0),
            tl.num_programs(0),
        ),
    )


def _fused_qk_complex_rope_launch_raw_tokens(
    q,
    k,
    freqs_cis,
    copy=False,
):
    num_heads, head_dim = q.shape[-2:]
    num_tokens = q.numel() // (num_heads * head_dim)
    q_storage_address = q.untyped_storage().data_ptr()
    k_storage_address = k.untyped_storage().data_ptr()
    freq_storage_address = freqs_cis.untyped_storage().data_ptr()
    inplace = (
        q_storage_address != k_storage_address
        and q_storage_address != freq_storage_address
        and k_storage_address != freq_storage_address
    )
    q_out = (
        q if inplace else torch.empty(q.shape, dtype=q.dtype, device=q.device)
    )
    k_out = (
        k if inplace else torch.empty(k.shape, dtype=k.dtype, device=k.device)
    )
    freqs_address = freqs_cis.data_ptr()
    num_programs = min(12, (num_tokens * (num_heads // 16) + 64 - 1) // 64)
    _fused_qk_complex_rope_raw_tokens_kernel.run(
        q,
        k,
        freqs_address,
        q_out,
        k_out,
        (num_tokens, num_heads, int(copy)),
        grid=(num_programs,),
        warmup=False,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return (q_out, k_out)


def _fused_qk_complex_rope_dispatch_raw_tokens(
    q,
    k,
    freqs_cis,
):
    if _fused_qk_complex_rope_can_use_raw(q, k, freqs_cis):
        if (
            q.shape[-2] == 16
            and q.numel() >= 2097152
            and (q.numel() + 4096 < 2147483648)
        ):
            return _fused_qk_complex_rope_launch_raw_tokens(q, k, freqs_cis)
        return _fused_qk_complex_rope_launch_raw(q, k, freqs_cis)
    return _fused_qk_complex_rope_compute_general(q, k, freqs_cis)


def fused_qk_complex_rope(
    q,
    k,
    freqs_cis,
):
    return _fused_qk_complex_rope_dispatch_raw_tokens(q, k, freqs_cis)


__all__ = ["fused_qk_complex_rope"]
