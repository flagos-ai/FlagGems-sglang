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

_INV = 1.0 / 448.0


@triton.jit
def encode_legacy_S(value):
    af = tl.minimum(tl.abs(value), 448.0)
    ab = af.to(tl.uint32, bitcast=True)
    normal = (ab + 524287 + (ab >> 20 & 1) >> 20) - 960
    sub = (af + 16384.0).to(tl.uint32, bitcast=True) & 15
    code = tl.where(af >= 0.015625, normal, sub)
    code = tl.where(value != value, 127, code)
    return (code | value.to(tl.uint32, bitcast=True) >> 24 & 128).to(tl.uint8)


@triton.jit
def _tight_S(
    X,
    P,
    Q,
    output_ptr,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    RDIM: tl.constexpr,
    QROW: tl.constexpr,
    DIM: tl.constexpr,
    INV: tl.constexpr,
):
    g = tl.arange(0, GROUP)
    gc = tl.arange(0, GROUPS)
    rc = tl.arange(0, RDIM)
    b = tl.arange(0, 4)
    h = tl.arange(0, 2)
    row = tl.program_id(0)
    value = tl.load(X + row * DIM + gc[:, None] * GROUP + g[None, :]).to(
        tl.float32
    )
    scale = tl.max(tl.abs(value), 1) * INV
    tl.store(
        Q + row * QROW + gc[:, None] * GROUP + g[None, :],
        encode_legacy_S(tl.div_rn(value, scale[:, None])),
    )
    bits = scale.to(tl.uint32, bitcast=True)
    tl.store(
        Q + row * QROW + DIM + gc[:, None] * 4 + b[None, :],
        (bits[:, None] >> (b * 8)[None, :] & 255).to(tl.uint8),
    )
    word = (
        tl.load(P + row * RDIM + rc).to(tl.uint16, bitcast=True).to(tl.uint32)
    )
    tl.store(
        output_ptr + row * (RDIM * 2) + rc[:, None] * 2 + h[None, :],
        (word[:, None] >> (h * 8)[None, :] & 255).to(tl.uint8),
    )


@triton.jit
def _strided_S(
    X,
    P,
    Q,
    output_ptr,
    GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    RDIM: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    QROW: tl.constexpr,
    DIM: tl.constexpr,
    INV: tl.constexpr,
):
    g = tl.arange(0, GROUP)
    gc = tl.arange(0, GROUPS)
    rc = tl.arange(0, RDIM)
    b = tl.arange(0, 4)
    h = tl.arange(0, 2)
    row = tl.program_id(0)
    value = tl.load(X + row * X0 + (gc[:, None] * GROUP + g[None, :]) * X1).to(
        tl.float32
    )
    scale = tl.max(tl.abs(value), 1) * INV
    tl.store(
        Q + row * QROW + gc[:, None] * GROUP + g[None, :],
        encode_legacy_S(tl.div_rn(value, scale[:, None])),
    )
    bits = scale.to(tl.uint32, bitcast=True)
    tl.store(
        Q + row * QROW + DIM + gc[:, None] * 4 + b[None, :],
        (bits[:, None] >> (b * 8)[None, :] & 255).to(tl.uint8),
    )
    word = (
        tl.load(P + row * P0 + rc * P1)
        .to(tl.uint16, bitcast=True)
        .to(tl.uint32)
    )
    tl.store(
        output_ptr + row * (RDIM * 2) + rc[:, None] * 2 + h[None, :],
        (word[:, None] >> (h * 8)[None, :] & 255).to(tl.uint8),
    )


@triton.jit
def _encode_S(value):
    quot = value
    af = tl.minimum(tl.abs(quot), 448.0)
    raw = af.to(tl.int32, bitcast=True)
    exponent = tl.maximum(raw >> 23, 121)
    bias = exponent * 8388616 + 167771192
    code = (af + bias.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
    code = tl.where(quot != quot, 127, code)
    sign = value.to(tl.uint32, bitcast=True) >> 24 & 128
    return (code & 127 | sign).to(tl.uint8)


def entry_base_S(k_nope, k_rope, group_size=128):
    tokens, dim_nope = k_nope.shape
    dim_rope = k_rope.shape[1]
    groups = dim_nope // group_size
    width = dim_nope + groups * 4
    nope_part = torch.empty(
        (tokens, 1, width), dtype=torch.uint8, device=k_nope.device
    )
    rope_part = torch.empty(
        (tokens, 1, dim_rope * 2), dtype=torch.uint8, device=k_rope.device
    )
    if tokens:
        warps = 1 if tokens >= 1024 else 2
        if k_nope.is_contiguous() and k_rope.is_contiguous():
            _tight_S.run(
                k_nope,
                k_rope,
                nope_part,
                rope_part,
                group_size,
                groups,
                dim_rope,
                width,
                dim_nope,
                _INV,
                grid=(tokens,),
                warmup=False,
                num_warps=warps,
                enable_fp_fusion=False,
            )
        else:
            _strided_S.run(
                k_nope,
                k_rope,
                nope_part,
                rope_part,
                group_size,
                groups,
                dim_rope,
                *k_nope.stride(),
                *k_rope.stride(),
                width,
                dim_nope,
                _INV,
                grid=(tokens,),
                warmup=False,
                num_warps=warps,
                enable_fp_fusion=False,
            )
    return (nope_part, rope_part)


@triton.jit
def persistent_typed_S(
    X,
    P,
    Q,
    output_ptr,
    ROWS: tl.constexpr,
    STREAM: tl.constexpr,
    INV: tl.constexpr,
):
    g = tl.arange(0, 128)
    groups = tl.arange(0, 4)
    words = tl.arange(0, 32)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        value = tl.load(
            X + row * 512 + groups[:, None] * 128 + g[None, :],
            cache_modifier=".cg" if STREAM else "",
        ).to(tl.float32)
        scale = tl.max(tl.abs(value), 1) * INV
        code = _encode_S(tl.div_rn(value, scale[:, None]))
        tl.store(
            Q + row * 528 + groups[:, None] * 128 + g[None, :],
            code,
            cache_modifier=".cs" if STREAM else "",
        )
        tl.store(
            Q.to(tl.pointer_type(tl.float32)) + row * 132 + 128 + groups,
            scale,
            cache_modifier=".cs" if STREAM else "",
        )
        rope = tl.load(
            P.to(tl.pointer_type(tl.uint32)) + row * 32 + words,
            cache_modifier=".cg" if STREAM else "",
        )
        tl.store(
            output_ptr.to(tl.pointer_type(tl.uint32)) + row * 32 + words,
            rope,
            cache_modifier=".cs" if STREAM else "",
        )


@triton.jit
def persistent2_S(
    X,
    P,
    Q,
    output_ptr,
    ROWS: tl.constexpr,
    INV: tl.constexpr,
    UNROLL: tl.constexpr,
    CHUNK: tl.constexpr,
):
    g = tl.arange(0, 128)
    groups = tl.arange(0, 4)
    words = tl.arange(0, 32)
    cap = tl.num_programs(0)
    loops = tl.cdiv(ROWS, cap)
    for step in tl.range(0, loops, loop_unroll_factor=UNROLL):
        row = (
            tl.program_id(0) * loops + step
            if CHUNK
            else tl.program_id(0) + step * cap
        )
        value = tl.load(X + row * 512 + groups[:, None] * 128 + g[None, :]).to(
            tl.float32
        )
        scale = tl.max(tl.abs(value), 1) * INV
        code = _encode_S(tl.div_rn(value, scale[:, None]))
        tl.store(Q + row * 528 + groups[:, None] * 128 + g[None, :], code)
        tl.store(
            Q.to(tl.pointer_type(tl.float32)) + row * 132 + 128 + groups, scale
        )
        rope = tl.load(P.to(tl.pointer_type(tl.uint32)) + row * 32 + words)
        tl.store(
            output_ptr.to(tl.pointer_type(tl.uint32)) + row * 32 + words, rope
        )


@triton.jit
def group4_kernel_S(X, P, Q, output_ptr, INV: tl.constexpr, TG: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // (4 // TG)
    first = pid % (4 // TG) * TG
    groups = first + tl.arange(0, TG)
    g = tl.arange(0, 128)
    value = tl.load(X + row * 512 + groups[:, None] * 128 + g[None, :]).to(
        tl.float32
    )
    scale = tl.max(tl.abs(value), 1) * INV
    code = _encode_S(tl.div_rn(value, scale[:, None]))
    tl.store(Q + row * 528 + groups[:, None] * 128 + g[None, :], code)
    tl.store(
        Q.to(tl.pointer_type(tl.float32)) + row * 132 + 128 + groups, scale
    )
    i = first * 8 + tl.arange(0, TG * 8)
    rope = tl.load(P.to(tl.pointer_type(tl.uint32)) + row * 32 + i)
    tl.store(output_ptr.to(tl.pointer_type(tl.uint32)) + row * 32 + i, rope)


def entry_candidate_S(k_nope, k_rope, group_size=128):
    rows = k_nope.shape[0]
    if not (
        rows > 0
        and k_nope.shape[1] == 512
        and k_rope.shape[1] == 64
        and group_size == 128
        and k_nope.is_contiguous()
        and k_rope.is_contiguous()
        and k_rope.storage_offset() % 2 == 0
    ):
        return entry_base_S(k_nope, k_rope, group_size)
    q = torch.empty((rows, 1, 528), dtype=torch.uint8, device=k_nope.device)
    o = torch.empty((rows, 1, 128), dtype=torch.uint8, device=k_rope.device)
    if rows <= 256:
        tg = 4 if rows <= 2 else 1 if rows <= 64 else 2
        group4_kernel_S.run(
            k_nope,
            k_rope,
            q,
            o,
            _INV,
            tg,
            grid=(rows * (4 // tg),),
            warmup=False,
            num_warps=2 if rows <= 2 else 1,
            enable_fp_fusion=False,
        )
    elif rows >= 8192 and rows % 1024 == 0:
        persistent2_S.run(
            k_nope,
            k_rope,
            q,
            o,
            rows,
            _INV,
            1,
            True,
            grid=(1024,),
            warmup=False,
            num_warps=1,
            enable_fp_fusion=False,
        )
    else:
        persistent_typed_S.run(
            k_nope,
            k_rope,
            q,
            o,
            rows,
            False,
            _INV,
            grid=(min(rows, 1024),),
            warmup=False,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return q, o


def quantize_k_cache_fast_separate_small(k_nope, k_rope, group_size=128):
    return entry_candidate_S(k_nope, k_rope, group_size)


@triton.jit
def _quantize_k_cache_fast_separate_encode_e4m3fn_L(value):
    bits = value.to(tl.int32, bitcast=True)
    magnitude = bits & 2147483647
    sign = bits >> 24 & 128
    rounded = (magnitude + 524287 + (magnitude >> 20 & 1) >> 20) - 960
    absolute = tl.abs(value)
    subnormal = (absolute + 16384.0).to(tl.int32, bitcast=True) - 1182793728
    code = tl.where(absolute < 0.015625, subnormal, rounded)
    code = tl.minimum(code, 126)
    code = tl.where(value != value, 127, code)
    return (code | sign).to(tl.int8)


@triton.jit
def _quantize_k_cache_fast_separate_pack_kernel_L(
    k_nope_ptr,
    k_rope_ptr,
    quant_ptr,
    scale_ptr,
    rope_out_ptr,
    NUM_TOKENS: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    STRIDE_NOPE_TOKEN: tl.constexpr,
    STRIDE_NOPE_COL: tl.constexpr,
    STRIDE_ROPE_TOKEN: tl.constexpr,
    STRIDE_ROPE_COL: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
    STRIDE_ROPE_OUT: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64) * GROUPS + tl.arange(0, GROUPS)
    row = group // NUM_GROUPS
    tile = group % NUM_GROUPS
    cols = tl.arange(0, BLOCK_SIZE)
    mask = (row[:, None] < NUM_TOKENS) & (cols[None, :] < GROUP_SIZE)
    scaled_value = tl.load(
        k_nope_ptr
        + row[:, None] * STRIDE_NOPE_TOKEN
        + (tile[:, None] * GROUP_SIZE + cols[None, :]) * STRIDE_NOPE_COL,
        mask,
        other=0,
    ).to(tl.float32)
    amax_bits = tl.max(
        scaled_value.to(tl.int32, bitcast=True) & 2147483647, axis=1
    )
    amax = amax_bits.to(tl.float32, bitcast=True)
    scale = amax / 448.0
    tiny = (scale > 0.0) & (scale < 7.888609052210118e-31)
    denominator = tl.where(tiny, scale * 1.8446744073709552e19, scale)
    inv_scale = 1.0 / denominator
    numerator = tl.where(
        tiny[:, None], scaled_value * 1.8446744073709552e19, scaled_value
    )
    value = numerator * inv_scale[:, None]
    value = tl.where(value < -448.0, -448.0, value)
    value = tl.where(value > 448.0, 448.0, value)
    encoded = _quantize_k_cache_fast_separate_encode_e4m3fn_L(value)
    tl.store(
        quant_ptr
        + row[:, None] * OUTPUT_WIDTH
        + tile[:, None] * GROUP_SIZE
        + cols[None, :],
        encoded,
        mask,
    )
    tl.store(
        scale_ptr + row * (OUTPUT_WIDTH // 4) + tile, scale, row < NUM_TOKENS
    )
    rope_cols = tl.arange(0, BLOCK_ROWS)
    rope_mask = (
        (row[:, None] < NUM_TOKENS)
        & (tile[:, None] == 0)
        & (rope_cols[None, :] < ROPE_DIM)
    )
    rope = tl.load(
        k_rope_ptr
        + row[:, None] * STRIDE_ROPE_TOKEN
        + rope_cols[None, :] * STRIDE_ROPE_COL,
        rope_mask,
        other=0,
    )
    tl.store(
        rope_out_ptr + row[:, None] * STRIDE_ROPE_OUT + rope_cols[None, :],
        rope,
        rope_mask,
    )


def quantize_k_cache_fast_separate_large(
    k_nope,
    k_rope,
    group_size=128,
):
    (num_tokens, nope_dim) = k_nope.shape
    rope_dim = k_rope.shape[1]
    num_groups = nope_dim // group_size
    width = nope_dim + num_groups * 4
    nope_part = torch.empty(
        (num_tokens, width), dtype=torch.uint8, device=k_nope.device
    )
    rope_part = torch.empty_strided(
        (num_tokens, rope_dim * 2),
        (max(2, rope_dim * 2), 1),
        dtype=torch.uint8,
        device=k_rope.device,
    )
    if num_tokens:
        raw = nope_part.view(torch.int8)
        scales = raw[:, nope_dim:].view(torch.float32)
        rope = rope_part.view(torch.int16)
        group_batch = max(
            1, min(8, 1024 // triton.next_power_of_2(group_size))
        )
        _quantize_k_cache_fast_separate_pack_kernel_L[
            (triton.cdiv(num_tokens * num_groups, group_batch),)
        ](
            k_nope,
            k_rope.view(torch.int16),
            raw,
            scales,
            rope,
            num_tokens,
            nope_dim,
            rope_dim,
            num_groups,
            group_size,
            k_nope.stride(0),
            k_nope.stride(1),
            k_rope.stride(0),
            k_rope.stride(1),
            width,
            triton.next_power_of_2(group_size),
            triton.next_power_of_2(max(1, rope_dim)),
            group_batch,
            rope_dim,
            num_warps=4,
        )
    return (nope_part.unsqueeze(1), rope_part.unsqueeze(1))


def quantize_k_cache_fast_separate(k_nope, k_rope, group_size=128):
    if k_nope.shape[0] <= 2048:
        return quantize_k_cache_fast_separate_small(k_nope, k_rope, group_size)
    return quantize_k_cache_fast_separate_large(k_nope, k_rope, group_size)


__all__ = ["quantize_k_cache_fast_separate"]
