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


@triton.jit
def _encode_fp8_e4m3(x_f32):
    """Round fp32 to e4m3fn codes (uint8): RNE, exact in the normal."""
    bits = x_f32.to(tl.int32, bitcast=True)
    mag = bits & 0x7FFFFFFF
    sign = (bits >> 24) & 0x80
    code = ((mag + 0x7FFFF + ((mag >> 20) & 1)) >> 20) - 960
    code = tl.minimum(tl.maximum(code, 0), 126)
    return (code | sign).to(tl.uint8)


@triton.jit
def _quantize_nope_rope_block_scale(
    kn_ptr, kr_ptr, q_ptr, r_ptr, BLOCK_T: tl.constexpr
):
    """Small-T fast path, 4 args, zero host views: q_ptr (uint8)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)

    # rope half: bit-exact passthrough. Load the bf16 row, bitcast to int16,
    # split into little-endian bytes and interleave them into the uint8 row.
    ow = tl.arange(0, 64)
    v16 = tl.load(kr_ptr + rows[:, None] * 64 + ow[None, :]).to(
        tl.int16, bitcast=True
    )
    lo = (v16 & 0xFF).to(tl.uint8)
    hi = ((v16 >> 8) & 0xFF).to(tl.uint8)
    tl.store(
        r_ptr + rows[:, None] * 128 + tl.arange(0, 128)[None, :],
        tl.reshape(tl.join(lo, hi), (BLOCK_T, 128)),
    )

    offs_g = tl.arange(0, 128)
    acc = tl.zeros((BLOCK_T, 4), dtype=tl.float32)
    for i in tl.range(0, 4, num_stages=2):
        x = tl.load(
            kn_ptr + rows[:, None] * 512 + i * 128 + offs_g[None, :]
        ).to(tl.float32)

        scale = tl.max(tl.abs(x), axis=1) / 448.0
        col = (tl.arange(0, 4)[None, :] == i).to(tl.float32)
        acc = acc * (1.0 - col) + scale[:, None] * col

        tl.store(
            q_ptr + rows[:, None] * 528 + i * 128 + offs_g[None, :],
            _encode_fp8_e4m3(x / scale[:, None]),
        )

    # fp32 scales as one contiguous 16-byte block per row: spread each int32
    # word into its four little-endian bytes, giving (BLOCK_T, 4*NT) bytes.
    s_bits = acc.to(tl.int32, bitcast=True)[:, :, None]
    s_bytes = (s_bits >> (8 * tl.arange(0, 4)[None, None, :])) & 0xFF
    tl.store(
        q_ptr + rows[:, None] * 528 + 512 + tl.arange(0, 16)[None, :],
        tl.reshape(s_bytes, (BLOCK_T, 16)).to(tl.uint8),
    )


@triton.jit
def _quantize_nope_rope_word_scale(
    kn_ptr, kr_ptr, q_ptr, w_ptr, rb_ptr, BLOCK_T: tl.constexpr
):
    """Large-T fast path, 5 args. Rope is a *direct bf16 load/store*."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)

    # rope half: bit-exact bf16 -> bf16 passthrough.
    ow = tl.arange(0, 64)
    v = tl.load(kr_ptr + rows[:, None] * 64 + ow[None, :])
    tl.store(rb_ptr + rows[:, None] * 64 + ow[None, :], v)

    offs_g = tl.arange(0, 128)
    acc = tl.zeros((BLOCK_T, 4), dtype=tl.float32)
    for i in tl.range(0, 4, num_stages=2):
        x = tl.load(
            kn_ptr + rows[:, None] * 512 + i * 128 + offs_g[None, :]
        ).to(tl.float32)

        scale = tl.max(tl.abs(x), axis=1) / 448.0
        col = (tl.arange(0, 4)[None, :] == i).to(tl.float32)
        acc = acc * (1.0 - col) + scale[:, None] * col

        tl.store(
            q_ptr + rows[:, None] * 528 + i * 128 + offs_g[None, :],
            _encode_fp8_e4m3(x / scale[:, None]),
        )

    tl.store(
        w_ptr + rows[:, None] * 132 + 128 + tl.arange(0, 4)[None, :],
        acc.to(tl.int32, bitcast=True),
    )


@triton.jit
def _quantize_separate_generic_kernel(
    kn_ptr,
    kr_ptr,
    q_ptr,
    u8_ptr,
    r_ptr,
    T,
    SKN,
    SKR,
    SQ,
    SR,
    DIM_NOPE: tl.constexpr,
    NOPE_ROW: tl.constexpr,
    NT: tl.constexpr,
    GROUP: tl.constexpr,
    GROUP_P: tl.constexpr,
    ROPE: tl.constexpr,
    ROPE_P: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Generic fallback for non-production shapes: element-granular stores;
    scales go out as four byte-plane stores per (row, tile)."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    m_r = rows < T
    tidx = tl.arange(0, NT)

    if HAS_ROPE:
        offs_r = tl.arange(0, ROPE_P)
        m_rc = m_r[:, None] & (offs_r < ROPE)[None, :]
        rope = tl.load(
            kr_ptr + rows[:, None] * SKR + offs_r[None, :], mask=m_rc, other=0
        )
        tl.store(r_ptr + rows[:, None] * SR + offs_r[None, :], rope, mask=m_rc)

    offs_g = tl.arange(0, GROUP_P)
    m_g = offs_g < GROUP
    acc = tl.zeros((BLOCK_T, NT), dtype=tl.float32)
    for i in tl.range(0, NT):
        x = tl.load(
            kn_ptr + rows[:, None] * SKN + i * GROUP + offs_g[None, :],
            mask=m_r[:, None] & m_g[None, :],
            other=0.0,
        )
        x = x.to(tl.float32)

        scale = tl.max(tl.abs(x), axis=1) / 448.0
        col = (tidx[None, :] == i).to(tl.float32)
        acc = acc * (1.0 - col) + scale[:, None] * col

        code = _encode_fp8_e4m3(x / scale[:, None])
        tl.store(
            q_ptr + rows[:, None] * SQ + i * GROUP + offs_g[None, :],
            code,
            mask=m_r[:, None] & m_g[None, :],
        )

    s_int = acc.to(tl.int32, bitcast=True)
    for b in tl.static_range(4):
        byte = ((s_int >> (8 * b)) & 0xFF).to(tl.uint8)
        tl.store(
            u8_ptr
            + rows[:, None] * NOPE_ROW
            + DIM_NOPE
            + tidx[None, :] * 4
            + b,
            byte,
            mask=m_r[:, None],
        )


def quantize_k_cache_fast_separate(k_nope, k_rope, group_size=128):
    T, dim_nope = k_nope.shape
    dim_rope = k_rope.shape[1]

    if (
        dim_nope == 512
        and group_size == 128
        and dim_rope == 64
        and k_rope.element_size() == 2
        and k_nope.dtype == torch.bfloat16
        and k_rope.dtype == torch.bfloat16
    ):
        if not k_nope.is_contiguous():
            k_nope = k_nope.contiguous()
        if not k_rope.is_contiguous():
            k_rope = k_rope.contiguous()

        # Allocate the 3-D outputs directly (new_empty inherits the device
        # without re-parsing a device argument).
        nope_out = k_nope.new_empty((T, 1, 528), dtype=torch.uint8)
        rope_out = k_rope.new_empty((T, 1, 128), dtype=torch.uint8)
        if T == 0:
            return nope_out, rope_out

        if T > 4096 and T % 32 == 0:
            _quantize_nope_rope_word_scale[(T // 32,)](
                k_nope,
                k_rope,
                nope_out,
                nope_out.view(torch.int32),
                rope_out.view(torch.bfloat16),
                32,
            )
        else:
            for block_t in (8, 4, 2, 1):
                if T % block_t == 0:
                    break
            _quantize_nope_rope_block_scale[(T // block_t,)](
                k_nope,
                k_rope,
                nope_out,
                rope_out,
                block_t,
                num_warps=1 if block_t > 1 else 2,
            )
        return nope_out, rope_out

    nt = dim_nope // group_size
    nope_row = dim_nope + nt * 4
    esz = k_rope.element_size()
    rope_row = dim_rope * esz

    if not k_nope.is_contiguous():
        k_nope = k_nope.contiguous()
    if not k_rope.is_contiguous():
        k_rope = k_rope.contiguous()

    nope_out = k_nope.new_empty((T, 1, nope_row), dtype=torch.uint8)
    rope_out = k_rope.new_empty((T, 1, rope_row), dtype=torch.uint8)
    if T == 0:
        return nope_out, rope_out

    # Generic path (unaligned shapes): bit-exact rope copy via int16 views,
    # uint8 fp8 codes, byte-plane scale stores.
    has_rope = dim_rope > 0 and esz > 0
    group_p = triton.next_power_of_2(max(group_size, 1))
    rope_p = triton.next_power_of_2(dim_rope) if dim_rope > 0 else 1
    kr_view = k_rope.view(torch.int16) if esz == 2 else k_rope
    r_view = rope_out.view(torch.int16) if esz == 2 else rope_out
    sr = dim_rope if esz == 2 else rope_row
    no3 = nope_out.view(T, nope_row)

    if T >= 4096:
        block_t, num_warps = 32, 8
    else:
        block_t, num_warps = 8, 2

    _quantize_separate_generic_kernel[((T + block_t - 1) // block_t,)](
        k_nope,
        kr_view,
        no3,
        no3,
        r_view,
        T,
        k_nope.stride(0),
        k_rope.stride(0),
        nope_row,
        sr,
        dim_nope,
        nope_row,
        nt,
        group_size,
        group_p,
        dim_rope,
        rope_p,
        has_rope,
        block_t,
        num_warps=num_warps,
    )
    return nope_out, rope_out


__all__ = ["quantize_k_cache_fast_separate"]
