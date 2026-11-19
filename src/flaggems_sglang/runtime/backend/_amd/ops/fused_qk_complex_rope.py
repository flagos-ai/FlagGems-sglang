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

TILE = 4096
PROGRAMS = 1024
WORDS_PER_WARP = 64
MAX_WARPS = 8


def _fused_qk_complex_rope_get_num_warps(num_words):
    """Warps enough to give each lane a word, up to the tile's worth."""
    return max(1, min(MAX_WARPS, num_words // WORDS_PER_WARP))


def _fused_qk_complex_rope_can_pack_pairs(
    x,
    stride,
):
    """Whether the rows of `x` can be read as 32-bit words."""
    return (
        stride[2] == 1
        and stride[0] % 2 == 0
        and stride[1] % 2 == 0
        and x.storage_offset() % 2 == 0
    )


def _fused_qk_complex_rope_get_block_heads(
    num_tokens,
    num_heads,
    num_pairs,
):
    block_size = 1 << (
        max(1, min(num_heads, TILE // num_pairs)).bit_length() - 1
    )
    while (
        block_size > 1
        and num_tokens * triton.cdiv(num_heads, block_size) < PROGRAMS
    ):
        block_size //= 2
    return block_size


@triton.jit
def _fused_qk_complex_rope_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    q_out_ptr,
    k_out_ptr,
    NUM_HEADS: tl.constexpr,
    NUM_PAIRS: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    NUM_HEAD_BLOCKS: tl.constexpr,
    STRIDE_Q_TOKEN: tl.constexpr,
    STRIDE_Q_HEAD: tl.constexpr,
    STRIDE_Q_COL: tl.constexpr,
    STRIDE_K_TOKEN: tl.constexpr,
    STRIDE_K_HEAD: tl.constexpr,
    STRIDE_K_COL: tl.constexpr,
    STRIDE_FREQ_TOKEN: tl.constexpr,
    STRIDE_FREQ_PAIR: tl.constexpr,
    STRIDE_FREQ_COMPONENT: tl.constexpr,
    ALIGNED_SHAPE: tl.constexpr,
    IS_BF16: tl.constexpr,
    PACK_PAIRS: tl.constexpr,
):
    program_id = tl.program_id(0)
    token = program_id // NUM_HEAD_BLOCKS
    head = (program_id % NUM_HEAD_BLOCKS) * BLOCK_HEADS + tl.arange(
        0, BLOCK_HEADS
    )[:, None]
    pair = tl.arange(0, BLOCK_PAIRS)
    if ALIGNED_SHAPE:
        mask = None
        row = None
    else:
        row = pair < NUM_PAIRS
        mask = (head < NUM_HEADS) & row[None, :]
    freq = freqs_ptr + token * STRIDE_FREQ_TOKEN + pair * STRIDE_FREQ_PAIR
    cos = tl.load(freq, mask=row)[None, :]
    sin = tl.load(freq + STRIDE_FREQ_COMPONENT, mask=row)[None, :]
    if PACK_PAIRS:
        # A pair is one word, so the row is one contiguous run of them
        # and the pointers arrive already viewed as words.
        q_word = tl.load(
            q_ptr
            + token * STRIDE_Q_TOKEN
            + head * STRIDE_Q_HEAD
            + pair[None, :],
            mask=mask,
            cache_modifier=".cg",
        )
        k_word = tl.load(
            k_ptr
            + token * STRIDE_K_TOKEN
            + head * STRIDE_K_HEAD
            + pair[None, :],
            mask=mask,
            cache_modifier=".cg",
        )
        q_real = _fused_qk_complex_rope_unpack_low(q_word, IS_BF16)
        q_imag = _fused_qk_complex_rope_unpack_high(q_word, IS_BF16)
        k_real = _fused_qk_complex_rope_unpack_low(k_word, IS_BF16)
        k_imag = _fused_qk_complex_rope_unpack_high(k_word, IS_BF16)
    else:
        q_in = (
            q_ptr
            + token * STRIDE_Q_TOKEN
            + head * STRIDE_Q_HEAD
            + pair[None, :] * (2 * STRIDE_Q_COL)
        )
        k_in = (
            k_ptr
            + token * STRIDE_K_TOKEN
            + head * STRIDE_K_HEAD
            + pair[None, :] * (2 * STRIDE_K_COL)
        )
        q_real = tl.load(q_in, mask=mask).to(tl.float32)
        q_imag = tl.load(q_in + STRIDE_Q_COL, mask=mask).to(tl.float32)
        k_real = tl.load(k_in, mask=mask).to(tl.float32)
        k_imag = tl.load(k_in + STRIDE_K_COL, mask=mask).to(tl.float32)
    q_real_out = q_real * cos - q_imag * sin
    q_imag_out = q_real * sin + q_imag * cos
    k_real_out = k_real * cos - k_imag * sin
    k_imag_out = k_real * sin + k_imag * cos
    if PACK_PAIRS:
        out = (
            token * (NUM_HEADS * NUM_PAIRS) + head * NUM_PAIRS + pair[None, :]
        )
        tl.store(
            q_out_ptr + out,
            _fused_qk_complex_rope_pack_pair(q_real_out, q_imag_out, IS_BF16),
            mask=mask,
            cache_modifier=".cs",
        )
        tl.store(
            k_out_ptr + out,
            _fused_qk_complex_rope_pack_pair(k_real_out, k_imag_out, IS_BF16),
            mask=mask,
            cache_modifier=".cs",
        )
    else:
        out = (
            token * (NUM_HEADS * 2 * NUM_PAIRS)
            + head * (2 * NUM_PAIRS)
            + pair[None, :] * 2
        )
        dtype = q_out_ptr.dtype.element_ty
        tl.store(q_out_ptr + out, q_real_out.to(dtype), mask=mask)
        tl.store(q_out_ptr + out + 1, q_imag_out.to(dtype), mask=mask)
        tl.store(k_out_ptr + out, k_real_out.to(dtype), mask=mask)
        tl.store(k_out_ptr + out + 1, k_imag_out.to(dtype), mask=mask)


@triton.jit
def _fused_qk_complex_rope_unpack_low(
    word,
    IS_BF16: tl.constexpr,
):
    """The first element of the pair, from the word's low half."""
    if IS_BF16:
        # BF16 is the top half of an FP32 word, so the shift is the cast.
        return (word << 16).to(tl.float32, bitcast=True)
    return (
        (word & 0xFFFF)
        .to(tl.int16)
        .to(tl.float16, bitcast=True)
        .to(tl.float32)
    )


@triton.jit
def _fused_qk_complex_rope_unpack_high(
    word,
    IS_BF16: tl.constexpr,
):
    """The second element of the pair, from the word's high half."""
    if IS_BF16:
        return ((word >> 16) << 16).to(tl.float32, bitcast=True)
    return (
        (word >> 16).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    )


@triton.jit
def _fused_qk_complex_rope_pack_pair(
    real,
    imag,
    IS_BF16: tl.constexpr,
):
    """Round both halves to the output type and lay them in one word."""
    if IS_BF16:
        low = real.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        high = imag.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    else:
        low = real.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
        high = imag.to(tl.float16).to(tl.int16, bitcast=True).to(tl.int32)
    # The 16-bit halves arrive sign extended, which the mask undoes for
    # the low one and the shift drops for the high one.
    return (low & 0xFFFF) | (high << 16)


def fused_qk_complex_rope(
    q,
    k,
    freqs_cis,
):
    """Rotate Q and K by one complex frequency row per token."""
    shape = q.shape
    num_heads = shape[-2]
    dim = shape[-1]
    half = dim // 2
    q_flat = q if q.ndim == 3 else q.reshape(-1, num_heads, dim)
    k_flat = k if k.ndim == 3 else k.reshape(-1, num_heads, dim)
    freqs = torch.view_as_real(freqs_cis)
    if freqs.ndim > 3:
        freqs = freqs.reshape(-1, half, 2)
    num_tokens = q_flat.shape[0]
    q_out = torch.empty(q_flat.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k_flat.shape, dtype=k.dtype, device=k.device)
    if num_tokens:
        q_stride = q_flat.stride()
        k_stride = k_flat.stride()
        num_pairs = triton.next_power_of_2(half)
        block_h = _fused_qk_complex_rope_get_block_heads(
            num_tokens, num_heads, num_pairs
        )
        hblocks = triton.cdiv(num_heads, block_h)
        num_words = _fused_qk_complex_rope_can_pack_pairs(
            q_flat, q_stride
        ) and _fused_qk_complex_rope_can_pack_pairs(k_flat, k_stride)
        if num_words:
            q_in = q_flat.view(torch.int32)
            k_in = k_flat.view(torch.int32)
            q_dst = q_out.view(torch.int32)
            k_dst = k_out.view(torch.int32)
            # A word holds two elements, so the view's strides are the
            # tensor's own halved, and the last one is not read.
            q_steps = (q_stride[0] // 2, q_stride[1] // 2, 1)
            k_steps = (k_stride[0] // 2, k_stride[1] // 2, 1)
        else:
            q_in, k_in, q_dst, k_dst = q_flat, k_flat, q_out, k_out
            q_steps, k_steps = q_stride, k_stride
        _fused_qk_complex_rope_kernel[(num_tokens * hblocks,)](
            q_in,
            k_in,
            freqs,
            q_dst,
            k_dst,
            num_heads,
            half,
            num_pairs,
            block_h,
            hblocks,
            # One call for the three strides: at these sizes the
            # kernel runs in the time the call around it takes.
            *q_steps,
            *k_steps,
            *freqs.stride(),
            num_pairs == half and num_heads % block_h == 0,
            q.dtype == torch.bfloat16,
            num_words,
            num_warps=_fused_qk_complex_rope_get_num_warps(
                block_h * num_pairs
            ),
        )
    return q_out.view(shape), k_out.view(shape)


__all__ = ["fused_qk_complex_rope"]
