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
import triton.language as tl


@triton.jit
def _rotary_gcu_v4w1(
    w_ptr,
    cos_ptr,
    sin_ptr,
    o_ptr,
    NUM_TOKENS: tl.constexpr,
    STRIDE_W_S: tl.constexpr,
    STRIDE_W_H: tl.constexpr,
    STRIDE_COS_S: tl.constexpr,
    STRIDE_COS_D: tl.constexpr,
    STRIDE_SIN_S: tl.constexpr,
    STRIDE_SIN_D: tl.constexpr,
    STRIDE_O_S: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HALF: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_HALF: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    t0 = tl.program_id(0) * BLOCK_T
    ts = t0 + tl.arange(0, BLOCK_T)
    hs = tl.arange(0, PAD_H)
    pair = tl.arange(0, PAD_HALF)
    t_ok = ts < NUM_TOKENS
    h_ok = hs < NUM_HEADS
    p_ok = pair < HALF
    mask3d = t_ok[:, None, None] & h_ok[None, :, None] & p_ok[None, None, :]

    cs_mask = t_ok[:, None] & p_ok[None, :]
    cos = tl.load(
        cos_ptr + ts[:, None] * STRIDE_COS_S + pair[None, :] * STRIDE_COS_D,
        mask=cs_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + ts[:, None] * STRIDE_SIN_S + pair[None, :] * STRIDE_SIN_D,
        mask=cs_mask,
        other=0.0,
    ).to(tl.float32)

    w_offs = (
        ts[:, None, None] * STRIDE_W_S
        + hs[None, :, None] * STRIDE_W_H
        + pair[None, None, :]
    )
    w = tl.load(w_ptr + w_offs, mask=mask3d, other=0)
    x1 = (w << 16).to(tl.float32, bitcast=True)
    x2 = (w & -65536).to(tl.float32, bitcast=True)
    o1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
    o2 = x1 * sin[:, None, :] + x2 * cos[:, None, :]
    b1 = o1.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    b2 = o2.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    tl.store(o_ptr + w_offs, (b2 << 16) | b1, mask=mask3d)


@triton.jit
def _rotary_gcu_v3w1(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    NUM_TOKENS: tl.constexpr,
    STRIDE_X_S: tl.constexpr,
    STRIDE_X_H: tl.constexpr,
    STRIDE_X_D: tl.constexpr,
    STRIDE_COS_S: tl.constexpr,
    STRIDE_COS_D: tl.constexpr,
    STRIDE_SIN_S: tl.constexpr,
    STRIDE_SIN_D: tl.constexpr,
    STRIDE_OUT_S: tl.constexpr,
    STRIDE_OUT_H: tl.constexpr,
    STRIDE_OUT_D: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    t0 = tl.program_id(0) * BLOCK_T
    ts = t0 + tl.arange(0, BLOCK_T)
    hs = tl.arange(0, PAD_H)
    d = tl.arange(0, PAD_D)
    t_ok = ts < NUM_TOKENS
    h_ok = hs < NUM_HEADS
    d_ok = d < HEAD_DIM
    mask3d = t_ok[:, None, None] & h_ok[None, :, None] & d_ok[None, None, :]

    pair = tl.arange(0, PAD_D // 2)
    cs_mask = t_ok[:, None] & (pair < HALF)[None, :]
    cos = tl.load(
        cos_ptr + ts[:, None] * STRIDE_COS_S + pair[None, :] * STRIDE_COS_D,
        mask=cs_mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + ts[:, None] * STRIDE_SIN_S + pair[None, :] * STRIDE_SIN_D,
        mask=cs_mask,
        other=0.0,
    ).to(tl.float32)
    cos3 = tl.broadcast_to(cos[:, None, :], (BLOCK_T, PAD_H, PAD_D // 2))
    sin3 = tl.broadcast_to(sin[:, None, :], (BLOCK_T, PAD_H, PAD_D // 2))

    x_offs = (
        ts[:, None, None] * STRIDE_X_S
        + hs[None, :, None] * STRIDE_X_H
        + d[None, None, :] * STRIDE_X_D
    )
    x = tl.load(x_ptr + x_offs, mask=mask3d, other=0.0).to(tl.float32)
    x1, x2 = tl.split(tl.reshape(x, (BLOCK_T, PAD_H, PAD_D // 2, 2)))
    o1 = x1 * cos3 - x2 * sin3
    o2 = x1 * sin3 + x2 * cos3
    out = tl.reshape(tl.join(o1, o2), (BLOCK_T, PAD_H, PAD_D))
    o_offs = (
        ts[:, None, None] * STRIDE_OUT_S
        + hs[None, :, None] * STRIDE_OUT_H
        + d[None, None, :] * STRIDE_OUT_D
    )
    tl.store(out_ptr + o_offs, out.to(out_ptr.dtype.element_ty), mask=mask3d)


def rotary_embedding(x, cos, sin, interleaved):
    num_tokens, num_heads, head_size = x.shape
    half = head_size // 2
    out = torch.empty_like(x)
    pad_h = 1 << (num_heads - 1).bit_length()
    if (
        x.dtype == torch.bfloat16
        and head_size % 2 == 0
        and x.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
    ):
        w = x.view(torch.int32)
        wo = out.view(torch.int32)
        pad_half = 1 << (half - 1).bit_length()
        if pad_half < 8:
            pad_half = 8
        block_t = 32768 // (pad_h * pad_half)
        if block_t < 1:
            block_t = 1
        elif block_t > 64:
            block_t = 64
        grid = ((num_tokens + block_t - 1) // block_t,)
        _rotary_gcu_v4w1[grid](
            w,
            cos,
            sin,
            wo,
            num_tokens,
            w.stride(0),
            w.stride(1),
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1),
            wo.stride(0),
            wo.stride(1),
            num_heads,
            half,
            pad_h,
            pad_half,
            block_t,
            num_warps=1,
            num_stages=1,
        )
        return out
    pad_d = 1 << (head_size - 1).bit_length()
    if pad_d < 16:
        pad_d = 16
    block_t = 32768 // (pad_h * pad_d)
    if block_t < 1:
        block_t = 1
    elif block_t > 64:
        block_t = 64
    grid = ((num_tokens + block_t - 1) // block_t,)
    _rotary_gcu_v3w1[grid](
        x,
        cos,
        sin,
        out,
        num_tokens,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_size,
        half,
        pad_h,
        pad_d,
        block_t,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["rotary_embedding"]
