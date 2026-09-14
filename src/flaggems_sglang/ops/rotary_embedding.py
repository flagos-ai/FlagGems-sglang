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
def _rotary_v3(
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
    HEAD_DIM: tl.constexpr,
    HALF: tl.constexpr,
    PAD_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(0)
    t0 = tl.program_id(1) * BLOCK_T
    ts = t0 + tl.arange(0, BLOCK_T)
    d = tl.arange(0, PAD_D)
    t_ok = ts < NUM_TOKENS
    d_ok = d < HEAD_DIM
    mask2d = t_ok[:, None] & d_ok[None, :]

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

    x_offs = (
        ts[:, None] * STRIDE_X_S + h * STRIDE_X_H + d[None, :] * STRIDE_X_D
    )
    x = tl.load(x_ptr + x_offs, mask=mask2d, other=0.0).to(tl.float32)
    x1, x2 = tl.split(tl.reshape(x, (BLOCK_T, PAD_D // 2, 2)))
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    out = tl.reshape(tl.join(o1, o2), (BLOCK_T, PAD_D))
    o_offs = (
        ts[:, None] * STRIDE_OUT_S
        + h * STRIDE_OUT_H
        + d[None, :] * STRIDE_OUT_D
    )
    tl.store(out_ptr + o_offs, out.to(out_ptr.dtype.element_ty), mask=mask2d)


def rotary_embedding(x, cos, sin, interleaved):
    num_tokens, num_heads, head_size = x.shape
    out = torch.empty_like(x)
    pad_d = 1 << (head_size - 1).bit_length()
    if pad_d < 16:
        pad_d = 16
    grid = (num_heads, (num_tokens + 15) // 16)
    _rotary_v3[grid](
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
        head_size,
        head_size // 2,
        pad_d,
        16,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["rotary_embedding"]
