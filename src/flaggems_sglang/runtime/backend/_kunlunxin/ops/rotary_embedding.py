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
def _rotary_xpu(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    stride_x_s,
    stride_x_h,
    stride_x_d,
    stride_cos_s,
    stride_cos_d,
    stride_sin_s,
    stride_sin_d,
    stride_out_s,
    stride_out_h,
    stride_out_d,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
):
    s_id = tl.program_id(0)
    ordered = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered < HEAD_DIM
    odd_mask = ordered % 2 == 0
    rotated = tl.where(odd_mask, ordered + 1, ordered - 1)
    sin_cos_block = ordered // 2
    cos = tl.load(
        cos_ptr + s_id * stride_cos_s + sin_cos_block * stride_cos_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        sin_ptr + s_id * stride_sin_s + sin_cos_block * stride_sin_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.where(odd_mask, -sin, sin)

    x_row = x_ptr + s_id * stride_x_s
    out_row = out_ptr + s_id * stride_out_s
    for off_h in range(0, NUM_HEADS):
        q = tl.load(
            x_row + off_h * stride_x_h + ordered * stride_x_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        rotated_q = tl.load(
            x_row + off_h * stride_x_h + rotated * stride_x_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        y = q * cos + rotated_q * sin
        tl.store(
            out_row + off_h * stride_out_h + ordered * stride_out_d,
            y.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def rotary_embedding(x, cos, sin, interleaved):
    num_tokens, num_heads, head_size = x.shape
    out = torch.empty(
        (num_tokens, num_heads, head_size), dtype=x.dtype, device=x.device
    )
    padded = max(triton.next_power_of_2(head_size), 16)
    _rotary_xpu[(num_tokens,)](
        x,
        cos,
        sin,
        out,
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
        NUM_HEADS=num_heads,
        HEAD_DIM=head_size,
        PADDED_HEAD_DIM=padded,
    )
    return out


__all__ = ["rotary_embedding"]
