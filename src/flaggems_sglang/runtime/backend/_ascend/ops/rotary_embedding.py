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
def _rotary_npu_v4_br4(
    w_ptr,
    cos_ptr,
    sin_ptr,
    o_ptr,
    NUM_TASKS: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    STRIDE_W_S: tl.constexpr,
    STRIDE_W_H: tl.constexpr,
    STRIDE_COS_S: tl.constexpr,
    STRIDE_COS_D: tl.constexpr,
    STRIDE_SIN_S: tl.constexpr,
    STRIDE_SIN_D: tl.constexpr,
    STRIDE_O_S: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    HALF: tl.constexpr,
    PAD_HALF: tl.constexpr,
    BLOCK_R: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    pair = tl.arange(0, PAD_HALF)
    for task in tl.range(pid, NUM_TASKS, num_programs):
        rows = task * BLOCK_R + tl.arange(0, BLOCK_R)
        rows_in = tl.minimum(rows, TOTAL_ROWS - 1)
        t = rows_in // NUM_HEADS
        h = rows_in - t * NUM_HEADS
        if EXACT:
            cs = tl.load(
                cos_ptr
                + t[:, None] * STRIDE_COS_S
                + pair[None, :] * STRIDE_COS_D
            )
            sn = tl.load(
                sin_ptr
                + t[:, None] * STRIDE_SIN_S
                + pair[None, :] * STRIDE_SIN_D
            )
            w_off = (
                t[:, None] * STRIDE_W_S
                + h[:, None] * STRIDE_W_H
                + pair[None, :]
            )
            w = tl.load(w_ptr + w_off)
        else:
            p_in = tl.minimum(pair, HALF - 1)
            cs = tl.load(
                cos_ptr
                + t[:, None] * STRIDE_COS_S
                + p_in[None, :] * STRIDE_COS_D
            )
            sn = tl.load(
                sin_ptr
                + t[:, None] * STRIDE_SIN_S
                + p_in[None, :] * STRIDE_SIN_D
            )
            w_off = (
                t[:, None] * STRIDE_W_S
                + h[:, None] * STRIDE_W_H
                + p_in[None, :]
            )
            w = tl.load(w_ptr + w_off)
        x1 = (w << 16).to(tl.float32, bitcast=True)
        x2 = (w & -65536).to(tl.float32, bitcast=True)
        csf = cs.to(tl.float32)
        snf = sn.to(tl.float32)
        o1 = x1 * csf - x2 * snf
        o2 = x1 * snf + x2 * csf
        b1 = o1.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
        b2 = o2.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
        w_out = (b2 << 16) | b1
        o_off = (
            t[:, None] * STRIDE_O_S + h[:, None] * STRIDE_O_H + pair[None, :]
        )
        if EXACT:
            tl.store(o_ptr + o_off, w_out)
        else:
            valid = (rows < TOTAL_ROWS)[:, None] & (pair < HALF)[None, :]
            tl.store(o_ptr + o_off, w_out, mask=valid)


@triton.jit
def _rotary_npu_v3(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    NUM_TASKS: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
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
    BLOCK_R: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    d = tl.arange(0, PAD_D)
    pair = tl.arange(0, PAD_D // 2)
    for task in tl.range(pid, NUM_TASKS, num_programs):
        rows = task * BLOCK_R + tl.arange(0, BLOCK_R)
        rows_in = tl.minimum(rows, TOTAL_ROWS - 1)
        t = rows_in // NUM_HEADS
        h = rows_in - t * NUM_HEADS
        if EXACT:
            cs = tl.load(
                cos_ptr
                + t[:, None] * STRIDE_COS_S
                + pair[None, :] * STRIDE_COS_D
            )
            sn = tl.load(
                sin_ptr
                + t[:, None] * STRIDE_SIN_S
                + pair[None, :] * STRIDE_SIN_D
            )
            x_off = (
                t[:, None] * STRIDE_X_S
                + h[:, None] * STRIDE_X_H
                + d[None, :] * STRIDE_X_D
            )
            x = tl.load(x_ptr + x_off)
        else:
            p_in = tl.minimum(pair, HALF - 1)
            d_in = tl.minimum(d, HEAD_DIM - 1)
            cs = tl.load(
                cos_ptr
                + t[:, None] * STRIDE_COS_S
                + p_in[None, :] * STRIDE_COS_D
            )
            sn = tl.load(
                sin_ptr
                + t[:, None] * STRIDE_SIN_S
                + p_in[None, :] * STRIDE_SIN_D
            )
            x_off = (
                t[:, None] * STRIDE_X_S
                + h[:, None] * STRIDE_X_H
                + d_in[None, :] * STRIDE_X_D
            )
            x = tl.load(x_ptr + x_off)
        xf = x.to(tl.float32)
        x1, x2 = tl.split(tl.reshape(xf, (BLOCK_R, PAD_D // 2, 2)))
        o1 = x1 * cs.to(tl.float32) - x2 * sn.to(tl.float32)
        o2 = x1 * sn.to(tl.float32) + x2 * cs.to(tl.float32)
        out = tl.reshape(tl.join(o1, o2), (BLOCK_R, PAD_D))
        o_off = (
            t[:, None] * STRIDE_OUT_S
            + h[:, None] * STRIDE_OUT_H
            + d[None, :] * STRIDE_OUT_D
        )
        if EXACT:
            tl.store(out_ptr + o_off, out.to(out_ptr.dtype.element_ty))
        else:
            valid = (rows < TOTAL_ROWS)[:, None] & (d < HEAD_DIM)[None, :]
            tl.store(
                out_ptr + o_off, out.to(out_ptr.dtype.element_ty), mask=valid
            )


def rotary_embedding(x, cos, sin, interleaved):
    num_tokens, num_heads, head_size = x.shape
    half = head_size // 2
    out = torch.empty_like(x)
    total_rows = num_tokens * num_heads
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
        block_r = 4
        num_tasks = (total_rows + block_r - 1) // block_r
        exact = 1 if (total_rows % block_r == 0 and half == pad_half) else 0
        _rotary_npu_v4_br4[(40,)](
            w,
            cos,
            sin,
            wo,
            num_tasks,
            total_rows,
            num_heads,
            w.stride(0),
            w.stride(1),
            cos.stride(0),
            cos.stride(1),
            sin.stride(0),
            sin.stride(1),
            wo.stride(0),
            wo.stride(1),
            half,
            pad_half,
            block_r,
            exact,
            num_warps=4,
            num_stages=1,
        )
        return out
    pad_d = 1 << (head_size - 1).bit_length()
    if pad_d < 16:
        pad_d = 16
    block_r = 8
    num_tasks = (total_rows + block_r - 1) // block_r
    exact = 1 if (total_rows % block_r == 0 and head_size == pad_d) else 0
    _rotary_npu_v3[(40,)](
        x,
        cos,
        sin,
        out,
        num_tasks,
        total_rows,
        num_heads,
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
        half,
        pad_d,
        block_r,
        exact,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["rotary_embedding"]
