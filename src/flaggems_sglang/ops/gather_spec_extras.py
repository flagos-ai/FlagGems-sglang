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
def _gather_spec_extras_kernel(
    idx_ptr,
    p_buf,
    i_buf,
    t_buf,
    h_buf,
    p_out,
    i_out,
    t_out,
    h_out,
    topk,
    hidden,
    stride_p,
    stride_i,
    stride_h,
    HAS_HIDDEN: tl.constexpr,
    EVEN_TK: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    pid_h = tl.program_id(1)
    src = tl.load(idx_ptr + row)

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_base = h_buf + src * stride_h
    if HAS_HIDDEN:
        hmask = offs_h < hidden
        hv = tl.load(h_base + offs_h, mask=hmask, cache_modifier=".cv")

    if pid_h == 0:
        offs_t = tl.arange(0, BLOCK_TK)
        p_base = p_buf + src * stride_p
        i_base = i_buf + src * stride_i
        if EVEN_TK:
            tl.store(p_out + row * topk + offs_t, tl.load(p_base + offs_t))
            tl.store(i_out + row * topk + offs_t, tl.load(i_base + offs_t))
        else:
            tmask = offs_t < topk
            tl.store(
                p_out + row * topk + offs_t,
                tl.load(p_base + offs_t, mask=tmask),
                mask=tmask,
            )
            tl.store(
                i_out + row * topk + offs_t,
                tl.load(i_base + offs_t, mask=tmask),
                mask=tmask,
            )
        tl.store(t_out + row, tl.load(t_buf + src))

    if HAS_HIDDEN:
        tl.store(h_out + row * hidden + offs_h, hv, mask=hmask)


_BLOCK_H = 1024
_NUM_WARPS = 8
_NUM_STAGES = 1


def gather_spec_extras(
    indices, topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf
):
    idx = indices.contiguous()
    if idx.dtype != torch.int64:
        idx = idx.long()

    p_buf = (
        topk_p_buf if topk_p_buf.is_contiguous() else topk_p_buf.contiguous()
    )
    i_buf = (
        topk_index_buf
        if topk_index_buf.is_contiguous()
        else topk_index_buf.contiguous()
    )
    t_buf = (
        output_tokens_buf
        if output_tokens_buf.is_contiguous()
        else output_tokens_buf.contiguous()
    )

    topk = p_buf.shape[1]
    m = idx.shape[0]

    out_p = torch.empty((m, topk), dtype=p_buf.dtype, device=p_buf.device)
    out_i = torch.empty((m, topk), dtype=i_buf.dtype, device=i_buf.device)
    out_t = torch.empty((m,), dtype=t_buf.dtype, device=t_buf.device)

    has_hidden = hidden_states_buf is not None
    if has_hidden:
        h_buf = (
            hidden_states_buf
            if hidden_states_buf.is_contiguous()
            else hidden_states_buf.contiguous()
        )
        hidden = h_buf.shape[1]
        out_h = torch.empty(
            (m, hidden), dtype=h_buf.dtype, device=h_buf.device
        )
        h_chunks = triton.cdiv(hidden, _BLOCK_H)
        stride_h = h_buf.stride(0)
    else:
        h_buf = p_buf
        out_h = out_p
        hidden = 0
        h_chunks = 1
        stride_h = 1

    block_tk = triton.next_power_of_2(topk)

    _gather_spec_extras_kernel[(m, h_chunks)](
        idx,
        p_buf,
        i_buf,
        t_buf,
        h_buf,
        out_p,
        out_i,
        out_t,
        out_h,
        topk,
        hidden,
        p_buf.stride(0),
        i_buf.stride(0),
        stride_h,
        HAS_HIDDEN=has_hidden,
        EVEN_TK=(block_tk == topk),
        BLOCK_TK=block_tk,
        BLOCK_H=_BLOCK_H,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    return out_p, out_i, out_t, (out_h if has_hidden else None)


__all__ = ["gather_spec_extras"]
