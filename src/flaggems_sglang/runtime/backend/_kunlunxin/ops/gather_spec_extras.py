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

"""Operator: speculative/gather_spec_extras"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_spec_kernel(
    indices_ptr,
    p_ptr,
    idx_ptr,
    tok_ptr,
    hid_ptr,
    out_p_ptr,
    out_idx_ptr,
    out_tok_ptr,
    out_hid_ptr,
    stride_p,
    stride_idx,
    stride_hid,
    hidden,
    topk,
    TOPK: tl.constexpr,
    HAS_HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Shared row index: loaded once, reused for every buffer.
    row = tl.load(indices_ptr + pid_m)

    if pid_h == 0:
        ko = tl.arange(0, TOPK)
        km = ko < topk
        p = tl.load(p_ptr + row * stride_p + ko, mask=km, other=0)
        tl.store(out_p_ptr + pid_m * TOPK + ko, p, mask=km)
        iv = tl.load(idx_ptr + row * stride_idx + ko, mask=km, other=0)
        tl.store(out_idx_ptr + pid_m * TOPK + ko, iv, mask=km)
        t = tl.load(tok_ptr + row)
        tl.store(out_tok_ptr + pid_m, t)

    if HAS_HIDDEN:
        ho = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        hm = ho < hidden
        h = tl.load(hid_ptr + row * stride_hid + ho, mask=hm, other=0)
        tl.store(out_hid_ptr + pid_m * hidden + ho, h, mask=hm)


def gather_spec_extras(
    indices, topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf
):
    """Gather the speculative side buffers with one shared row index."""
    idx = indices.contiguous().long()
    m = idx.numel()

    topk_p_buf = topk_p_buf.contiguous()
    topk_index_buf = topk_index_buf.contiguous()
    output_tokens_buf = output_tokens_buf.contiguous()

    topk = topk_p_buf.shape[1]
    out_p = torch.empty(
        (m, topk), dtype=topk_p_buf.dtype, device=topk_p_buf.device
    )
    out_i = torch.empty(
        (m, topk), dtype=topk_index_buf.dtype, device=topk_index_buf.device
    )
    out_t = torch.empty(
        (m,), dtype=output_tokens_buf.dtype, device=output_tokens_buf.device
    )

    has_hidden = hidden_states_buf is not None
    if has_hidden:
        hidden_states_buf = hidden_states_buf.contiguous()
        hidden = hidden_states_buf.shape[1]
        out_h = torch.empty(
            (m, hidden),
            dtype=hidden_states_buf.dtype,
            device=hidden_states_buf.device,
        )
        # Power-of-2 hidden tile: small for tiny m so the row still spreads
        # over several programs, larger for big m to cut program count.
        block_h = 1024 if m < 64 else 2048
        grid_h = triton.cdiv(hidden, block_h)
    else:
        hidden = 0
        out_h = None
        block_h = 1024
        grid_h = 1

    if m == 0:
        return out_p, out_i, out_t, out_h

    _gather_spec_kernel[(m, grid_h)](
        idx,
        topk_p_buf,
        topk_index_buf,
        output_tokens_buf,
        hidden_states_buf,
        out_p,
        out_i,
        out_t,
        out_h,
        topk_p_buf.stride(0),
        topk_index_buf.stride(0),
        hidden_states_buf.stride(0) if has_hidden else 0,
        hidden,
        topk,
        TOPK=triton.next_power_of_2(topk),
        HAS_HIDDEN=has_hidden,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out_p, out_i, out_t, out_h


__all__ = ["gather_spec_extras"]
