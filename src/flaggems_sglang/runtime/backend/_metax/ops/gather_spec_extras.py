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

"""gather_spec_extras -- fused speculative-decoding side-buffer gather."""

import torch
import triton
import triton.language as tl

_NARROW_AT = 4


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


@triton.jit
def _gather_spec_kernel(
    idx_ptr,  # [m] shared row indices
    p_ptr,  # [pool, topk]   probabilities buffer
    index_ptr,  # [pool, topk]   token-id buffer
    tok_ptr,  # [pool]         bonus-token buffer
    hid_ptr,  # [pool, hidden] hidden-state buffer
    out_p_ptr,  # [m, topk]      probabilities output
    out_index_ptr,  # [m, topk]      token-id output
    out_tok_ptr,  # [m]            bonus-token output
    out_hid_ptr,  # [m, hidden]
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    STRIDE_P: tl.constexpr,
    STRIDE_INDEX: tl.constexpr,
    STRIDE_HID: tl.constexpr,
    HAS_HIDDEN: tl.constexpr,
    NARROW_AT: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    src = tl.load(idx_ptr + row).to(tl.int64)

    if chunk == NARROW_AT:
        cols = tl.arange(0, BLOCK_TOPK)
        if TOPK == BLOCK_TOPK:
            # Exact-width row: masks are dead code, drop them.
            vals = tl.load(p_ptr + src * STRIDE_P + cols)
            tl.store(out_p_ptr + row * TOPK + cols, vals)
            ids = tl.load(index_ptr + src * STRIDE_INDEX + cols)
            tl.store(out_index_ptr + row * TOPK + cols, ids)
        else:
            cmask = cols < TOPK
            vals = tl.load(
                p_ptr + src * STRIDE_P + cols, mask=cmask, other=0.0
            )
            tl.store(out_p_ptr + row * TOPK + cols, vals, mask=cmask)
            ids = tl.load(
                index_ptr + src * STRIDE_INDEX + cols, mask=cmask, other=0
            )
            tl.store(out_index_ptr + row * TOPK + cols, ids, mask=cmask)
        tl.store(out_tok_ptr + row, tl.load(tok_ptr + src))
    elif HAS_HIDDEN:
        hchunk = chunk - (1 if chunk > NARROW_AT else 0)
        hoffs = hchunk * BLOCK_H + tl.arange(0, BLOCK_H)
        if HIDDEN % BLOCK_H == 0:
            hv = tl.load(hid_ptr + src * STRIDE_HID + hoffs)
            tl.store(out_hid_ptr + row * HIDDEN + hoffs, hv)
        else:
            hmask = hoffs < HIDDEN
            hv = tl.load(
                hid_ptr + src * STRIDE_HID + hoffs, mask=hmask, other=0
            )
            tl.store(out_hid_ptr + row * HIDDEN + hoffs, hv, mask=hmask)


def gather_spec_extras(
    indices, topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf
):
    indices = indices.contiguous()
    m = indices.numel()

    topk = topk_p_buf.shape[-1]
    has_hidden = hidden_states_buf is not None
    if has_hidden:
        hidden = hidden_states_buf.shape[-1]
        block_h = min(1024, _next_pow2(max(hidden, 1)))
        out_hid = torch.empty(
            (m, hidden),
            dtype=hidden_states_buf.dtype,
            device=hidden_states_buf.device,
        )
    else:
        hidden = 0
        block_h = 1
        out_hid = None

    out_p = torch.empty(
        (m, topk), dtype=topk_p_buf.dtype, device=topk_p_buf.device
    )
    out_index = torch.empty(
        (m, topk), dtype=topk_index_buf.dtype, device=topk_index_buf.device
    )
    out_tok = torch.empty(
        (m,), dtype=output_tokens_buf.dtype, device=output_tokens_buf.device
    )

    if m == 0:
        return (out_p, out_index, out_tok, out_hid)

    # The narrow lanes always get a dedicated chunk slot (see module
    # docstring); when hidden is absent the grid is a single narrow chunk.
    if has_hidden:
        num_chunks = triton.cdiv(hidden, block_h) + 1
    else:
        num_chunks = 1
    narrow_at = _NARROW_AT if has_hidden else 0

    block_topk = _next_pow2(max(topk, 1))
    _gather_spec_kernel[(m, num_chunks)](
        indices,
        topk_p_buf,
        topk_index_buf,
        output_tokens_buf,
        # Dummy valid pointer -- never dereferenced because HAS_HIDDEN == 0.
        hidden_states_buf if has_hidden else output_tokens_buf,
        out_p,
        out_index,
        out_tok,
        out_hid if has_hidden else output_tokens_buf,
        TOPK=topk,
        HIDDEN=hidden,
        STRIDE_P=topk_p_buf.stride(0),
        STRIDE_INDEX=topk_index_buf.stride(0),
        STRIDE_HID=hidden_states_buf.stride(0) if has_hidden else 0,
        HAS_HIDDEN=has_hidden,
        NARROW_AT=narrow_at,
        BLOCK_TOPK=block_topk,
        BLOCK_H=block_h,
        num_warps=1,
    )

    return (out_p, out_index, out_tok, out_hid)


__all__ = ["gather_spec_extras"]
