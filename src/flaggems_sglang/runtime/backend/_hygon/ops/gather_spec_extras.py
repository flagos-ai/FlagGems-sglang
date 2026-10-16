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

_NARROW_BATCH = 256


@triton.jit
def _gather_narrow(
    idx_ptr,
    tp_ptr,
    ti_ptr,
    tok_ptr,
    o_tp_ptr,
    o_ti_ptr,
    o_tok_ptr,
    rows,
    rmask,
    TOPK: tl.constexpr,
    EVEN_M: tl.constexpr,
):

    if EVEN_M:
        idx = tl.load(idx_ptr + rows).to(tl.int32)
    else:
        idx = tl.load(idx_ptr + rows, mask=rmask, other=0).to(tl.int32)

    cols = tl.arange(0, TOPK)
    src2 = idx[:, None] * TOPK + cols[None, :]
    dst2 = rows[:, None] * TOPK + cols[None, :]

    if EVEN_M:
        tl.store(o_tp_ptr + dst2, tl.load(tp_ptr + src2))
        tl.store(o_ti_ptr + dst2, tl.load(ti_ptr + src2))
        tl.store(o_tok_ptr + rows, tl.load(tok_ptr + idx))
    else:
        mk2 = rmask[:, None]
        tl.store(
            o_tp_ptr + dst2,
            tl.load(tp_ptr + src2, mask=mk2, other=0.0),
            mask=mk2,
        )
        tl.store(
            o_ti_ptr + dst2,
            tl.load(ti_ptr + src2, mask=mk2, other=0),
            mask=mk2,
        )
        tl.store(
            o_tok_ptr + rows,
            tl.load(tok_ptr + idx, mask=rmask, other=0),
            mask=rmask,
        )


@triton.jit
def _gather_wide(
    idx_ptr,
    hid_ptr,
    o_hid_ptr,
    rows,
    rmask,
    pid_h,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_H: tl.constexpr,
):

    if EVEN_M:
        idx = tl.load(idx_ptr + rows).to(tl.int32)
    else:
        idx = tl.load(idx_ptr + rows, mask=rmask, other=0).to(tl.int32)

    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    src = idx[:, None] * HIDDEN + cols[None, :]
    dst = rows[:, None] * HIDDEN + cols[None, :]

    if EVEN_M and EVEN_H:
        tl.store(o_hid_ptr + dst, tl.load(hid_ptr + src))
    else:
        cmask = rmask[:, None] & (cols[None, :] < HIDDEN)
        tl.store(
            o_hid_ptr + dst,
            tl.load(hid_ptr + src, mask=cmask, other=0.0),
            mask=cmask,
        )


@triton.jit
def _gather_spec_extras_kernel(
    idx_ptr,
    tp_ptr,
    ti_ptr,
    tok_ptr,
    hid_ptr,
    o_tp_ptr,
    o_ti_ptr,
    o_tok_ptr,
    o_hid_ptr,
    m,
    HIDDEN: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_H: tl.constexpr,
):
    N_H: tl.constexpr = (HIDDEN + BLOCK_H - 1) // BLOCK_H if HAS_HIDDEN else 0
    pid = tl.program_id(0)
    pid_m = pid // (N_H + 1)
    pid_h = pid % (N_H + 1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < m

    if pid_h == N_H:
        _gather_narrow(
            idx_ptr,
            tp_ptr,
            ti_ptr,
            tok_ptr,
            o_tp_ptr,
            o_ti_ptr,
            o_tok_ptr,
            rows,
            rmask,
            TOPK=TOPK,
            EVEN_M=EVEN_M,
        )
    elif HAS_HIDDEN:
        _gather_wide(
            idx_ptr,
            hid_ptr,
            o_hid_ptr,
            rows,
            rmask,
            pid_h,
            HIDDEN=HIDDEN,
            BLOCK_H=BLOCK_H,
            EVEN_M=EVEN_M,
            EVEN_H=EVEN_H,
        )


def _pick_config(m, hidden_size, has_hidden):
    if not has_hidden:
        return 64, 1024, 2
    if m < _NARROW_BATCH:
        return 2, 2048, 1
    return 1, 2048, 2


def gather_spec_extras(
    indices, topk_p_buf, topk_index_buf, output_tokens_buf, hidden_states_buf
):
    if not indices.is_contiguous():
        indices = indices.contiguous()
    if not topk_p_buf.is_contiguous():
        topk_p_buf = topk_p_buf.contiguous()
    if not topk_index_buf.is_contiguous():
        topk_index_buf = topk_index_buf.contiguous()
    if not output_tokens_buf.is_contiguous():
        output_tokens_buf = output_tokens_buf.contiguous()

    has_hidden = hidden_states_buf is not None
    if has_hidden and not hidden_states_buf.is_contiguous():
        hidden_states_buf = hidden_states_buf.contiguous()

    m = indices.shape[0]
    device = indices.device

    topk = topk_p_buf.shape[1]
    out_topk_p = torch.empty((m, topk), dtype=topk_p_buf.dtype, device=device)
    out_topk_index = torch.empty(
        (m, topk), dtype=topk_index_buf.dtype, device=device
    )
    out_tokens = torch.empty(
        (m,), dtype=output_tokens_buf.dtype, device=device
    )

    if m == 0:
        hidden = (
            torch.empty(
                (0, hidden_states_buf.shape[1]),
                dtype=hidden_states_buf.dtype,
                device=device,
            )
            if has_hidden
            else None
        )
        return out_topk_p, out_topk_index, out_tokens, hidden

    if has_hidden:
        hidden_size = hidden_states_buf.shape[1]
        out_hidden = torch.empty(
            (m, hidden_size), dtype=hidden_states_buf.dtype, device=device
        )
        hid_src = hidden_states_buf
        hid_dst = out_hidden
    else:
        hidden_size = 1
        hid_src = topk_p_buf
        hid_dst = out_topk_p
        out_hidden = None

    block_m, block_h, num_warps = _pick_config(m, hidden_size, has_hidden)
    n_h = triton.cdiv(hidden_size, block_h) if has_hidden else 0
    grid = (triton.cdiv(m, block_m) * (n_h + 1),)

    _gather_spec_extras_kernel[grid](
        indices,
        topk_p_buf,
        topk_index_buf,
        output_tokens_buf,
        hid_src,
        out_topk_p,
        out_topk_index,
        out_tokens,
        hid_dst,
        m,
        HIDDEN=hidden_size,
        TOPK=topk,
        HAS_HIDDEN=has_hidden,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        EVEN_M=(m % block_m == 0),
        EVEN_H=(has_hidden and hidden_size % block_h == 0),
        num_warps=num_warps,
    )

    return out_topk_p, out_topk_index, out_tokens, out_hidden


__all__ = ["gather_spec_extras"]
