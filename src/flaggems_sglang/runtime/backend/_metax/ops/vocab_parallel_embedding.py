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

"""Pure-Triton vocab-parallel embedding lookup (v5)."""

import triton
import triton.language as tl


@triton.jit
def _vpe_even_impl(
    tok_ptr,  # int32/int64 token ids, indexed linearly (contiguous)
    weight_ptr,  # [local_vocab, hidden]
    out_ptr,  # [n_tokens, hidden]
    org_start,
    org_end,
    added_start,
    added_end,
    added_offset,  # added_start - (org_end - org_start) - padding
    hidden,
    stride_wv,  # weight.stride(0)
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    tok = tl.load(tok_ptr + pid_t).to(tl.int64)

    in_org = (tok >= org_start) & (tok < org_end)
    in_added = (tok >= added_start) & (tok < added_end)
    valid = in_org | in_added

    local_id = tl.where(in_org, tok - org_start, tok - added_offset)
    local_id = tl.where(valid, local_id, 0)

    # hidden % BLOCK_H == 0: no bounds check on the hidden axis.
    offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    vals = tl.load(
        weight_ptr + local_id * stride_wv + offs,
        mask=valid,
        other=0.0,
    )
    tl.store(out_ptr + pid_t.to(tl.int64) * hidden + offs, vals)


@triton.jit
def _vpe_masked_impl(
    tok_ptr,
    weight_ptr,
    out_ptr,
    org_start,
    org_end,
    added_start,
    added_end,
    added_offset,
    hidden,
    stride_wv,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    tok = tl.load(tok_ptr + pid_t).to(tl.int64)

    in_org = (tok >= org_start) & (tok < org_end)
    in_added = (tok >= added_start) & (tok < added_end)
    valid = in_org | in_added

    local_id = tl.where(in_org, tok - org_start, tok - added_offset)
    local_id = tl.where(valid, local_id, 0)

    offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hmask = offs < hidden
    vals = tl.load(
        weight_ptr + local_id * stride_wv + offs,
        mask=hmask & valid,
        other=0.0,
    )
    tl.store(out_ptr + pid_t.to(tl.int64) * hidden + offs, vals, mask=hmask)


@triton.jit(
    do_not_specialize=[
        "org_start",
        "org_end",
        "added_start",
        "added_end",
        "added_offset",
    ]
)
def _vpe_row2048(
    tok_ptr,
    weight_ptr,
    out_ptr,
    org_start,
    org_end,
    added_start,
    added_end,
    added_offset,
    hidden,
    stride_wv,
):
    _vpe_even_impl(
        tok_ptr,
        weight_ptr,
        out_ptr,
        org_start,
        org_end,
        added_start,
        added_end,
        added_offset,
        hidden,
        stride_wv,
        BLOCK_H=2048,
    )


@triton.jit(
    do_not_specialize=[
        "org_start",
        "org_end",
        "added_start",
        "added_end",
        "added_offset",
    ]
)
def _vpe_row1024(
    tok_ptr,
    weight_ptr,
    out_ptr,
    org_start,
    org_end,
    added_start,
    added_end,
    added_offset,
    hidden,
    stride_wv,
):
    _vpe_even_impl(
        tok_ptr,
        weight_ptr,
        out_ptr,
        org_start,
        org_end,
        added_start,
        added_end,
        added_offset,
        hidden,
        stride_wv,
        BLOCK_H=1024,
    )


@triton.jit(
    do_not_specialize=[
        "org_start",
        "org_end",
        "added_start",
        "added_end",
        "added_offset",
    ]
)
def _vpe_masked(
    tok_ptr,
    weight_ptr,
    out_ptr,
    org_start,
    org_end,
    added_start,
    added_end,
    added_offset,
    hidden,
    stride_wv,
    BLOCK_H: tl.constexpr,
):
    _vpe_masked_impl(
        tok_ptr,
        weight_ptr,
        out_ptr,
        org_start,
        org_end,
        added_start,
        added_end,
        added_offset,
        hidden,
        stride_wv,
        BLOCK_H,
    )


def _launch(jitfn, grid0, grid1, args, num_warps=1):
    """Standard JIT dispatch: compiles on first sight of a configuration
    and replays Triton's own kernel cache afterwards."""
    jitfn[(grid0, grid1)](*args, num_warps=num_warps)


def vocab_parallel_embedding(
    input_,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    n_tokens = input_.numel()
    hidden = weight.shape[1]

    out = weight.new_empty((*input_.shape, hidden))
    if n_tokens == 0:
        return out

    # The kernel indexes the token ids linearly, so a contiguous view of
    # any shape works directly; a non-contiguous input is normalized once.
    if not input_.is_contiguous():
        input_ = input_.contiguous()

    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    stride_wv = weight.stride(0)

    args = (
        input_,
        weight,
        out,
        org_vocab_start_index,
        org_vocab_end_index,
        added_vocab_start_index,
        added_vocab_end_index,
        added_offset,
        hidden,
        stride_wv,
    )

    # Pick a hidden chunk so the axis is covered mask-free when possible.
    if hidden % 2048 == 0:
        _launch(_vpe_row2048, n_tokens, hidden // 2048, args)
    elif hidden % 1024 == 0:
        _launch(_vpe_row1024, n_tokens, hidden // 1024, args)
    else:
        block_h = min(2048, triton.next_power_of_2(hidden))
        _launch(
            _vpe_masked,
            n_tokens,
            triton.cdiv(hidden, block_h),
            args + (block_h,),
        )
    return out


__all__ = ["vocab_parallel_embedding"]
