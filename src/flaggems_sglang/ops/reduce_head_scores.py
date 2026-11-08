# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""Reduce the 32 DSA index head scores into one weighted score row.

``out[r, c] = bf16(sum_h bf16(max(scores[r, h, c], 0) * weights[r, h]))``

The accumulation order here is part of the contract, not an
implementation detail. Torch's CUDA ``Reduce.cuh`` rounds every head
product to bf16 before it enters the sum, walks the 32 heads with four
interleaved accumulators, and folds those four as ``((a0+a1)+a2)+a3``.
Reproducing that order is what makes this kernel bit-identical to the
reference rather than merely close, and the model's index scores were
produced against that order.

``width`` and the two row strides stay runtime scalars so one compiled
kernel serves every sequence length; only the column stride and the tiny
weight strides are constexpr.
"""

import torch
import triton
import triton.language as tl

NUM_HEADS = 32
BLOCK = 256


@triton.jit
def _head_product(
    scores,
    weights,
    row,
    columns,
    head: tl.constexpr,
    width,
    score_row,
    score_head,
    score_col: tl.constexpr,
    weight_row: tl.constexpr,
    weight_head: tl.constexpr,
):
    value = tl.load(
        scores + row * score_row + head * score_head + columns * score_col,
        columns < width,
        0,
    ).to(tl.float32)
    weight = tl.load(weights + row * weight_row + head * weight_head).to(
        tl.float32
    )
    return (tl.maximum(value, 0.0) * weight).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _reduce_head_scores_kernel(
    scores,
    weights,
    output,
    width,
    score_row,
    score_head,
    score_col: tl.constexpr,
    weight_row: tl.constexpr,
    weight_head: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc0 = tl.full((BLOCK,), 0.0, tl.float32)
    acc1 = tl.full((BLOCK,), 0.0, tl.float32)
    acc2 = tl.full((BLOCK,), 0.0, tl.float32)
    acc3 = tl.full((BLOCK,), 0.0, tl.float32)
    # Four interleaved accumulators match Torch Reduce.cuh (vt0=4) exactly.
    for group in tl.static_range(8):
        acc0 = acc0 + _head_product(
            scores,
            weights,
            row,
            columns,
            group * 4,
            width,
            score_row,
            score_head,
            score_col,
            weight_row,
            weight_head,
        )
        acc1 = acc1 + _head_product(
            scores,
            weights,
            row,
            columns,
            group * 4 + 1,
            width,
            score_row,
            score_head,
            score_col,
            weight_row,
            weight_head,
        )
        acc2 = acc2 + _head_product(
            scores,
            weights,
            row,
            columns,
            group * 4 + 2,
            width,
            score_row,
            score_head,
            score_col,
            weight_row,
            weight_head,
        )
        acc3 = acc3 + _head_product(
            scores,
            weights,
            row,
            columns,
            group * 4 + 3,
            width,
            score_row,
            score_head,
            score_col,
            weight_row,
            weight_head,
        )
    result = ((acc0 + acc1) + acc2) + acc3
    result = result.to(tl.bfloat16).to(tl.float32)
    tl.store(output + row * width + columns, result, columns < width)


def reduce_head_scores(
    scores: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Weighted sum of the 32 head scores, per row and per column.

    Args:
        scores: ``(rows, 32, width)`` bf16, raw index scores.
        weights: ``(rows, 32)`` bf16, one weight per head.

    Returns:
        ``(rows, width)`` float32 holding a bf16-rounded value.
    """
    if scores.ndim != 3:
        raise ValueError(
            f"scores must be 3-D (rows, heads, width), got {scores.ndim}-D"
        )
    rows, heads, width = scores.shape
    if tuple(weights.shape) != (rows, heads):
        raise ValueError(
            f"weights must be {(rows, heads)}, got {tuple(weights.shape)}"
        )
    if heads != NUM_HEADS:
        raise ValueError(
            f"the reduction order is defined for {NUM_HEADS} heads, "
            f"got {heads}"
        )
    if scores.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16:
        raise ValueError(
            "both inputs must be bfloat16 so the per-head product rounds "
            f"the same way, got {scores.dtype} and {weights.dtype}"
        )

    out = torch.empty((rows, width), dtype=torch.float32, device=scores.device)
    if rows == 0 or width == 0:
        return out

    _reduce_head_scores_kernel[(rows, triton.cdiv(width, BLOCK))](
        scores,
        weights,
        out,
        width,
        *scores.stride(),
        *weights.stride(),
        BLOCK,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["reduce_head_scores"]
