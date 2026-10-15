# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Align distinct expert IDs for single-token MoE dispatch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _align(
    Ids,
    Sorted,
    Experts,
    Post,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    FILL: tl.constexpr,
):
    i = tl.arange(0, R)
    expert = tl.load(Ids + i * STRIDE, i < K, other=0)
    smaller = (expert[None, :] < expert[:, None]) & (i[None, :] < K)
    rank = tl.sum(smaller.to(tl.int32), axis=1)
    tl.store(Experts + rank, expert, i < K)
    tl.store(Sorted + rank * BLOCK, i, i < K)
    p = tl.arange(0, FILL)
    tl.store(Sorted + p, K, (p < K * BLOCK) & (p % BLOCK != 0))
    tl.store(Post, K * BLOCK)


@triton.jit
def _align_per_expert(
    Ids,
    Sorted,
    Experts,
    Post,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    R: tl.constexpr,
    FILL: tl.constexpr,
):
    # One program owns one input expert and its complete output block.
    # The reduction produces a scalar block base, so no vector scatter or
    # K-by-K temporary is needed on larger shapes.
    slot = tl.program_id(0)
    if K > 0:
        expert = tl.load(Ids + slot * STRIDE)
        i = tl.arange(0, R)
        others = tl.load(Ids + i * STRIDE, i < K, other=0)
        rank = tl.sum(((others < expert) & (i < K)).to(tl.int32), axis=0)
        tl.store(Experts + rank, expert)
        p = tl.arange(0, FILL)
        tl.store(
            Sorted + rank * BLOCK + p, tl.where(p == 0, slot, K), p < BLOCK
        )
    if slot == 0:
        tl.store(Post, K * BLOCK)


def moe_align_single_token(topk_ids, block_size):
    k = topk_ids.shape[1]
    sorted_ids = torch.empty(
        (k * block_size,), dtype=torch.int32, device=topk_ids.device
    )
    experts = torch.empty((k,), dtype=torch.int32, device=topk_ids.device)
    post = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    if k <= 32 and k * block_size <= 4096:
        _align[(1,)](
            topk_ids,
            sorted_ids,
            experts,
            post,
            topk_ids.stride(1),
            # Avoid the singleton reduction-derived address rejected by
            # Ascend's pointer analysis. The extra lane is masked out.
            k,
            block_size,
            triton.next_power_of_2(max(k, 2)),
            triton.next_power_of_2(max(k * block_size, 1)),
            num_warps=4,
        )
    else:
        _align_per_expert[(max(k, 1),)](
            topk_ids,
            sorted_ids,
            experts,
            post,
            topk_ids.stride(1),
            k,
            block_size,
            triton.next_power_of_2(max(k, 2)),
            triton.next_power_of_2(max(block_size, 1)),
            num_warps=4,
        )
    return sorted_ids, experts, post


__all__ = ["moe_align_single_token"]
