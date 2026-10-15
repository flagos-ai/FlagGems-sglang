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
def _align_per_expert(
    Ids,
    Sorted,
    Experts,
    Post,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    FILL: tl.constexpr,
):
    slot = tl.program_id(0)
    if K > 0:
        expert = tl.load(Ids + slot * STRIDE)
        # Keep rank scalar. A masked, short vector reduction on the XPU
        # backend can incorrectly restrict the following wider block store
        # to the cores participating in the input load.
        rank = tl.full((), 0, tl.int32)
        for other_slot in range(K):
            other = tl.load(Ids + other_slot * STRIDE)
            rank += (other < expert).to(tl.int32)
        # Distinct expert IDs give each program its own output block.
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
    _align_per_expert[(max(k, 1),)](
        topk_ids,
        sorted_ids,
        experts,
        post,
        topk_ids.stride(1),
        k,
        block_size,
        triton.next_power_of_2(max(block_size, 1)),
        num_warps=4,
    )
    return sorted_ids, experts, post


__all__ = ["moe_align_single_token"]
