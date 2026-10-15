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

BLOCKS = (2048, 1024, 512, 256, 128)
TARGET_PROGRAMS = 1024


@triton.jit
def _fast(
    D,
    OT,
    IT,
    E,
    W,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    NLE: tl.constexpr,
    B: tl.constexpr,
):
    t = tl.program_id(0)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    mask = cols < H
    acc = tl.zeros([B], dtype=tl.float32)
    for i in range(K):
        eid = tl.load(E + t * K + i)
        dst = tl.load(IT + t * K + i)
        w = tl.load(W + t * K + i).to(tl.float32)
        w = tl.where(eid != NLE, w, 0.0)
        row = tl.maximum(dst, 0)
        v = tl.load(D + row * H + cols, mask=mask, other=0.0)
        acc += v.to(tl.float32) * w
    tl.store(OT + t * H + cols, acc * scale, mask=mask)


@triton.jit
def _post_reorder_kernel(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    ids_ptr,
    weight_ptr,
    scale,
    N: tl.constexpr,
    SD: tl.constexpr,
    SD1: tl.constexpr,
    SO: tl.constexpr,
    SS: tl.constexpr,
    SS1: tl.constexpr,
    SI: tl.constexpr,
    SI1: tl.constexpr,
    SW: tl.constexpr,
    SW1: tl.constexpr,
    ROWS: tl.constexpr,
    NLE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for i in range(TOPK):
        eid = tl.load(ids_ptr + t * SI + i * SI1)
        dst = tl.load(src2dst_ptr + t * SS + i * SS1)
        w = tl.load(weight_ptr + t * SW + i * SW1).to(tl.float32)
        w = tl.where(eid != NLE, w, 0.0)
        row = tl.where(dst < 0, dst + ROWS, dst)
        v = tl.load(down_ptr + row * SD + cols * SD1, mask=mask, other=0.0)
        acc += v.to(tl.float32) * w
    tl.store(out_ptr + t * SO + cols, acc * scale, mask=mask)


def _block_for(num_tokens, hidden_size):
    cap = triton.next_power_of_2(hidden_size)
    for b in BLOCKS:
        if (
            b <= cap
            and num_tokens * ((hidden_size + b - 1) // b) >= TARGET_PROGRAMS
        ):
            return b
    return min(cap, BLOCKS[-1])


def post_reorder_cutlass(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    num_local_experts,
    topk,
    num_tokens,
    hidden_size,
    routed_scaling_factor,
):
    out = output.new_empty((num_tokens, hidden_size))
    if num_tokens == 0 or hidden_size == 0:
        return out
    block_n = _block_for(num_tokens, hidden_size)
    grid = (num_tokens, triton.cdiv(hidden_size, block_n))
    if (
        down_output.is_contiguous()
        and out.is_contiguous()
        and src2dst.is_contiguous()
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
    ):
        _fast[grid](
            down_output,
            out,
            src2dst,
            topk_ids,
            topk_weights,
            float(routed_scaling_factor),
            hidden_size,
            topk,
            num_local_experts,
            block_n,
            num_warps=4,
        )
    else:
        idx = src2dst.to(torch.int32)
        ids = topk_ids.to(torch.int32)
        _post_reorder_kernel[grid](
            down_output,
            out,
            idx,
            ids,
            topk_weights,
            float(routed_scaling_factor),
            hidden_size,
            down_output.stride(0),
            down_output.stride(1),
            out.stride(0),
            idx.stride(0),
            idx.stride(1),
            ids.stride(0),
            ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            down_output.shape[0],
            num_local_experts,
            topk,
            block_n,
        )
    return out


__all__ = ["post_reorder_cutlass"]
