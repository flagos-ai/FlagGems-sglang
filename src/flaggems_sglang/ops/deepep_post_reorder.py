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

"""Operator: moe/deepep_post_reorder.

DeepEP MoE epilogue: output[t] = sum_i down_output[src2dst[t, i]] *
(topk_weights[t, i] * routed_scaling_factor) over valid slots, one
fused output-driven pass, fp32 accumulate, zero host syncs.
"""

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Bandwidth-bound winner (T=2048: 229.1 us, T=512: 61.0 us).
        triton.Config({"BLOCK_H": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
        # Latency-bound winner (T<=64); 128 doubles CTA count again for the
        # single-token cases where occupancy, not bandwidth, is the limit.
        triton.Config({"BLOCK_H": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 128}, num_warps=2, num_stages=1),
    ],
    key=["T", "TOPK"],
)
@triton.jit
def _post_reorder_kernel(
    down_ptr,  # [num_tokens * topk, hidden] expert outputs
    out_ptr,  # [num_tokens, hidden] merged output (written in place)
    src2dst_ptr,  # [num_tokens, topk] int32 destination rows (-1 = skip)
    weights_ptr,  # [num_tokens, topk] float32 routing weights
    T,  # num_tokens (autotune key only)
    scale,  # routed_scaling_factor (fp32)
    H: tl.constexpr,  # hidden_size (compile-time: folds address math)
    TOPK: tl.constexpr,  # exact topk: unrolled loop + folded slot strides
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    # Flattened schedule: consecutive pids cover all hidden chunks of one
    # token before moving to the next, keeping each token's src2dst/weights
    # lines and gathered rows shared in-flight across concurrent CTAs. NB
    # folds at compile time since H and BLOCK_H are both constexpr.
    NB: tl.constexpr = (H + BLOCK_H - 1) // BLOCK_H
    t = pid // NB
    hblk = pid % NB

    offs = hblk * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = offs < H

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    # v10 change vs v9 — software-pipeline the per-slot scalar index/weight
    # loads. In v9 each loop iteration was ``load dst_i -> gather row_i``: an
    # eight-deep dependent-latency chain (scalar round trip, then the gather
    # it feeds). Here slot i+1's scalar loads are issued *before* slot i's
    # gather, so after the first iteration every scalar round trip overlaps
    # the previous gather and the gathers' addresses are always ready up
    # front. (TOPK is constexpr, so the ``i + 1 < TOPK`` guard folds away and
    # the fully unrolled schedule is identical to hoisting all TOPK loads
    # ahead of all TOPK gathers — memory-level parallelism, same traffic.)
    dst_cur = tl.load(src2dst_ptr + t * TOPK)
    w_cur = tl.load(weights_ptr + t * TOPK)
    for i in tl.static_range(TOPK):
        dst_i = dst_cur
        w_i = w_cur
        if i + 1 < TOPK:
            dst_cur = tl.load(src2dst_ptr + t * TOPK + i + 1)
            w_cur = tl.load(weights_ptr + t * TOPK + i + 1)
        # Mimic the reference: weight is rounded to down_output.dtype before
        # the fp32 scale multiply.
        w_i = w_i.to(down_ptr.dtype.element_ty).to(tl.float32) * scale
        valid = dst_i >= 0
        # Clamp the address base so masked-off lanes never form a wild pointer.
        base = tl.where(valid, dst_i, 0).to(tl.int64) * H
        x = tl.load(
            down_ptr + base + offs,
            mask=valid & hm,
            other=0.0,
            eviction_policy="evict_first",
        )
        acc += x.to(tl.float32) * w_i

    # The output row is written once and never re-read: stream it out too.
    tl.store(
        out_ptr + t.to(tl.int64) * H + offs,
        acc.to(out_ptr.dtype.element_ty),
        mask=hm,
        eviction_policy="evict_first",
    )


def deepep_post_reorder(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    hidden_size,
    routed_scaling_factor,
):
    num_tokens = src2dst.shape[0]
    if num_tokens == 0 or hidden_size == 0 or topk == 0:
        return output

    def grid(meta):
        nb = triton.cdiv(hidden_size, meta["BLOCK_H"])
        return (num_tokens * nb,)

    _post_reorder_kernel[grid](
        down_output,
        output,
        src2dst,
        topk_weights,
        num_tokens,
        float(routed_scaling_factor),
        H=hidden_size,
        TOPK=topk,
    )
    return output


__all__ = ["deepep_post_reorder"]
