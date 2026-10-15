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
FIXED_BLOCK = 2048


@triton.jit
def _strided(
    D,
    O,
    I,
    E,
    W,
    H: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    NLE: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    B: tl.constexpr,
    KP: tl.constexpr,
):
    token = tl.program_id(0)
    start = tl.program_id(1) * B
    acc = tl.full((B,), 0, tl.float32)
    lanes = tl.arange(0, KP)
    live = lanes < K
    dv = tl.load(I + token * I0 + lanes * I1, live, other=0).to(tl.int32)
    ev = tl.load(E + token * E0 + lanes * E1, live, other=NLE).to(tl.int32)
    wv = (
        tl.load(W + token * W0 + lanes * W1, live, other=0).to(tl.float32)
        * SCALE
    )
    wv = tl.where(ev != NLE, wv, 0.0)
    rv = tl.maximum(dv, 0)
    for slot in tl.static_range(K):
        pick = lanes == slot
        row = tl.sum(tl.where(pick, rv, 0))
        w = tl.sum(tl.where(pick, wv, 0.0))
        ptr = tl.make_block_ptr(
            D + row * D0, (H,), (D1,), (start,), (B,), (0,)
        )
        val = tl.load(
            ptr,
            boundary_check=(0,),
            padding_option="zero",
            cache_modifier=".cg",
        ).to(tl.float32)
        acc = acc + val * w
    ptr = tl.make_block_ptr(O + token * O0, (H,), (O1,), (start,), (B,), (0,))
    tl.store(ptr, acc.to(O.dtype.element_ty), boundary_check=(0,))


@triton.jit
def _contiguous(
    D,
    O,
    I,
    E,
    W,
    H: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    NLE: tl.constexpr,
    B: tl.constexpr,
    KP: tl.constexpr,
):
    t = tl.program_id(0)
    start = tl.program_id(1) * B
    acc = tl.full((B,), 0, tl.float32)
    lanes = tl.arange(0, KP)
    live = lanes < K
    dv = tl.load(I + t * K + lanes, live, other=0).to(tl.int32)
    ev = tl.load(E + t * K + lanes, live, other=NLE).to(tl.int32)
    wv = tl.load(W + t * K + lanes, live, other=0).to(tl.float32) * SCALE
    wv = tl.where(ev != NLE, wv, 0.0)
    rv = tl.maximum(dv, 0)
    for slot in tl.static_range(K):
        pick = lanes == slot
        row = tl.sum(tl.where(pick, rv, 0))
        w = tl.sum(tl.where(pick, wv, 0.0))
        ptr = tl.make_block_ptr(D + row * H, (H,), (1,), (start,), (B,), (0,))
        v = tl.load(
            ptr,
            boundary_check=(0,),
            padding_option="zero",
            cache_modifier=".cg",
        ).to(tl.float32)
        acc = acc + v * w
    ptr = tl.make_block_ptr(O + t * H, (H,), (1,), (start,), (B,), (0,))
    tl.store(ptr, acc.to(O.dtype.element_ty), boundary_check=(0,))


def _block_for(num_tokens, hidden_size):
    cap = triton.next_power_of_2(hidden_size)
    if FIXED_BLOCK:
        return min(cap, FIXED_BLOCK)
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
    out = torch.empty_like(output)
    if num_tokens == 0 or hidden_size == 0:
        return out
    b = _block_for(num_tokens, hidden_size)
    nt = (hidden_size + b - 1) // b
    scale = float(routed_scaling_factor)
    kp = triton.next_power_of_2(topk)
    if (
        down_output.stride() == (hidden_size, 1)
        and out.stride() == (hidden_size, 1)
        and src2dst.stride() == (topk, 1)
        and topk_ids.stride() == (topk, 1)
        and topk_weights.stride() == (topk, 1)
    ):
        _contiguous[(num_tokens, nt)](
            down_output,
            out,
            src2dst,
            topk_ids,
            topk_weights,
            hidden_size,
            topk,
            scale,
            num_local_experts,
            b,
            kp,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        _strided[(num_tokens, nt)](
            down_output,
            out,
            src2dst,
            topk_ids,
            topk_weights,
            hidden_size,
            topk,
            scale,
            num_local_experts,
            *down_output.stride(),
            *out.stride(),
            *src2dst.stride(),
            *topk_ids.stride(),
            *topk_weights.stride(),
            b,
            kp,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["post_reorder_cutlass"]
