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


@triton.jit
def _gather(
    D,
    I,
    E,
    W,
    X,
    TOTAL: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    NLE: tl.constexpr,
    ROWS: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    I0: tl.constexpr,
    I1: tl.constexpr,
    E0: tl.constexpr,
    E1: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.program_id(0) * B + tl.arange(0, B)
    keep = x < TOTAL
    token = x // H // K
    slot = x // H % K
    col = x % H
    row = tl.load(I + token * I0 + slot * I1, keep, other=0).to(tl.int32)
    row = tl.where(row < 0, row + ROWS, row)
    eid = tl.load(E + token * E0 + slot * E1, keep, other=NLE)
    w = tl.load(W + token * W0 + slot * W1, keep, other=0).to(tl.float32)
    w = tl.where(eid != NLE, w, 0.0)
    v = tl.load(D + row * D0 + col * D1, keep, other=0).to(tl.float32)
    tl.store(X + x, v * w, keep)


@triton.jit
def _reduce(
    X,
    O,
    TOTAL: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    B: tl.constexpr,
):
    x = tl.program_id(0) * B + tl.arange(0, B)
    keep = x < TOTAL
    t = x // H
    col = x % H
    acc = tl.zeros((B,), tl.float32)
    for slot in tl.static_range(K):
        v = tl.load(X + (t * K + slot) * H + col, keep, other=0)
        acc += v
    tl.store(O + x, acc * S, keep)


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
    total = num_tokens * hidden_size
    if total == 0:
        return out
    scratch = output.new_empty((total * topk,), dtype=torch.float32)
    if topk > 0:
        _gather[((total * topk + 8191) // 8192,)](
            down_output,
            src2dst,
            topk_ids,
            topk_weights,
            scratch,
            total * topk,
            hidden_size,
            topk,
            num_local_experts,
            down_output.shape[0],
            *down_output.stride(),
            *src2dst.stride(),
            *topk_ids.stride(),
            *topk_weights.stride(),
            8192,
            num_warps=4,
            num_stages=1,
            enable_fp_fusion=False,
        )
    _reduce[((total + 8191) // 8192,)](
        scratch,
        out,
        total,
        hidden_size,
        topk,
        float(routed_scaling_factor),
        8192,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["post_reorder_cutlass"]
