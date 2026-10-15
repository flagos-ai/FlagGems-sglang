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
def _general(
    D,
    O,
    I,
    E,
    W,
    H: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
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
    MASKED: tl.constexpr,
    NTP: tl.constexpr,
):
    pid = tl.program_id(0)
    t = pid // NTP
    cols = pid % NTP * B + tl.arange(0, B)
    keep = cols < H
    acc = tl.zeros((B,), tl.float32)
    for slot in tl.static_range(K):
        dst = tl.load(I + t * I0 + slot * I1).to(tl.int32)
        eid = tl.load(E + t * E0 + slot * E1).to(tl.int32)
        w = tl.load(W + t * W0 + slot * W1).to(tl.float32) * S
        w = tl.where(eid != NLE, w, 0.0)
        base = D + tl.maximum(dst, 0) * D0 + cols * D1
        if MASKED:
            v = tl.load(base, mask=keep, other=0.0).to(tl.float32)
        else:
            v = tl.load(base).to(tl.float32)
        acc += v * w
    if MASKED:
        tl.store(O + t * O0 + cols * O1, acc.to(O.dtype.element_ty), mask=keep)
    else:
        tl.store(O + t * O0 + cols * O1, acc.to(O.dtype.element_ty))


def _tiling(n, hidden_size):
    if n <= 8 and hidden_size % 1024 == 0:
        return 1024, 0, hidden_size // 1024
    b = min(8192, triton.next_power_of_2(hidden_size))
    return b, 0 if hidden_size % b == 0 else 1, triton.cdiv(hidden_size, b)


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
    b, masked, ntp = _tiling(num_tokens, hidden_size)
    _general[(num_tokens * ntp,)](
        down_output,
        out,
        src2dst,
        topk_ids,
        topk_weights,
        hidden_size,
        topk,
        float(routed_scaling_factor),
        num_local_experts,
        *down_output.stride(),
        *out.stride(),
        *src2dst.stride(),
        *topk_ids.stride(),
        *topk_weights.stride(),
        b,
        masked,
        ntp,
        num_warps=4,
        num_stages=2,
    )
    return out


__all__ = ["post_reorder_cutlass"]
