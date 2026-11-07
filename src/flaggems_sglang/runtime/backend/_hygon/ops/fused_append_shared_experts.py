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

import triton
import triton.language as tl


@triton.jit
def _bits_append(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    SCALE: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    I0: tl.constexpr = META[5]
    I1: tl.constexpr = META[6]
    W0: tl.constexpr = META[7]
    W1: tl.constexpr = META[8]
    B: tl.constexpr = META[9]
    if M * (K + S) % B == 0:
        x = tl.program_id(0) * B + tl.arange(0, B)
        row = x // (K + S)
        col = x % (K + S)
        mask = (row < M) & (col < K)
        ids = tl.load(IdsPtr + row * I0 + col * I1, col < K)
        weights = tl.load(W + row * W0 + col * W1, col < K)
        ids = tl.where(col < K, ids, N + col - K)
        weights = tl.where(col < K, weights, SCALE)
        tl.store(OI + x, ids)
        tl.store(OW + x, weights)
    else:
        x = tl.program_id(0) * B + tl.arange(0, B)
        row = x // (K + S)
        col = x % (K + S)
        mask = (row < M) & (col < K)
        ids = tl.load(IdsPtr + row * I0 + col * I1, mask, other=0)
        weights = tl.load(W + row * W0 + col * W1, mask, other=0)
        ids = tl.where(col < K, ids, N + col - K)
        weights = tl.where(col < K, weights, SCALE)
        tl.store(OI + x, ids, x < M * (K + S))
        tl.store(OW + x, weights, x < M * (K + S))


@triton.jit
def _bits_slim_append(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.float32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.float32))
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    SCALE: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    I0: tl.constexpr = K
    I1: tl.constexpr = 1
    W0: tl.constexpr = K
    W1: tl.constexpr = 1
    B: tl.constexpr = 256
    if M * (K + S) % B == 0:
        x = tl.program_id(0) * B + tl.arange(0, B)
        row = x // (K + S)
        col = x % (K + S)
        mask = (row < M) & (col < K)
        ids = tl.load(IdsPtr + row * I0 + col * I1, col < K)
        weights = tl.load(W + row * W0 + col * W1, col < K)
        ids = tl.where(col < K, ids, N + col - K)
        weights = tl.where(col < K, weights, SCALE)
        tl.store(OI + x, ids)
        tl.store(OW + x, weights)
    else:
        x = tl.program_id(0) * B + tl.arange(0, B)
        row = x // (K + S)
        col = x % (K + S)
        mask = (row < M) & (col < K)
        ids = tl.load(IdsPtr + row * I0 + col * I1, mask, other=0)
        weights = tl.load(W + row * W0 + col * W1, mask, other=0)
        ids = tl.where(col < K, ids, N + col - K)
        weights = tl.where(col < K, weights, SCALE)
        tl.store(OI + x, ids, x < M * (K + S))
        tl.store(OW + x, weights, x < M * (K + S))


def fused_append_shared_experts(
    topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
):
    m, k = topk_ids.shape
    s = num_fused_shared_experts
    if s <= 0:
        return (topk_ids, topk_weights)
    ids = topk_ids.new_empty((m, k + s))
    weights = topk_weights.new_empty((m, k + s))
    if m:
        if topk_ids.is_contiguous() and topk_weights.is_contiguous():
            _bits_slim_append.run(
                topk_ids,
                topk_weights,
                ids,
                weights,
                (m, k, s, scale_factor, N),
                grid=((m * (k + s) + 256 - 1) // 256,),
                warmup=False,
            )
        else:
            _bits_append.run(
                topk_ids,
                topk_weights,
                ids,
                weights,
                (
                    m,
                    k,
                    s,
                    scale_factor,
                    N,
                    *topk_ids.stride(),
                    *topk_weights.stride(),
                    256,
                ),
                grid=((m * (k + s) + 256 - 1) // 256,),
                warmup=False,
            )
    return (ids, weights)


__all__ = ["fused_append_shared_experts"]
