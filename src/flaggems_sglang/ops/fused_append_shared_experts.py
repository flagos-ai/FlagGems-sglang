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
def _append(IdsPtr, W, OI, OW, META: tl.constexpr):
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
def _slim_append(IdsPtr, W, OI, OW, META: tl.constexpr):
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
def _aligned_append(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = tl.multiple_of(IdsPtr.to(tl.pointer_type(tl.int32)), 16)
    W = tl.multiple_of(W.to(tl.pointer_type(tl.float32)), 16)
    OI = tl.multiple_of(OI.to(tl.pointer_type(tl.int32)), 16)
    OW = tl.multiple_of(OW.to(tl.pointer_type(tl.float32)), 16)
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
def _slim_append_b128w4(IdsPtr, W, OI, OW, META: tl.constexpr):
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
    B: tl.constexpr = 128
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
    p = k + s
    output_shape = (m, p)
    ids = topk_ids.new_empty(output_shape)
    weights = topk_weights.new_empty(output_shape)
    if m:
        pointers = (
            topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            ids.data_ptr(),
            weights.data_ptr(),
        )
        if topk_ids.is_contiguous() and topk_weights.is_contiguous():
            if torch.version.hip is not None:
                _slim_append_b128w4.run(
                    topk_ids,
                    topk_weights,
                    ids,
                    weights,
                    (m, k, s, scale_factor, N),
                    grid=((m * p + 127) // 128,),
                    warmup=False,
                )
            elif (
                k <= 32
                and s <= 32
                and (
                    (pointers[0] | pointers[1] | pointers[2] | pointers[3])
                    & 15
                    == 0
                )
            ):
                _aligned_append.run(
                    topk_ids,
                    topk_weights,
                    ids,
                    weights,
                    (m, k, s, scale_factor, N),
                    grid=((m * p + 256 - 1) // 256,),
                    warmup=False,
                )
            else:
                _slim_append.run(
                    topk_ids,
                    topk_weights,
                    ids,
                    weights,
                    (m, k, s, scale_factor, N),
                    grid=((m * p + 256 - 1) // 256,),
                    warmup=False,
                )
        else:
            _append.run(
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
                grid=((m * p + 256 - 1) // 256,),
                warmup=False,
            )
    return (ids, weights)


__all__ = ["fused_append_shared_experts"]
