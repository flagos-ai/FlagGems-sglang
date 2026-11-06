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

_tensor_new_empty = torch.Tensor.new_empty
_tensor_is_contiguous = torch.Tensor.is_contiguous
_tensor_stride = torch.Tensor.stride


@triton.jit
def _bits_append(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.uint32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.uint32))
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
    weights = tl.where(
        col < K,
        weights,
        tl.full((), SCALE, tl.float32).to(tl.uint32, bitcast=True),
    )
    tl.store(OI + x, ids, x < M * (K + S))
    tl.store(OW + x, weights, x < M * (K + S))


@triton.jit
def _bits_slim_append(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.uint32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.uint32))
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
    ids = tl.load(IdsPtr + row * I0 + col * I1, mask, other=N + col - K)
    weights = tl.load(
        W + row * W0 + col * W1,
        mask,
        other=tl.full((), SCALE, tl.float32).to(tl.uint32, bitcast=True),
    )
    tl.store(OI + x, ids, x < M * (K + S))
    tl.store(OW + x, weights, x < M * (K + S))


@triton.jit
def _bits_large_append(IdsPtr, W, OI, OW, META: tl.constexpr):
    IdsPtr = IdsPtr.to(tl.pointer_type(tl.int32))
    W = W.to(tl.pointer_type(tl.uint32))
    OI = OI.to(tl.pointer_type(tl.int32))
    OW = OW.to(tl.pointer_type(tl.uint32))
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    SCALE: tl.constexpr = META[3]
    N: tl.constexpr = META[4]
    I0: tl.constexpr = K
    I1: tl.constexpr = 1
    W0: tl.constexpr = K
    W1: tl.constexpr = 1
    B: tl.constexpr = 512
    x = tl.program_id(0) * B + tl.arange(0, B)
    row = x // (K + S)
    col = x % (K + S)
    mask = (row < M) & (col < K)
    ids = tl.load(IdsPtr + row * I0 + col * I1, mask, other=N + col - K)
    weights = tl.load(
        W + row * W0 + col * W1,
        mask,
        other=tl.full((), SCALE, tl.float32).to(tl.uint32, bitcast=True),
    )
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
    ids = _tensor_new_empty(topk_ids, output_shape)
    weights = _tensor_new_empty(topk_weights, output_shape)
    if m:
        if _tensor_is_contiguous(topk_ids) and _tensor_is_contiguous(
            topk_weights
        ):
            if m >= 12288 and k == 8 and (s == 1):
                _bits_large_append.run(
                    topk_ids,
                    topk_weights,
                    ids,
                    weights,
                    (m, k, s, scale_factor, N),
                    grid=((m * p + 512 - 1) // 512,),
                    warmup=False,
                )
            else:
                _bits_slim_append.run(
                    topk_ids,
                    topk_weights,
                    ids,
                    weights,
                    (m, k, s, scale_factor, N),
                    grid=((m * p + 256 - 1) // 256,),
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
                    *_tensor_stride(topk_ids),
                    *_tensor_stride(topk_weights),
                    256,
                ),
                grid=((m * p + 256 - 1) // 256,),
                warmup=False,
            )
    return (ids, weights)


__all__ = ["fused_append_shared_experts"]
