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
    R: tl.constexpr = META[9]
    BK: tl.constexpr = K
    BS: tl.constexpr = S
    WORK: tl.constexpr = tl.cdiv(M, R)
    a = tl.arange(0, BK)
    b = tl.arange(0, BS)
    for work in range(tl.program_id(0), WORK, tl.num_programs(0)):
        row = work * R + tl.arange(0, R)
        out = row[:, None] * (K + S)
        mask = (row[:, None] < M) & (a[None, :] < K)
        ids = tl.load(
            IdsPtr + row[:, None] * I0 + a[None, :] * I1, mask, other=0
        )
        weights = tl.load(
            W + row[:, None] * W0 + a[None, :] * W1, mask, other=0
        )
        tl.store(OI + out + a[None, :], ids, mask)
        tl.store(OW + out + a[None, :], weights, mask)
        tail = (row[:, None] < M) & (b[None, :] < S)
        tl.store(OI + out + K + b[None, :], N + b[None, :], tail)
        tl.store(OW + out + K + b[None, :], SCALE, tail)


@triton.jit
def _direct(IdsPtr, W, OI, OW, META: tl.constexpr):
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
    R: tl.constexpr = META[9]
    BK: tl.constexpr = K
    BS: tl.constexpr = S
    a = tl.arange(0, BK)
    b = tl.arange(0, BS)
    work = tl.program_id(0)
    row = work * R + tl.arange(0, R)
    out = row[:, None] * (K + S)
    mask = (row[:, None] < M) & (a[None, :] < K)
    ids = tl.load(IdsPtr + row[:, None] * I0 + a[None, :] * I1, mask, other=0)
    weights = tl.load(W + row[:, None] * W0 + a[None, :] * W1, mask, other=0)
    tl.store(OI + out + a[None, :], ids, mask)
    tl.store(OW + out + a[None, :], weights, mask)
    tail = (row[:, None] < M) & (b[None, :] < S)
    tl.store(OI + out + K + b[None, :], N + b[None, :], tail)
    tl.store(OW + out + K + b[None, :], SCALE, tail)


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
        r = max(1, 256 // max(k, s))
        if m < r:
            r = 1 << (m - 1).bit_length()
        work = (m + r - 1) // r
        if work < 512:
            _direct.run(
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
                    r,
                ),
                grid=(work,),
                warmup=False,
            )
        elif work <= 65535:
            r = r * 4
            work = (m + r - 1) // r
            _direct.run(
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
                    r,
                ),
                grid=(work,),
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
                    r,
                ),
                grid=(min(40, work),),
                warmup=False,
            )
    return (ids, weights)


__all__ = ["fused_append_shared_experts"]
