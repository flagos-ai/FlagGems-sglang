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
def _gather(X, IDS, W, IDX, Y, S: tl.constexpr, B: tl.constexpr):
    T: tl.constexpr = S[0]
    C: tl.constexpr = S[1]
    K: tl.constexpr = S[2]
    N: tl.constexpr = S[3]
    XS0: tl.constexpr = S[4]
    XS1: tl.constexpr = S[5]
    IS0: tl.constexpr = S[6]
    IS1: tl.constexpr = S[7]
    WS0: tl.constexpr = S[8]
    WS1: tl.constexpr = S[9]
    SS0: tl.constexpr = S[10]
    SS1: tl.constexpr = S[11]
    YS0: tl.constexpr = S[12]
    YS1: tl.constexpr = S[13]
    cols = tl.arange(0, B)
    for block in range(tl.program_id(0), T * C, tl.num_programs(0)):
        row = block // C
        col = block % C * B + cols
        acc = tl.zeros((B,), tl.float32)
        for j in tl.static_range(K):
            expert = tl.load(IDS + row * IS0 + j * IS1)
            if expert >= 0:
                weight = tl.load(W + row * WS0 + j * WS1)
                src = tl.load(IDX + row * SS0 + j * SS1).to(tl.int64)
                src = tl.where(src < 0, src + N, src)
                value = tl.load(X + src * XS0 + col * XS1)
                acc = acc + value.to(tl.float32) * weight
        tl.store(Y + row * YS0 + col * YS1, acc)


@triton.jit
def _packed(X, META, Y, S: tl.constexpr, B: tl.constexpr):
    T: tl.constexpr = S[0]
    C: tl.constexpr = S[1]
    K: tl.constexpr = S[2]
    XS0: tl.constexpr = S[4]
    XS1: tl.constexpr = S[5]
    YS0: tl.constexpr = S[12]
    YS1: tl.constexpr = S[13]
    cols = tl.arange(0, B)
    for block in range(tl.program_id(0), T * C, tl.num_programs(0)):
        row = block // C
        col = block % C * B + cols
        acc = tl.zeros((B,), tl.float32)
        for j in tl.static_range(K):
            bits = tl.load(META + row * K + j)
            src = bits.to(tl.uint32).to(tl.int32, bitcast=True).to(tl.int64)
            if src >= 0:
                weight = (
                    (bits >> 32).to(tl.uint32).to(tl.float32, bitcast=True)
                )
                value = tl.load(X + src * XS0 + col * XS1)
                acc = acc + value.to(tl.float32) * weight
        tl.store(Y + row * YS0 + col * YS1, acc)


@triton.jit
def _prepare(
    IDS,
    W,
    IDX,
    META,
    T: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    B: tl.constexpr,
):
    q = tl.program_id(0) * B + tl.arange(0, B)
    safe = tl.minimum(q, T * K - 1)
    row = safe // K
    slot = safe % K
    live = tl.load(IDS + row * IS0 + slot * IS1) >= 0
    weight = tl.load(W + row * WS0 + slot * WS1).to(tl.float32)
    src = tl.load(IDX + row * SS0 + slot * SS1).to(tl.int64)
    src = tl.where(src < 0, src + N, src)
    src = tl.where(live, src, -1).to(tl.int32).to(tl.uint32, bitcast=True)
    bits = src.to(tl.uint64) | (
        weight.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
    )
    tl.store(META + q, bits)


def ep_gather(
    input_tensor, recv_topk_ids, recv_topk_weight, input_index, output_tensor
):
    t, h = output_tensor.shape
    k = recv_topk_ids.shape[1]
    n = input_tensor.shape[0]
    b = (
        4096
        if h % 4096 == 0
        else 2048 if h % 2048 == 0 else 1024 if h % 1024 == 0 else 128
    )
    c = h // b
    total = t * c
    if total:
        info = (
            t,
            c,
            k,
            n,
            *input_tensor.stride(),
            *recv_topk_ids.stride(),
            *recv_topk_weight.stride(),
            *input_index.stride(),
            *output_tensor.stride(),
        )
        if t >= 128 and k and n <= 2147483648:
            blocks = triton.cdiv(t * k, 512)
            meta = torch.empty(
                (blocks * 512,), dtype=torch.uint64, device=input_tensor.device
            )
            _prepare[(blocks,)](
                recv_topk_ids,
                recv_topk_weight,
                input_index,
                meta,
                t,
                k,
                n,
                *recv_topk_ids.stride(),
                *recv_topk_weight.stride(),
                *input_index.stride(),
                512,
            )
            _packed.run(
                input_tensor,
                meta,
                output_tensor,
                info,
                b,
                grid=(min(total, 12),),
                warmup=False,
                isCloseMemoryAsync=False,
            )
        else:
            _gather.run(
                input_tensor,
                recv_topk_ids,
                recv_topk_weight,
                input_index,
                output_tensor,
                info,
                b,
                grid=(min(total, 12),),
                warmup=False,
                isCloseMemoryAsync=False,
            )
    return output_tensor


__all__ = ["ep_gather"]
