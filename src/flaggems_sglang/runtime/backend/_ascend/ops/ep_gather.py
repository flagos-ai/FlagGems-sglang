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
            weight = tl.load(W + row * WS0 + j * WS1)
            src = tl.load(IDX + row * SS0 + j * SS1).to(tl.int64)
            src = tl.where(src < 0, src + N, src)
            value = tl.load(X + src * XS0 + col * XS1, expert >= 0, other=0.0)
            acc = acc + tl.where(
                expert >= 0, value.to(tl.float32) * weight, 0.0
            )
        tl.store(Y + row * YS0 + col * YS1, acc)


def ep_gather(
    input_tensor, recv_topk_ids, recv_topk_weight, input_index, output_tensor
):
    t, h = output_tensor.shape
    b = 1024 if h % 1024 == 0 else 128
    c = h // b
    total = t * c
    if total:
        _gather.run(
            input_tensor,
            recv_topk_ids,
            recv_topk_weight,
            input_index,
            output_tensor,
            (
                t,
                c,
                recv_topk_ids.shape[1],
                input_tensor.shape[0],
                *input_tensor.stride(),
                *recv_topk_ids.stride(),
                *recv_topk_weight.stride(),
                *input_index.stride(),
                *output_tensor.stride(),
            ),
            b,
            grid=(total if total < 40 else 40,),
            warmup=False,
        )
    return output_tensor


__all__ = ["ep_gather"]
