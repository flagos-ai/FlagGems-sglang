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
def _gather(
    X,
    IDS,
    W,
    IDX,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    acc = tl.full((BR, BC), 0, tl.float32)
    for j in tl.static_range(K):
        expert = tl.load(IDS + rows * IS0 + j * IS1, rows < T, other=-1)
        valid = (rows < T) & (expert >= 0)
        src = tl.load(IDX + rows * SS0 + j * SS1, valid, other=0).to(tl.int64)
        src = tl.where(src < 0, src + N, src)
        weight = tl.load(W + rows * WS0 + j * WS1, valid, other=0).to(
            tl.float32
        )
        value = tl.load(
            X + src[:, None] * XS0 + cols[None, :] * XS1,
            valid[:, None] & (cols[None, :] < H),
            other=0,
        ).to(tl.float32)
        acc = acc + value * weight[:, None]
    tl.store(
        Y + rows[:, None] * YS0 + cols[None, :] * YS1,
        acc,
        (rows[:, None] < T) & (cols[None, :] < H),
    )


def ep_gather(
    input_tensor, recv_topk_ids, recv_topk_weight, input_index, output_tensor
):
    t, h = output_tensor.shape
    if t and h:
        _gather[(triton.cdiv(t, 8), triton.cdiv(h, 128))](
            input_tensor,
            recv_topk_ids,
            recv_topk_weight,
            input_index,
            output_tensor,
            t,
            h,
            recv_topk_ids.shape[1],
            input_tensor.shape[0],
            *input_tensor.stride(),
            *recv_topk_ids.stride(),
            *recv_topk_weight.stride(),
            *input_index.stride(),
            *output_tensor.stride(),
            8,
            128,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output_tensor


__all__ = ["ep_gather"]
