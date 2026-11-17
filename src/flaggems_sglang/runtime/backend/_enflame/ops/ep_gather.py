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
def _gather_pair(
    X, IDS, W, IDX, Y, S: tl.constexpr, B: tl.constexpr, TAIL: tl.constexpr
):
    K: tl.constexpr = S[0]
    N: tl.constexpr = S[1]
    XS0: tl.constexpr = S[2]
    XS1: tl.constexpr = S[3]
    IS0: tl.constexpr = S[4]
    IS1: tl.constexpr = S[5]
    WS0: tl.constexpr = S[6]
    WS1: tl.constexpr = S[7]
    SS0: tl.constexpr = S[8]
    SS1: tl.constexpr = S[9]
    YS0: tl.constexpr = S[10]
    YS1: tl.constexpr = S[11]
    H: tl.constexpr = S[12]
    T: tl.constexpr = S[13]
    # GCU grid limits: grid.x <= 65535, grid.y <= 255. Rows go to grid.x
    # (matches the generic CUDA grid.y domain of 65535), column blocks to
    # grid.y (capped on host by scaling B).
    #
    # Two rows per program with the A/B scalar + gather loads interleaved
    # inside one K loop (moe_fused_mul_sum PAIR recipe): every j segment
    # issues 6 metadata scalars and 2 row gathers back-to-back, doubling the
    # in-flight window that hides the gather latency (and halving the
    # program count / dispatch waves).  The odd tail clamps rowB to the last
    # row; the duplicate store is an idempotent overwrite, and the per-row
    # arithmetic (j-ascending fp32 accumulation, where-select on live) is
    # unchanged, so outputs are bitwise-identical to the single-row form.
    rowA = tl.program_id(0) * 2
    rowB = tl.minimum(rowA + 1, T - 1)
    col = tl.program_id(1) * B + tl.arange(0, B)
    if TAIL:
        cm = col < H
    accA = tl.zeros((B,), tl.float32)
    accB = tl.zeros((B,), tl.float32)
    for j in tl.static_range(K):
        expertA = tl.load(IDS + rowA * IS0 + j * IS1)
        weightA = tl.load(W + rowA * WS0 + j * WS1)
        srcA = tl.load(IDX + rowA * SS0 + j * SS1).to(tl.int64)
        expertB = tl.load(IDS + rowB * IS0 + j * IS1)
        weightB = tl.load(W + rowB * WS0 + j * WS1)
        srcB = tl.load(IDX + rowB * SS0 + j * SS1).to(tl.int64)
        liveA = expertA >= 0
        srcA = tl.where(liveA, tl.where(srcA < 0, srcA + N, srcA), 0)
        liveB = expertB >= 0
        srcB = tl.where(liveB, tl.where(srcB < 0, srcB + N, srcB), 0)
        if TAIL:
            valueA = tl.load(X + srcA * XS0 + col * XS1, mask=cm, other=0.0)
            valueB = tl.load(X + srcB * XS0 + col * XS1, mask=cm, other=0.0)
        else:
            valueA = tl.load(X + srcA * XS0 + col * XS1)
            valueB = tl.load(X + srcB * XS0 + col * XS1)
        accA = tl.where(liveA, accA + valueA.to(tl.float32) * weightA, accA)
        accB = tl.where(liveB, accB + valueB.to(tl.float32) * weightB, accB)
    if TAIL:
        tl.store(Y + rowA * YS0 + col * YS1, accA, mask=cm)
        tl.store(Y + rowB * YS0 + col * YS1, accB, mask=cm)
    else:
        tl.store(Y + rowA * YS0 + col * YS1, accA)
        tl.store(Y + rowB * YS0 + col * YS1, accB)


def _pick_block(t, h):
    # Round-2 (load-op dispatch throughput bound): wider column blocks
    # amortize the per-load fixed cost, t <= 8 is launch-latency bound.
    # Round-3: rows are paired two-per-program in every arm (measured faster
    # from t=1 to t=4096; the pair form doubles the in-flight load window and
    # halves the dispatch waves).  num_warps on the paired wide arm flips at
    # one SP wave: nw=2 wins while the paired grid fits in ~32 programs,
    # nw=1 wins above (t=512/4096 measured -8%).
    if t <= 8:
        b, nw = 2048, 2
    elif h >= 2048:
        b, nw = 4096, 2
    else:
        b, nw = 1024, 1
    nc = (h + b - 1) // b
    while nc > 255:
        b *= 2
        nc = (h + b - 1) // b
    if h >= 2048 and t > 8 and ((t + 1) // 2) * nc > 32:
        nw = 1
    return b, nw, nc


def ep_gather(
    input_tensor, recv_topk_ids, recv_topk_weight, input_index, output_tensor
):
    t, h = output_tensor.shape
    if t and h:
        b, nw, nc = _pick_block(t, h)
        info = (
            recv_topk_ids.shape[1],
            input_tensor.shape[0],
            *input_tensor.stride(),
            *recv_topk_ids.stride(),
            *recv_topk_weight.stride(),
            *input_index.stride(),
            *output_tensor.stride(),
            h,
            t,
        )
        _gather_pair.run(
            input_tensor,
            recv_topk_ids,
            recv_topk_weight,
            input_index,
            output_tensor,
            info,
            b,
            h % b != 0,
            grid=((t + 1) // 2, nc),
            warmup=False,
            num_warps=nw,
        )
    return output_tensor


__all__ = ["ep_gather"]
