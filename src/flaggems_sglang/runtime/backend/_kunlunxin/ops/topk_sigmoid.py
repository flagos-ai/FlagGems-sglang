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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import triton
import triton.language as tl


@triton.jit
def _topk(
    X,
    W,
    I,
    M: tl.constexpr,  # num rows in gating_output
    N: tl.constexpr,  # num experts (gating_output row width)
    K: tl.constexpr,  # top-k
    RENORM: tl.constexpr,
    SCALE: tl.constexpr,  # routed_scaling_factor
    B: tl.constexpr,  # N padded up to a power of two
    A: tl.constexpr,  # K padded up to a power of two (min 2)
    R: tl.constexpr,  # rows handled per program
):
    neg_large = -3.4028234663852886e38

    rows = tl.program_id(0) * R + tl.arange(0, R)
    offs_n = tl.arange(0, B)
    offs_k = tl.arange(0, A)

    # (R, B) tile of this program's rows; out-of-range rows/columns read as
    # -inf so they can never win the max.
    live = (rows[:, None] < M) & (offs_n[None, :] < N)
    vals = tl.load(
        X + rows[:, None] * N + offs_n[None, :],
        mask=live,
        other=neg_large,
    )
    vals = tl.where(live, vals, neg_large)

    top_v = tl.full((R, A), neg_large, tl.float32)
    top_i = tl.full((R, A), N, tl.int32)
    total = tl.full((R,), 1.0e-20, tl.float32)

    # Iterative selection peel: find the row max, take its lowest column as
    # the winning index, then mask that value out and repeat.
    best = tl.max(vals, axis=1)
    for slot in tl.static_range(0, K):
        # lowest column index holding the current max (ties -> lowest)
        bidx = tl.min(
            tl.where(vals == best[:, None], offs_n[None, :], N), axis=1
        )
        value = tl.sigmoid(best)
        total += value

        # write the winning slot into the (R, A) register arrays
        top_v = tl.where(offs_k[None, :] == slot, value[:, None], top_v)
        top_i = tl.where(
            offs_k[None, :] == slot, bidx[:, None].to(tl.int32), top_i
        )

        # knock out the winner and re-select
        best = tl.max(tl.where(vals < best[:, None], vals, neg_large), axis=1)

    if RENORM:
        top_v = top_v * (SCALE / total)[:, None]

    keep = (rows[:, None] < M) & (offs_k[None, :] < K)
    tl.store(W + rows[:, None] * K + offs_k[None, :], top_v, mask=keep)
    tl.store(I + rows[:, None] * K + offs_k[None, :], top_i, mask=keep)


def _rows_per_block(num_tokens, width):
    """R0 doubles with ~cube-root(num_tokens) (measured sweep, B=256); scale
    by 256/width for narrower rows so the (R, B) tile stays in budget."""
    if num_tokens <= 2:
        r0 = 1
    elif num_tokens <= 16:
        r0 = 4
    elif num_tokens <= 128:
        r0 = 8
    elif num_tokens <= 1024:
        r0 = 16
    else:
        r0 = 32
    return max(1, min(64, r0 * 256 // width))


def topk_sigmoid(
    topk_weights, topk_ids, gating_output, renormalize, routed_scaling_factor
):
    m, n = gating_output.shape
    k = topk_weights.shape[1]

    if m > 0 and k > 0:
        width = 1 << (n - 1).bit_length()  # power-of-two >= n
        rows_per_block = _rows_per_block(m, width)

        _topk[((m + rows_per_block - 1) // rows_per_block,)](
            gating_output,
            topk_weights,
            topk_ids,
            m,
            n,
            k,
            renormalize,
            routed_scaling_factor,
            width,
            max(2, 1 << (k - 1).bit_length()),  # A: power-of-two >= k, min 2
            rows_per_block,
            num_warps=1,
            num_stages=1,
        )
    return topk_weights, topk_ids


__all__ = ["topk_sigmoid"]
