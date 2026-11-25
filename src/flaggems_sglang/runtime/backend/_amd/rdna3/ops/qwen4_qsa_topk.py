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

"""qwen4_qsa_topk for RDNA3 (gfx1100).

One kernel instead of a chain of PyTorch kernels: each program sorts one
row. Every column becomes a 64-bit key, an order-preserving image of the
negated logit above the column index, so an ascending sort yields
descending logits with ties in column order, as in the reference. Columns
outside [start, end) sort after every selectable column and are written as
-1.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _key(L, row, col, start, end, L0, L1, COLS: tl.constexpr):
    valid = (col >= start) & (col < end) & (col < COLS)
    x = tl.load(L + row * L0 + col * L1, valid, 0.0).to(tl.float32)
    bits = tl.where(x == 0.0, 0.0, -x).to(tl.uint32, bitcast=True)
    key = tl.where((bits >> 31) != 0, ~bits, bits | 0x80000000)
    return tl.where(valid, key, 0xFFFFFFFF)


@triton.jit
def _sort_kernel(
    L,
    S,
    E,
    O,
    L0,
    L1,
    S0,
    E0,
    O0,
    COLS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(S + row * S0).to(tl.int32)
    end = tl.load(E + row * E0).to(tl.int32)
    col = tl.arange(0, BLOCK)
    key = _key(L, row, col, start, end, L0, L1, COLS)
    key = tl.sort((key.to(tl.uint64) << 32) | col.to(tl.uint64))
    picked = (key & 0xFFFFFFFF).to(tl.int32) - start
    picked = tl.where((key >> 32) != 0xFFFFFFFF, picked, -1)
    tl.store(O + row * O0 + col, picked, col < TOPK)


def qwen4_qsa_topk(logits, row_starts, row_ends, topk):
    rows, cols = logits.shape
    out = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    if not rows or not topk:
        return out
    starts = row_starts.to(device=logits.device).reshape(-1)
    ends = row_ends.to(device=logits.device).reshape(-1)
    _sort_kernel[(rows,)](
        logits,
        starts,
        ends,
        out,
        logits.stride(0),
        logits.stride(1),
        starts.stride(0),
        ends.stride(0),
        out.stride(0),
        cols,
        topk,
        triton.next_power_of_2(max(cols, topk)),
        num_warps=8 if rows <= 1024 else 1,
    )
    return out


__all__ = ["qwen4_qsa_topk"]
