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
def _position(
    P,
    S,
    O,
    A,
    N,
    PS: tl.constexpr,
    SS: tl.constexpr,
    HAS: tl.constexpr,
    C: tl.constexpr,
    CP: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    first = tl.program_id(0) * C
    start = tl.full((), 0, tl.int32)
    if first > 0:
        lane = tl.arange(0, R)
        acc = tl.full((R,), 0, tl.int32)
        for base in range(0, first, R):
            index = base + lane
            acc += tl.load(S + index * SS, index < first, other=0)
        start = tl.sum(acc, 0)
    if C == 1:
        length = tl.load(S + first * SS)
        if HAS:
            prefix = tl.load(P + first * PS)
        else:
            prefix = tl.full((), 0, tl.int32)
        tl.store(A + first, start)
        for base in range(0, length, B):
            index = base + tl.arange(0, B)
            tl.store(O + start + index, prefix.to(tl.int64) + index, index < length)
    else:
        gl = tl.arange(0, CP)
        rows = first + gl
        valid = (gl < C) & (rows < N)
        lens = tl.load(S + rows * SS, valid, other=0)
        if HAS:
            prefs = tl.load(P + rows * PS, valid, other=0)
        else:
            prefs = tl.zeros((CP,), tl.int32)
        excl = tl.cumsum(lens, 0) - lens + start
        tl.store(A + rows, excl, valid)
        maxlen = tl.max(lens, 0)
        for base in range(0, maxlen, B):
            tile = base + tl.arange(0, B)
            mask = valid[:, None] & (tile[None, :] < lens[:, None])
            tl.store(
                O + excl[:, None] + tile[None, :],
                prefs[:, None].to(tl.int64) + tile[None, :],
                mask,
            )


def compute_position(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    n = extend_seq_lens.shape[0]
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    start = torch.empty(n, dtype=torch.int32, device=extend_seq_lens.device)
    if n:
        count = min(triton.cdiv(n, 48), 32)
        mean_len = (extend_seq_lens_sum + n - 1) // n
        if count == 1:
            block = triton.next_power_of_2(2 * mean_len)
            b = min(1024, max(64, block))
        else:
            block = triton.next_power_of_2(max(mean_len, 1))
            b = max(512, min(1024, block))
        _position[(triton.cdiv(n, count),)](
            extend_prefix_lens,
            extend_seq_lens,
            positions,
            start,
            n,
            extend_prefix_lens.stride(0),
            extend_seq_lens.stride(0),
            extend_prefix_lens.shape[0] == n,
            count,
            triton.next_power_of_2(count),
            min(triton.next_power_of_2(n), 1024),
            b,
            num_warps=1,
        )
    return positions, start


__all__ = ["compute_position"]
