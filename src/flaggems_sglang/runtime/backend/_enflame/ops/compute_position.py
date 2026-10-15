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
    lane = tl.arange(0, R)
    acc = tl.full((R,), 0, tl.int32)
    for base in range(0, first, R):
        index = base + lane
        acc += tl.load(S + index * SS, index < first, other=0)
    start = tl.sum(acc, 0)
    gl = tl.arange(0, CP)
    rows = first + gl
    valid = (gl < C) & (rows < N)
    lens = tl.load(S + rows * SS, valid, other=0)
    if HAS:
        prefs = tl.load(P + rows * PS, valid, other=0)
    else:
        prefs = tl.zeros((CP,), tl.int32)
    tri = tl.where(gl[None, :] < gl[:, None], lens[None, :], 0)
    excl = tl.sum(tri, 1) + start
    tl.store(A + rows, excl, valid)
    offsets = tl.arange(0, B)
    run = start
    for row in range(first, tl.minimum(first + C, N)):
        pick = gl == (row - first)
        length = tl.sum(tl.where(pick, lens, 0), 0)
        prefix = tl.sum(tl.where(pick, prefs, 0), 0)
        for base in range(0, length, B):
            index = base + offsets
            tl.store(O + run + index, prefix + index, index < length)
        run += length


def compute_position(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    n = extend_seq_lens.shape[0]
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    start = torch.empty(n, dtype=torch.int32, device=extend_seq_lens.device)
    if n:
        count = triton.cdiv(n, 48 if n >= 512 else 24)
        _position[(triton.cdiv(n, count),)](
            extend_prefix_lens,
            extend_seq_lens,
            triton.reinterpret(positions, tl.int32),
            start,
            n,
            extend_prefix_lens.stride(0),
            extend_seq_lens.stride(0),
            extend_prefix_lens.shape[0] == n,
            count,
            triton.next_power_of_2(count),
            max(32, min(triton.next_power_of_2(n), 1024)),
            1024,
            num_warps=1,
        )
    return positions, start


__all__ = ["compute_position"]
