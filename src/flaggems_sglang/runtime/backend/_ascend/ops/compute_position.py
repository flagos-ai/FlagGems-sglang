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
def _grouped(
    P,
    S,
    O,
    A,
    N,
    PS: tl.constexpr,
    SS: tl.constexpr,
    HAS: tl.constexpr,
    G: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    first = tl.program_id(0) * G
    lane = tl.arange(0, R)
    acc = tl.full((R,), 0, tl.int32)
    for base in range(0, first, R):
        idx = base + lane
        acc += tl.load(S + idx * SS, idx < first, other=0)
    running = tl.sum(acc, 0)
    offsets = tl.arange(0, B)
    for step in tl.static_range(G):
        row = first + step
        if row < N:
            length = tl.load(S + row * SS)
            tl.store(A + row, running)
            prefix = tl.full((), 0, tl.int32)
            if HAS:
                prefix = tl.load(P + row * PS)
            for base in range(0, length, B):
                idx = base + offsets
                tl.store(
                    O + running + idx,
                    (prefix + idx).to(tl.int64),
                    idx < length,
                )
            running += length


def compute_position(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    n = extend_seq_lens.shape[0]
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    start = torch.empty(n, dtype=torch.int32, device=extend_seq_lens.device)
    if n >= 512:
        _grouped[(triton.cdiv(n, 2),)](
            extend_prefix_lens,
            extend_seq_lens,
            positions,
            start,
            n,
            extend_prefix_lens.stride(0),
            extend_seq_lens.stride(0),
            extend_prefix_lens.shape[0] == n,
            2,
            max(32, min(triton.next_power_of_2(n), 1024)),
            256,
            num_warps=1,
        )
    elif n:
        _grouped[(n,)](
            extend_prefix_lens,
            extend_seq_lens,
            positions,
            start,
            n,
            extend_prefix_lens.stride(0),
            extend_seq_lens.stride(0),
            extend_prefix_lens.shape[0] == n,
            1,
            max(32, min(triton.next_power_of_2(n), 1024)),
            256,
            num_warps=1,
        )
    return positions, start


__all__ = ["compute_position"]
