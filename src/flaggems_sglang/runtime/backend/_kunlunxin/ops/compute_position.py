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
    N: tl.constexpr,
    PS: tl.constexpr,
    SS: tl.constexpr,
    HAS: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, R)
    acc = tl.full((R,), 0, tl.int32)
    for base in range(0, row, R):
        idx = base + lane
        value = tl.load(S + idx * SS, idx < row, other=0)
        acc += value
    start = tl.sum(acc, 0)
    tl.store(A + row, start)
    length = tl.load(S + row * SS)
    prefix = tl.full((), 0, tl.int32)
    if HAS:
        prefix = tl.load(P + row * PS)
    offsets = tl.arange(0, B)
    for base in range(0, length, B):
        idx = base + offsets
        tl.store(O + start + idx, prefix.to(tl.int64) + idx, idx < length)


def _small(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    n = extend_seq_lens.shape[0]
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    start = torch.empty(n, dtype=torch.int32, device=extend_seq_lens.device)
    if n:
        _position[(n,)](
            extend_prefix_lens,
            extend_seq_lens,
            positions,
            start,
            n,
            extend_prefix_lens.stride(0),
            extend_seq_lens.stride(0),
            extend_prefix_lens.shape[0] == n,
            min(triton.next_power_of_2(n), 1024),
            256,
        )
    return positions, start


@triton.jit
def _scan(S, A, N: tl.constexpr, SS: tl.constexpr, R: tl.constexpr):
    block = tl.program_id(0)
    lane = tl.arange(0, R)
    acc = tl.full((R,), 0, tl.int64)
    for base in range(0, block * R, R):
        acc += tl.load(S + (base + lane) * SS).to(tl.int64)
    previous = tl.sum(acc, 0)
    index = block * R + lane
    values = tl.load(S + index * SS, index < N, other=0).to(tl.int64)
    starts = previous + tl.cumsum(values, 0) - values
    tl.store(A + index, starts, index < N)


@triton.jit
def _write(
    P,
    S,
    O,
    A,
    PS: tl.constexpr,
    SS: tl.constexpr,
    HAS: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(A + row).to(tl.int64)
    length = tl.load(S + row * SS)
    prefix = tl.full((), 0, tl.int64)
    if HAS:
        prefix = tl.load(P + row * PS).to(tl.int64)
    lane = tl.arange(0, B)
    for base in range(0, length, B):
        index = base + lane
        tl.store(O + start + index, prefix + index, index < length)


def _large(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    n = extend_seq_lens.shape[0]
    positions = torch.empty(
        extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
    )
    start = torch.empty(n, dtype=torch.int32, device=extend_seq_lens.device)
    if n:
        r = min(triton.next_power_of_2(n), 1024)
        _scan[(triton.cdiv(n, r),)](
            extend_seq_lens,
            start,
            n,
            extend_seq_lens.stride(0),
            r,
            num_warps=4,
        )
        _write[(n,)](
            extend_prefix_lens,
            extend_seq_lens,
            positions,
            start,
            extend_prefix_lens.stride(0),
            extend_seq_lens.stride(0),
            extend_prefix_lens.shape[0] == n,
            2048,
            num_warps=4,
        )
    return positions, start


def compute_position(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum):
    if extend_seq_lens.shape[0] >= 256:
        return _large(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum)
    return _small(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum)


__all__ = ["compute_position"]
