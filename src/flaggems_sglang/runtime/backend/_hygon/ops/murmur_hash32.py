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
def _input_offset(index, STRIDE: tl.constexpr, SIZE: tl.constexpr):
    if (SIZE - 1) * STRIDE >= -268435456 and (SIZE - 1) * STRIDE <= 268435455:
        return index * STRIDE
    else:
        return index.to(tl.int64) * STRIDE


@triton.jit
def _mix(h, k):
    k = (k.to(tl.uint32) * 3432918353).to(tl.uint32)
    k = k << 15 | k >> 17
    k = (k * 461845907).to(tl.uint32)
    h = h.to(tl.uint32) ^ k
    h = h << 13 | h >> 19
    return (h * 5 + 3864292196).to(tl.uint32)


@triton.jit
def _small(S, P, C, OUT, META: tl.constexpr):
    N: tl.constexpr = META[4]
    M: tl.constexpr = META[0]
    SS: tl.constexpr = META[1]
    PS: tl.constexpr = META[2]
    CS: tl.constexpr = META[3]
    B: tl.constexpr = 512
    tiles: tl.constexpr = triton.cdiv(M, B)
    row = tl.program_id(0) // tiles
    cols = tl.program_id(0) % tiles * B + tl.arange(0, B)
    seed = tl.load(S + _input_offset(row, SS, N)).to(tl.uint64)
    pos = tl.load(P + _input_offset(row, PS, N)).to(tl.uint32)
    key = tl.load(C + _input_offset(cols, CS, M), cols < M, 0).to(tl.uint32)
    h = _mix(tl.full((), 0, tl.uint32), seed.to(tl.uint32))
    if S.dtype.element_ty == tl.int32:
        hi_key = tl.where(seed.to(tl.int32) < 0, 3152982032, 0).to(tl.uint32)
        h = h ^ hi_key
        h = h << 13 | h >> 19
        h = (h * 5 + 3864292196).to(tl.uint32)
    else:
        h = _mix(h, (seed >> 32).to(tl.uint32))
    h = _mix(h, pos)
    h = _mix(h, key) ^ 16
    h = h ^ h >> 16
    h = (h * 2246822507).to(tl.uint32)
    h = h ^ h >> 13
    h = (h * 3266489909).to(tl.uint32)
    h = h ^ h >> 16
    tl.store(OUT + row.to(tl.int64) * M + cols, h, cols < M)


@triton.jit
def _large(S, P, C, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    M: tl.constexpr = META[1]
    SS: tl.constexpr = META[2]
    PS: tl.constexpr = META[3]
    CS: tl.constexpr = META[4]
    R: tl.constexpr = META[5]
    B: tl.constexpr = META[6]
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    seed = tl.load(
        S + _input_offset(rows[:, None], SS, N),
        (N % R == 0) | (rows[:, None] < N),
        0,
    ).to(tl.uint64)
    pos = tl.load(
        P + _input_offset(rows[:, None], PS, N),
        (N % R == 0) | (rows[:, None] < N),
        0,
    ).to(tl.uint32)
    key = tl.load(
        C + _input_offset(cols[None, :], CS, M),
        (M % B == 0) | (cols[None, :] < M),
        0,
    ).to(tl.uint32)
    h = _mix(tl.full((R, 1), 0, tl.uint32), seed.to(tl.uint32))
    h = _mix(h, (seed >> 32).to(tl.uint32))
    h = _mix(h, pos)
    h = _mix(h, key) ^ 16
    h = h ^ h >> 16
    h = (h * 2246822507).to(tl.uint32)
    h = h ^ h >> 13
    h = (h * 3266489909).to(tl.uint32)
    h = h ^ h >> 16
    tl.store(
        OUT + rows[:, None].to(tl.int64) * M + cols[None, :],
        h,
        ((N % R == 0) | (rows[:, None] < N))
        & ((M % B == 0) | (cols[None, :] < M)),
    )


def murmur_hash32(seed, positions, col_indices):
    n = seed.numel()
    m = col_indices.numel()
    out = seed.new_empty((n, m), dtype=torch.uint32)
    if n and m:
        if n < 256:
            _small.run(
                seed,
                positions,
                col_indices,
                out,
                (
                    m,
                    seed.stride(0),
                    positions.stride(0),
                    col_indices.stride(0),
                    n,
                ),
                grid=(n * ((m + 511) // 512),),
                num_warps=4,
                warmup=False,
            )
        else:
            _large.run(
                seed,
                positions,
                col_indices,
                out,
                (
                    n,
                    m,
                    seed.stride(0),
                    positions.stride(0),
                    col_indices.stride(0),
                    2,
                    1024,
                ),
                grid=((n + 1) // 2, (m + 1023) // 1024),
                num_warps=4,
                warmup=False,
            )
    return out


__all__ = ["murmur_hash32"]
