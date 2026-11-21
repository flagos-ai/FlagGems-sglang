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
def _offset(i, S: tl.constexpr, N: tl.constexpr):
    if (N - 1) * S <= 268435455 and (N - 1) * S >= -268435456:
        return i * S
    else:
        return i.to(tl.int64) * S


@triton.jit
def _mix(h, k):
    k = (k.to(tl.uint32) * 3432918353).to(tl.uint32)
    k = k << 15 | k >> 17
    k = (k * 461845907).to(tl.uint32)
    h = h.to(tl.uint32) ^ k
    h = h << 13 | h >> 19
    return (h * 5 + 3864292196).to(tl.uint32)


@triton.jit
def _kernel(S, P, C, OUT, META: tl.constexpr):
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
        S + _offset(rows[:, None], SS, N),
        (B == 256) & (N % R == 0) | (rows[:, None] < N),
        0,
    ).to(tl.uint64)
    pos = tl.load(
        P + _offset(rows[:, None], PS, N),
        (B == 256) & (N % R == 0) | (rows[:, None] < N),
        0,
    ).to(tl.uint32)
    key = tl.load(
        C + _offset(cols[None, :], CS, M),
        (B == 256) & (M % B == 0) | (cols[None, :] < M),
        0,
    ).to(tl.uint32)
    h = _mix(tl.full((R, 1), 0, tl.uint32), seed.to(tl.uint32))
    if S.dtype.element_ty == tl.int32:
        high_key = tl.where(seed.to(tl.int32) < 0, 3152982032, 0).to(tl.uint32)
        h = h ^ high_key
        h = h << 13 | h >> 19
        h = (h * 5 + 3864292196).to(tl.uint32)
    else:
        h = _mix(h, (seed >> 32).to(tl.uint32))
    h = _mix(h, pos)
    key = (key * 3432918353).to(tl.uint32)
    key = key << 15 | key >> 17
    key = (key * 461845907).to(tl.uint32)
    h = (h << 13 | h >> 19) ^ (key << 13 | key >> 19)
    h = (h * 5 + 3864292196).to(tl.uint32) ^ 16
    h = h ^ h >> 16
    h = (h * 2246822507).to(tl.uint32)
    h = h ^ h >> 13
    h = (h * 3266489909).to(tl.uint32)
    h = h ^ h >> 16
    tl.store(
        OUT + rows[:, None].to(tl.int64) * M + cols[None, :],
        h,
        ((B == 256) & (N % R == 0) | (rows[:, None] < N))
        & ((B == 256) & (M % B == 0) | (cols[None, :] < M)),
    )


def murmur_hash32(seed, positions, col_indices):
    n = seed.numel()
    m = col_indices.numel()
    out = seed.new_empty((n, m), dtype=torch.uint32)
    if n and m:
        rows = 1 << (n - 1).bit_length() if n < 8 else 8
        block = 64 if (n + 7) // 8 * ((m + 255) // 256) <= 128 else 256
        if block == 64:
            work_tiles = n * ((m + 63) // 64)
            if work_tiles < 512:
                rows = 1
            elif work_tiles < 1024:
                if rows > 2:
                    rows = 2
            elif work_tiles < 2048:
                if rows > 4:
                    rows = 4
        _kernel.run(
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
                rows,
                block,
            ),
            num_warps=1 if block == 64 else 4,
            grid=((n + rows - 1) // rows, (m + block - 1) // block),
            warmup=False,
        )
    return out


__all__ = ["murmur_hash32"]
