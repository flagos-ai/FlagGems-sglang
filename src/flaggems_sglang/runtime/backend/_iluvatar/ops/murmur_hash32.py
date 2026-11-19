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
def _mix(h, k):
    k = (k.to(tl.uint32) * 3432918353).to(tl.uint32)
    k = k << 15 | k >> 17
    k = (k * 461845907).to(tl.uint32)
    h = h.to(tl.uint32) ^ k
    h = h << 13 | h >> 19
    return (h * 5 + 3864292196).to(tl.uint32)


@triton.jit(do_not_specialize=["S", "P", "C", "OUT"])
def _kernel(S, P, C, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    M: tl.constexpr = META[1]
    SS: tl.constexpr = META[2]
    PS: tl.constexpr = META[3]
    CS: tl.constexpr = META[4]
    R: tl.constexpr = 1024 if N > 1024 else N
    B: tl.constexpr = 512
    tiles: tl.constexpr = triton.cdiv(M, B)
    first_row = tl.program_id(0) // tiles
    cols = tl.program_id(0) % tiles * B + tl.arange(0, B)
    key = tl.load(C + cols.to(tl.int64) * CS, cols < M, 0).to(tl.uint32)
    key = (key * 3432918353).to(tl.uint32)
    key = key << 15 | key >> 17
    key = (key * 461845907).to(tl.uint32)
    key = key << 13 | key >> 19
    for offset in range(triton.cdiv(N, R)):
        row = first_row + offset * R
        valid = (N % R == 0) | (row < N)
        seed = tl.load(S + row.to(tl.int64) * SS, valid, 0).to(tl.uint64)
        pos = tl.load(P + row.to(tl.int64) * PS, valid, 0).to(tl.uint32)
        h = _mix(tl.full((), 0, tl.uint32), seed.to(tl.uint32))
        if S.dtype.element_ty == tl.int32:
            hi_key = tl.where(seed.to(tl.int32) < 0, 3152982032, 0).to(
                tl.uint32
            )
            h = h ^ hi_key
            h = h << 13 | h >> 19
            h = (h * 5 + 3864292196).to(tl.uint32)
        else:
            h = _mix(h, (seed >> 32).to(tl.uint32))
        h = _mix(h, pos)
        h = (h << 13 | h >> 19) ^ key
        h = (h * 5 + 3864292196).to(tl.uint32) ^ 16
        h = h ^ h >> 16
        h = (h * 2246822507).to(tl.uint32)
        h = h ^ h >> 13
        h = (h * 3266489909).to(tl.uint32)
        h = h ^ h >> 16
        tl.store(OUT + row.to(tl.int64) * M + cols, h, valid & (cols < M))


def murmur_hash32(seed, positions, col_indices):
    n = seed.numel()
    m = col_indices.numel()
    out = seed.new_empty((n, m), dtype=torch.uint32)
    if n and m:
        rows = min(n, 1024)
        block = 512
        _kernel.run(
            seed,
            positions,
            col_indices,
            out,
            (n, m, seed.stride(0), positions.stride(0), col_indices.stride(0)),
            num_warps=4,
            grid=(rows * ((m + block - 1) // block),),
            warmup=False,
        )
    return out


__all__ = ["murmur_hash32"]
