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


@triton.jit
def _kernel(S, P, C, OUT, META: tl.constexpr):
    M: tl.constexpr = META[0]
    SS: tl.constexpr = META[1]
    PS: tl.constexpr = META[2]
    CS: tl.constexpr = META[3]
    B: tl.constexpr = 512
    tiles: tl.constexpr = triton.cdiv(M, B)
    row = tl.program_id(0) // tiles
    cols = tl.program_id(0) % tiles * B + tl.arange(0, B)
    seed = tl.load(S + row.to(tl.int64) * SS).to(tl.uint64)
    pos = tl.load(P + row.to(tl.int64) * PS).to(tl.uint32)
    key = tl.load(C + cols.to(tl.int64) * CS, cols < M, 0).to(tl.uint32)
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
    tl.store(OUT + row.to(tl.int64) * M + cols, h, (M % B == 0) | (cols < M))


@triton.jit
def _tiny_kernel(S, P, C, OUT, META: tl.constexpr):
    M: tl.constexpr = META[0]
    SS: tl.constexpr = META[1]
    PS: tl.constexpr = META[2]
    CS: tl.constexpr = META[3]
    B: tl.constexpr = 64
    tiles: tl.constexpr = triton.cdiv(M, B)
    row = tl.program_id(0) // tiles
    cols = tl.program_id(0) % tiles * B + tl.arange(0, B)
    seed = tl.load(S + row.to(tl.int64) * SS).to(tl.uint64)
    pos = tl.load(P + row.to(tl.int64) * PS).to(tl.uint32)
    key = tl.load(C + cols.to(tl.int64) * CS, cols < M, 0).to(tl.uint32)
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


def murmur_hash32(seed, positions, col_indices):
    n = seed.numel()
    m = col_indices.numel()
    out = seed.new_empty((n, m), dtype=torch.uint32)
    if n and m:
        blocks = n * ((m + 511) // 512)
        if n * m <= 4096:
            _tiny_kernel.run(
                seed,
                positions,
                col_indices,
                out,
                (
                    m,
                    seed.stride(0),
                    positions.stride(0),
                    col_indices.stride(0),
                ),
                num_warps=1,
                grid=(n * ((m + 63) // 64),),
                warmup=False,
            )
        else:
            _kernel.run(
                seed,
                positions,
                col_indices,
                out,
                (
                    m,
                    seed.stride(0),
                    positions.stride(0),
                    col_indices.stride(0),
                ),
                num_warps=2,
                grid=(blocks,),
                warmup=False,
            )
    return out


__all__ = ["murmur_hash32"]
