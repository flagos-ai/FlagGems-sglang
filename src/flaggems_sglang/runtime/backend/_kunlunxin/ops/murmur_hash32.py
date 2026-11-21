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
def _rotl(x, R: tl.constexpr):
    return x << R | x >> 32 - R & (1 << R) - 1


@triton.jit
def _mix(h, k):
    k = k * -862048943
    k = _rotl(k, 15)
    k = k * 461845907
    h = _rotl(h ^ k, 13)
    return h * 5 - 430675100


@triton.jit(do_not_specialize=["S", "P", "C", "OUT"])
def _kernel(S, P, C, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    M: tl.constexpr = META[1]
    SS: tl.constexpr = META[2]
    PS: tl.constexpr = META[3]
    CS: tl.constexpr = META[4]
    HIGH: tl.constexpr = META[5]
    UNSIGNED: tl.constexpr = META[6]
    B: tl.constexpr = 1024
    tiles: tl.constexpr = triton.cdiv(M, B)
    OUT = OUT.to(tl.pointer_type(tl.int32))
    col = tl.program_id(0) % tiles * B + tl.arange(0, B)
    key = tl.load(C + col.to(tl.int64) * CS, col < M, 0).to(tl.int32)
    key = _rotl(_rotl(key * -862048943, 15) * 461845907, 13)
    ITERATIONS: tl.constexpr = (
        16
        if N >= 4096 and N % 16 == 0
        else (
            4
            if N >= 4096 and N % 4 == 0
            else 2 if N >= 2048 and N % 2 == 0 else 1
        )
    )
    ACTIVE: tl.constexpr = N // ITERATIONS
    for step in tl.static_range(ITERATIONS):
        row = tl.program_id(0) // tiles + step * ACTIVE
        valid = (N % ACTIVE == 0) | (row < N)
        if HIGH:
            low = tl.load(S + row.to(tl.int64) * SS, valid, 0)
            high = tl.load(S + row.to(tl.int64) * SS + 1, valid, 0)
        else:
            seed = tl.load(S + row.to(tl.int64) * SS, valid, 0)
            low = seed.to(tl.int32)
            if UNSIGNED:
                high = tl.full((), 0, tl.int32)
            else:
                high = tl.where(seed < 0, -1, 0)
        pos = tl.load(P + row.to(tl.int64) * PS, valid, 0).to(tl.int32)
        h = _mix(tl.full((), 0, tl.int32), low)
        if HIGH:
            h = _mix(h, high)
        else:
            high_key = tl.where(high < 0, -1141985264, 0)
            h = _rotl(h ^ high_key, 13) * 5 - 430675100
        h = _mix(h, pos)
        h = (_rotl(h, 13) ^ key) * 5 - 430675100 ^ 16
        h = h ^ h >> 16 & 65535
        h = h * -2048144789
        h = h ^ h >> 13 & 524287
        h = h * -1028477387
        h = h ^ h >> 16 & 65535
        tl.store(OUT + row.to(tl.int64) * M + col, h, valid & (col < M))


@triton.jit(do_not_specialize=["S", "P", "C", "OUT"])
def _vector(S, P, C, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    M: tl.constexpr = META[1]
    SS: tl.constexpr = META[2]
    PS: tl.constexpr = META[3]
    CS: tl.constexpr = META[4]
    R: tl.constexpr = META[5]
    B: tl.constexpr = META[6]
    HAS_HIGH: tl.constexpr = META[7]
    UNSIGNED: tl.constexpr = META[8]
    tiles: tl.constexpr = triton.cdiv(M, B)
    OUT = OUT.to(tl.pointer_type(tl.int32))
    tile = tl.program_id(0)
    col = tile % tiles * B + tl.arange(0, B)[None, :]
    I: tl.constexpr = META[9]
    ACTIVE: tl.constexpr = N // I
    key = tl.load(C + col.to(tl.int64) * CS, (M % B == 0) | (col < M), 0).to(
        tl.int32
    )
    key = _rotl(_rotl(key * -862048943, 15) * 461845907, 13)
    for step in tl.static_range(I):
        row = tile // tiles * R + tl.arange(0, R)[:, None] + step * ACTIVE
        if HAS_HIGH:
            low = tl.load(
                S + row.to(tl.int64) * SS, (N % R == 0) | (row < N), 0
            )
            high = tl.load(
                S + row.to(tl.int64) * SS + 1, (N % R == 0) | (row < N), 0
            )
        else:
            seed = tl.load(
                S + row.to(tl.int64) * SS, (N % R == 0) | (row < N), 0
            )
            low = seed.to(tl.int32)
            if UNSIGNED:
                high = tl.full((R, 1), 0, tl.int32)
            else:
                high = tl.where(seed < 0, -1, 0)
        pos = tl.load(
            P + row.to(tl.int64) * PS, (N % R == 0) | (row < N), 0
        ).to(tl.int32)
        h = _mix(tl.full((R, 1), 0, tl.int32), low)
        if HAS_HIGH:
            h = _mix(h, high)
        else:
            high_key = tl.where(high < 0, -1141985264, 0)
            h = _rotl(h ^ high_key, 13) * 5 - 430675100
        h = _mix(h, pos)
        h = (_rotl(h, 13) ^ key) * 5 - 430675100 ^ 16
        h = h ^ h >> 16 & 65535
        h = h * -2048144789
        h = h ^ h >> 13 & 524287
        h = h * -1028477387
        h = h ^ h >> 16 & 65535
        tl.store(
            OUT + row.to(tl.int64) * M + col,
            h,
            ((N % R == 0) | (row < N)) & ((M % B == 0) | (col < M)),
        )


def murmur_hash32(seed, positions, col_indices):
    n = seed.numel()
    m = col_indices.numel()
    out = seed.new_empty((n, m), dtype=torch.uint32)
    if n and m:
        iterations = (
            16
            if n >= 4096 and n % 16 == 0
            else (
                4
                if n >= 4096 and n % 4 == 0
                else 2 if n >= 2048 and n % 2 == 0 else 1
            )
        )
        high = seed.dtype in (torch.int64, torch.uint64)
        words = seed.unsqueeze(-1).view(torch.int32) if high else seed
        block = min(1024, 1 << (m - 1).bit_length())
        if n >= 32 and n % 2 == 0 and (block >= 16) and (m % block == 0):
            rows = 4 if n % 4 == 0 else 2
            vector_iterations = 2 if n >= 4096 and n % (rows * 2) == 0 else 1
            _vector.run(
                words,
                positions,
                col_indices,
                out,
                (
                    n,
                    m,
                    words.stride(0),
                    positions.stride(0),
                    col_indices.stride(0),
                    rows,
                    block,
                    high,
                    seed.dtype
                    in (torch.uint8, torch.uint16, torch.uint32, torch.uint64),
                    vector_iterations,
                ),
                grid=(
                    n
                    // (rows * vector_iterations)
                    * ((m + block - 1) // block),
                ),
                warmup=False,
                per_op_decision=False,
                isCloseUnrollControl=True,
                groups_per_cluster=rows,
                isCloseMemoryAsync=False,
            )
        else:
            _kernel.run(
                words,
                positions,
                col_indices,
                out,
                (
                    n,
                    m,
                    words.stride(0),
                    positions.stride(0),
                    col_indices.stride(0),
                    high,
                    seed.dtype
                    in (torch.uint8, torch.uint16, torch.uint32, torch.uint64),
                ),
                grid=(n // iterations * ((m + 1023) // 1024),),
                warmup=False,
                per_op_decision=False,
                isCloseUnrollControl=True,
                isCloseMemoryAsync=False,
            )
    return out


__all__ = ["murmur_hash32"]
