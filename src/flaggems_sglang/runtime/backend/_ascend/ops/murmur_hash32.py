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
def _rotl(x, R: tl.constexpr):
    return x << R | x >> 32 - R & (1 << R) - 1


@triton.jit
def _mix(h, k):
    k = k * -862048943
    k = _rotl(k, 15)
    k = k * 461845907
    h = _rotl(_xor_subtract(h, k), 13)
    return h * 5 + -430675100


@triton.jit(do_not_specialize=["S", "P", "C", "OUT"])
def _kernel(S, P, C, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    M: tl.constexpr = META[1]
    SS: tl.constexpr = META[2]
    PS: tl.constexpr = META[3]
    CS: tl.constexpr = META[4]
    R: tl.constexpr = triton.next_power_of_2(N) if N < 4 else 4
    B: tl.constexpr = triton.next_power_of_2(M) if M < 1024 else 1024
    HAS_HIGH: tl.constexpr = META[5]
    UNSIGNED: tl.constexpr = (
        S.dtype.element_ty == tl.uint8
        or S.dtype.element_ty == tl.uint16
        or S.dtype.element_ty == tl.uint32
        or (S.dtype.element_ty == tl.uint64)
    )
    tiles: tl.constexpr = triton.cdiv(M, B)
    OUT = OUT.to(tl.pointer_type(tl.int32))
    for tile in range(
        tl.program_id(0), triton.cdiv(N, R) * tiles, tl.num_programs(0)
    ):
        row = tile // tiles * R + tl.arange(0, R)[:, None]
        col = tile % tiles * B + tl.arange(0, B)[None, :]
        if HAS_HIGH:
            low = tl.load(
                S + _input_offset(row, SS, N), (N % R == 0) | (row < N), 0
            )
            high = tl.load(
                S + _input_offset(row, SS, N) + 1, (N % R == 0) | (row < N), 0
            )
        else:
            seed = tl.load(
                S + _input_offset(row, SS, N), (N % R == 0) | (row < N), 0
            )
            low = seed.to(tl.int32)
            if UNSIGNED:
                high = tl.full((R, 1), 0, tl.int32)
            else:
                high = tl.where(seed < 0, -1, 0)
        pos = tl.load(
            P + _input_offset(row, PS, N), (N % R == 0) | (row < N), 0
        ).to(tl.int32)
        key = tl.load(
            C + _input_offset(col, CS, M), (M % B == 0) | (col < M), 0
        ).to(tl.int32)
        h = _mix(tl.full((R, 1), 0, tl.int32), low)
        h = _mix(h, high)
        h = _mix(h, pos)
        h = _xor_subtract(_mix(h, key), 16)
        h = _xor_subtract(h, h >> 16 & 65535)
        h = h * -2048144789
        h = _xor_subtract(h, h >> 13 & 524287)
        h = h * -1028477387
        h = _xor_subtract(h, h >> 16 & 65535)
        tl.store(
            OUT + row.to(tl.int64) * M + col,
            h,
            ((N % R == 0) | (row < N)) & ((M % B == 0) | (col < M)),
        )


@triton.jit
def _xor_subtract(x, y):
    return (x | y) - (x & y)


@triton.jit(do_not_specialize=["S", "P", "C", "OUT"])
def _row_direct(S, P, C, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    M: tl.constexpr = META[1]
    SS: tl.constexpr = META[2]
    PS: tl.constexpr = META[3]
    CS: tl.constexpr = META[4]
    R: tl.constexpr = triton.next_power_of_2(N) if N < 8 else 8
    B: tl.constexpr = triton.next_power_of_2(M) if M < 1024 else 1024
    HAS_HIGH: tl.constexpr = META[5]
    UNSIGNED: tl.constexpr = (
        S.dtype.element_ty == tl.uint8
        or S.dtype.element_ty == tl.uint16
        or S.dtype.element_ty == tl.uint32
        or (S.dtype.element_ty == tl.uint64)
    )
    tiles: tl.constexpr = triton.cdiv(M, B)
    OUT = OUT.to(tl.pointer_type(tl.int32))
    tile = tl.program_id(0)
    row = tile // tiles * R + tl.arange(0, R)[:, None]
    col = tile % tiles * B + tl.arange(0, B)[None, :]
    if HAS_HIGH:
        low = tl.load(
            S + _input_offset(row, SS, N), (N % R == 0) | (row < N), 0
        )
        high = tl.load(
            S + _input_offset(row, SS, N) + 1, (N % R == 0) | (row < N), 0
        )
    else:
        seed = tl.load(
            S + _input_offset(row, SS, N), (N % R == 0) | (row < N), 0
        )
        low = seed.to(tl.int32)
        if UNSIGNED:
            high = tl.full((R, 1), 0, tl.int32)
        else:
            high = tl.where(seed < 0, -1, 0)
    pos = tl.load(
        P + _input_offset(row, PS, N), (N % R == 0) | (row < N), 0
    ).to(tl.int32)
    key = tl.load(
        C + _input_offset(col, CS, M), (M % B == 0) | (col < M), 0
    ).to(tl.int32)
    h = _mix(tl.full((R, 1), 0, tl.int32), low)
    if HAS_HIGH:
        h = _mix(h, high)
    else:
        high_key = tl.where(high == -1, -1141985264, 0)
        h = _rotl(_xor_subtract(h, high_key), 13) * 5 - 430675100
    h = _mix(h, pos)
    key = _rotl(key * -862048943, 15) * 461845907
    h = _xor_subtract(
        _xor_subtract(_rotl(h, 13), _rotl(key, 13)) * 5 - 430675100, 16
    )
    h = _xor_subtract(h, h >> 16 & 65535)
    h = h * -2048144789
    h = _xor_subtract(h, h >> 13 & 524287)
    h = h * -1028477387
    h = _xor_subtract(h, h >> 16 & 65535)
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
        has_high = seed.dtype in (torch.int64, torch.uint64)
        words = seed.unsqueeze(-1).view(torch.int32) if has_high else seed
        position_words = (
            positions.unsqueeze(-1).view(torch.int32)
            if positions.dtype in (torch.int64, torch.uint64)
            else positions
        )
        column_words = (
            col_indices.unsqueeze(-1).view(torch.int32)
            if col_indices.dtype in (torch.int64, torch.uint64)
            else col_indices
        )
        if n * m >= 262144 and (n + 7) // 8 * ((m + 1023) // 1024) <= 32768:
            _row_direct.run(
                words,
                position_words,
                column_words,
                out,
                (
                    n,
                    m,
                    words.stride(0),
                    position_words.stride(0),
                    column_words.stride(0),
                    has_high,
                ),
                grid=((n + 7) // 8 * ((m + 1023) // 1024),),
                warmup=False,
            )
        else:
            _kernel.run(
                words,
                position_words,
                column_words,
                out,
                (
                    n,
                    m,
                    words.stride(0),
                    position_words.stride(0),
                    column_words.stride(0),
                    has_high,
                ),
                grid=(min((n + 3) // 4 * ((m + 1023) // 1024), 32768),),
                warmup=False,
            )
    return out


__all__ = ["murmur_hash32"]
