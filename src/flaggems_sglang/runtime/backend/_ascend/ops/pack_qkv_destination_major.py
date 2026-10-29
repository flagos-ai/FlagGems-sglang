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
from triton.language.extra.cann.extension import extract_slice, insert_slice


@triton.jit
def _base_fast(Q, K, V, Out, META: tl.constexpr):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    Q0: tl.constexpr = META[4]
    Q1: tl.constexpr = META[5]
    K0: tl.constexpr = META[6]
    K1: tl.constexpr = META[7]
    V0: tl.constexpr = META[8]
    V1: tl.constexpr = META[9]
    BM: tl.constexpr = META[10]
    if R * H * D > 0:
        BC: tl.constexpr = triton.next_power_of_2(L * D)
        for tile in range(tl.program_id(0), H // L * tl.cdiv(R, BM), META[12]):
            rank = tile // tl.cdiv(R, BM)
            row = tile % tl.cdiv(R, BM) * BM + tl.arange(0, BM)
            col = tl.arange(0, BC)
            head = col // D
            d = col % D
            gh = rank * L + head
            mask = (row[:, None] < R) & (col[None, :] < L * D)
            if Q1 == D:
                qi = row[:, None] * Q0 + rank * L * D + col[None, :]
            else:
                qi = row[:, None] * Q0 + gh[None, :] * Q1 + d[None, :]
            if K1 == D:
                ki = row[:, None] * K0 + rank * L * D + col[None, :]
            else:
                ki = row[:, None] * K0 + gh[None, :] * K1 + d[None, :]
            if V1 == D:
                vi = row[:, None] * V0 + rank * L * D + col[None, :]
            else:
                vi = row[:, None] * V0 + gh[None, :] * V1 + d[None, :]
            dst = (
                (rank * R + row[:, None]) * (L * 3 * D)
                + head[None, :] * (3 * D)
                + d[None, :]
            )
            if R % BM == 0 and L * D == BC:
                q = tl.load(Q + qi)
                tl.store(Out + dst, q)
                k = tl.load(K + ki)
                tl.store(Out + dst + D, k)
                v = tl.load(V + vi)
                tl.store(Out + dst + 2 * D, v)
            else:
                q = tl.load(Q + qi, mask)
                tl.store(Out + dst, q, mask)
                k = tl.load(K + ki, mask)
                tl.store(Out + dst + D, k, mask)
                v = tl.load(V + vi, mask)
                tl.store(Out + dst + 2 * D, v, mask)


@triton.jit
def _base_pack(Q, K, V, Out, META: tl.constexpr):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D0: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    QS0: tl.constexpr = META[4]
    QS1: tl.constexpr = META[5]
    KS0: tl.constexpr = META[6]
    KS1: tl.constexpr = META[7]
    VS0: tl.constexpr = META[8]
    VS1: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    D: tl.constexpr = D0
    BD: tl.constexpr = triton.next_power_of_2(D)
    BL: tl.constexpr = triton.next_power_of_2(L)
    if R * H * D > 0:
        rank = tl.program_id(0) // tl.cdiv(R, B)
        row = tl.program_id(0) % tl.cdiv(R, B) * B + tl.arange(0, B)
        h = tl.arange(0, BL)
        d = tl.arange(0, BD)
        mask = (
            (row[:, None, None] < R)
            & (h[None, :, None] < L)
            & (d[None, None, :] < D)
        )
        gh = rank * L + h
        q = tl.load(
            Q
            + row[:, None, None] * QS0
            + gh[None, :, None] * QS1
            + d[None, None, :],
            mask,
        )
        k = tl.load(
            K
            + row[:, None, None] * KS0
            + gh[None, :, None] * KS1
            + d[None, None, :],
            mask,
        )
        v = tl.load(
            V
            + row[:, None, None] * VS0
            + gh[None, :, None] * VS1
            + d[None, None, :],
            mask,
        )
        dst = ((rank * R + row[:, None, None]) * L + h[None, :, None]) * (
            3 * D
        ) + d[None, None, :]
        tl.store(Out + dst, q, mask)
        tl.store(Out + dst + D, k, mask)
        tl.store(Out + dst + 2 * D, v, mask)


@triton.jit
def _base_pack_large(Q, K, V, Out, META: tl.constexpr):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D0: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    QS0: tl.constexpr = META[4]
    QS1: tl.constexpr = META[5]
    KS0: tl.constexpr = META[6]
    KS1: tl.constexpr = META[7]
    VS0: tl.constexpr = META[8]
    VS1: tl.constexpr = META[9]
    B: tl.constexpr = META[10]
    D: tl.constexpr = D0
    BD: tl.constexpr = triton.next_power_of_2(D)
    BL: tl.constexpr = triton.next_power_of_2(L)
    if R * H * D > 0:
        for tile in range(
            tl.program_id(0).to(tl.int64),
            H // L * tl.cdiv(R, B),
            tl.num_programs(0),
        ):
            rank = tile // tl.cdiv(R, B)
            row = tile % tl.cdiv(R, B) * B + tl.arange(0, B)
            h = tl.arange(0, BL)
            d = tl.arange(0, BD)
            mask = (
                (row[:, None, None] < R)
                & (h[None, :, None] < L)
                & (d[None, None, :] < D)
            )
            gh = rank * L + h
            q = tl.load(
                Q
                + row[:, None, None] * QS0
                + gh[None, :, None] * QS1
                + d[None, None, :],
                mask,
            )
            k = tl.load(
                K
                + row[:, None, None] * KS0
                + gh[None, :, None] * KS1
                + d[None, None, :],
                mask,
            )
            v = tl.load(
                V
                + row[:, None, None] * VS0
                + gh[None, :, None] * VS1
                + d[None, None, :],
                mask,
            )
            dst = ((rank * R + row[:, None, None]) * L + h[None, :, None]) * (
                3 * D
            ) + d[None, None, :]
            tl.store(Out + dst, q, mask)
            tl.store(Out + dst + D, k, mask)
            tl.store(Out + dst + 2 * D, v, mask)


@triton.jit
def _base_copy_output(X, Y, N: tl.constexpr, B: tl.constexpr):
    for tile in range(tl.program_id(0), tl.cdiv(N, B), tl.num_programs(0)):
        x = tile * B + tl.arange(0, B)
        value = tl.load(X + x, x < N)
        tl.store(Y + x, value, x < N)


@triton.jit
def _slice_chunks(Q, K, V, Out, META: tl.constexpr):
    R: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    D: tl.constexpr = META[2]
    L: tl.constexpr = META[3]
    BM: tl.constexpr = META[10]
    BC: tl.constexpr = triton.next_power_of_2(L * D)
    N: tl.constexpr = BM * BC
    tl.static_assert(L * D == BC)
    if R * H * D > 0:
        col = tl.arange(0, BC)
        head = col // D
        d = col % D
        lane = tl.arange(0, N)
        for tile in range(
            (
                tl.program_id(0).to(tl.int64)
                if R % BM == 0
                else tl.program_id(0).to(tl.int64) * (H // L * tl.cdiv(R, BM))
            ),
            H // L * tl.cdiv(R, BM),
            tl.num_programs(0) if R % BM == 0 else 1,
        ):
            rank = tile // tl.cdiv(R, BM)
            row_start = tile % tl.cdiv(R, BM) * BM
            row = row_start + tl.arange(0, BM)
            gh = rank * L + head
            qi = row[:, None] * META[4] + gh[None, :] * META[5] + d[None, :]
            ki = row[:, None] * META[6] + gh[None, :] * META[7] + d[None, :]
            vi = row[:, None] * META[8] + gh[None, :] * META[9] + d[None, :]
            if R % BM == 0:
                q = tl.load(Q + qi)
                k = tl.load(K + ki)
                v = tl.load(V + vi)
            else:
                q = tl.load(Q + qi, row[:, None] < R, 0)
                k = tl.load(K + ki, row[:, None] < R, 0)
                v = tl.load(V + vi, row[:, None] < R, 0)
            packed = tl.full((4 * BM * L, D), 0, tl.int32)
            packed = insert_slice(
                packed,
                q.reshape(BM * L, D),
                offsets=(0, 0),
                sizes=(BM * L, D),
                strides=(3, 1),
            )
            packed = insert_slice(
                packed,
                k.reshape(BM * L, D),
                offsets=(1, 0),
                sizes=(BM * L, D),
                strides=(3, 1),
            )
            packed = insert_slice(
                packed,
                v.reshape(BM * L, D),
                offsets=(2, 0),
                sizes=(BM * L, D),
                strides=(3, 1),
            )
            for chunk in tl.static_range(3):
                j = chunk * N + lane
                value = extract_slice(
                    packed.reshape(4 * N),
                    offsets=(chunk * N,),
                    sizes=(N,),
                    strides=(1,),
                )
                dst = (rank * R + row_start) * L * 3 * D + j
                if R % BM == 0:
                    tl.store(Out + dst, value)
                else:
                    tl.store(
                        Out + dst, value, row_start + j // (3 * L * D) < R
                    )


def _eligible32(q, k, v, world_size):
    if q.dtype != torch.bfloat16:
        return False
    r, h, d = q.shape
    element = q.element_size()
    factor = 8 // element
    if factor <= 1 or q.numel() * element < 33554432 or r % 64:
        return False
    if not all(
        (
            t.is_contiguous()
            and d % factor == 0
            and (t.storage_offset() % factor == 0)
            for t in (q, k, v)
        )
    ):
        return False
    width = h // world_size * (d // factor)
    return (
        width > 0 and width & width - 1 == 0 and (h // world_size * d <= 512)
    )


def pack_qkv_destination_major(q, k, v, world_size, out):
    shape_in = q.shape
    if len(shape_in) != 3 or k.shape != shape_in or v.shape != shape_in:
        raise ValueError("q, k, v must have matching 3D shapes")
    rows, heads, size = shape_in
    if (
        not isinstance(world_size, int)
        or world_size <= 0
        or heads % world_size
    ):
        raise ValueError("world_size must be positive and divide global_heads")
    dtype, device = (q.dtype, q.device)
    if (
        k.dtype != dtype
        or v.dtype != dtype
        or k.device != device
        or (v.device != device)
    ):
        raise ValueError("q, k, v must have matching dtype and device")
    qs, ks, vs = (q.stride(), k.stride(), v.stride())
    if qs[2] != 1 or ks[2] != 1 or vs[2] != 1:
        raise ValueError("q, k, v must have final stride 1")
    local = heads // world_size
    shape = (world_size, rows, local, 3 * size)
    requested = out
    if out is None:
        out = torch.empty(shape, dtype=dtype, device=device)
    elif (
        out.shape != shape
        or out.dtype != dtype
        or out.device != device
        or (not out.is_contiguous())
    ):
        raise ValueError(
            "out must be contiguous with matching shape, dtype and device"
        )
    element = q.element_size()
    if requested is not None and rows * heads * size:
        begin = requested.data_ptr()
        end = begin + 3 * rows * heads * size * element
        for tensor, strides in ((q, qs), (k, ks), (v, vs)):
            ptr = tensor.data_ptr()
            if ptr < end and begin < ptr + element * (
                (rows - 1) * strides[0] + (heads - 1) * strides[1] + size
            ):
                out = torch.empty(shape, dtype=dtype, device=device)
                break
    if (
        _eligible32(q, k, v, world_size) and local * size <= 512
    ) and out.storage_offset() % 2 == 0:
        block = 16 if rows * heads * size * element >= 134217728 else 8
        programs = max(1, min(80, world_size * ((rows + block - 1) // block)))
        meta = (
            (rows, heads, size // 2, local)
            + tuple((s // 2 for s in qs[:2] + ks[:2] + vs[:2]))
            + (block, 1, 1)
        )
        _slice_chunks[programs,](
            q.view(torch.int32),
            k.view(torch.int32),
            v.view(torch.int32),
            out.view(torch.int32),
            meta,
            num_warps=1,
            multibuffer=False,
        )
    else:
        factor = (
            1 if rows * heads * size * element < 33554432 else 8 // element
        )
        strides = qs[:2] + ks[:2] + vs[:2]
        if factor > 1 and (
            size % factor
            or any((s % factor for s in strides))
            or (
                q.storage_offset()
                | k.storage_offset()
                | v.storage_offset()
                | out.storage_offset()
            )
            % factor
        ):
            factor = 1
        width = size // factor
        if factor > 1:
            q, k, v, target = (
                q.view(torch.int64),
                k.view(torch.int64),
                v.view(torch.int64),
                out.view(torch.int64),
            )
            strides = tuple((s // factor for s in strides))
        else:
            target = out
        fast = (
            factor > 1
            and rows * heads * size * element >= 67108864
            and (rows % 64 == 0)
            and (
                local * width == 1 << (max(1, local * width) - 1).bit_length()
            )
            and (strides == (heads * width, width) * 3)
        )
        if fast:
            block = max(
                1,
                min(
                    64,
                    8192
                    // max(1, 1 << (max(1, local * width) - 1).bit_length()),
                ),
            )
            programs = max(
                1, min(80, world_size * ((rows + block - 1) // block))
            )
            meta = (rows, heads, width, local) + strides + (block, 1, programs)
            _base_fast[programs,](
                q, k, v, target, meta, num_warps=1, multibuffer=False
            )
        else:
            block = max(
                1,
                min(
                    16,
                    4096
                    // max(
                        1,
                        (1 << (max(1, local) - 1).bit_length())
                        * (1 << (max(1, width) - 1).bit_length()),
                    ),
                ),
            )
            tiles = max(1, world_size * ((rows + block - 1) // block))
            programs = min(80, tiles) if tiles > 65535 else tiles
            meta = (rows, heads, width, local) + strides + (block, 1, programs)
            if tiles > 65535:
                _base_pack_large[programs,](q, k, v, target, meta, num_warps=1)
            else:
                _base_pack[programs,](q, k, v, target, meta, num_warps=1)
    if requested is not None and out is not requested:
        total = 3 * rows * heads * size
        _base_copy_output[max(1, min(80, (total + 1024 - 1) // 1024)),](
            out, requested, total, 1024
        )
        return requested
    return out


__all__ = ["pack_qkv_destination_major"]
