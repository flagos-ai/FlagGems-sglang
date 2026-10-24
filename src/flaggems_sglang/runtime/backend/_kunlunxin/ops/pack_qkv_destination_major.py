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
def _stage(
    Q,
    K,
    V,
    T,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0) // tl.cdiv(H, BH)
    h = tl.program_id(0) % tl.cdiv(H, BH) * BH + tl.arange(0, BH)
    d = tl.arange(0, BD)
    mask = (h[:, None] < H) & (d[None, :] < D)
    q = tl.load(Q + row * Q0 + h[:, None] * Q1 + d[None, :], mask)
    k = tl.load(K + row * K0 + h[:, None] * K1 + d[None, :], mask)
    v = tl.load(V + row * V0 + h[:, None] * V1 + d[None, :], mask)
    dst = (row * H + h[:, None]) * (3 * D) + d[None, :]
    tl.store(T + dst, q, mask)
    tl.store(T + dst + D, k, mask)
    tl.store(T + dst + 2 * D, v, mask)


@triton.jit
def _transpose(
    T,
    O,
    R: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BM: tl.constexpr,
    BC: tl.constexpr,
):
    rank = tl.program_id(0) // tl.cdiv(R, BM)
    row = tl.program_id(0) % tl.cdiv(R, BM) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = (row[:, None] < R) & (col[None, :] < C)
    value = tl.load(T + (row[:, None] * W + rank) * C + col[None, :], mask)
    tl.store(O + (rank * R + row[:, None]) * C + col[None, :], value, mask)


@triton.jit
def _pack(Q, K, V, Out, META: tl.constexpr):
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
        rank = tl.program_id(0) // tl.cdiv(R, BM)
        row = tl.program_id(0) % tl.cdiv(R, BM) * BM + tl.arange(0, BM)
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
            k = tl.load(K + ki)
            v = tl.load(V + vi)
            tl.store(Out + dst, q)
            tl.store(Out + dst + D, k)
            tl.store(Out + dst + 2 * D, v)
        else:
            q = tl.load(Q + qi, mask)
            k = tl.load(K + ki, mask)
            v = tl.load(V + vi, mask)
            tl.store(Out + dst, q, mask)
            tl.store(Out + dst + D, k, mask)
            tl.store(Out + dst + 2 * D, v, mask)


@triton.jit
def _copy_output(X, Y, N: tl.constexpr, B: tl.constexpr):
    x = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + x, x < N)
    tl.store(Y + x, value, x < N)


def pack_qkv_destination_major(q, k, v, world_size, out):
    qshape = q.shape
    if len(qshape) != 3 or k.shape != qshape or v.shape != qshape:
        raise ValueError("q, k, v must have matching 3D shapes")
    rows, heads, size = qshape
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
    total = rows * heads * size
    if requested is not None and total:
        element = q.element_size()
        begin = requested.data_ptr()
        end = begin + 3 * total * element
        for tensor, strides in ((q, qs), (k, ks), (v, vs)):
            ptr = tensor.data_ptr()
            if ptr < end and begin < ptr + element * (
                (rows - 1) * strides[0] + (heads - 1) * strides[1] + size
            ):
                out = torch.empty(shape, dtype=dtype, device=device)
                break
    element = q.element_size()
    factor = 8 // element
    strides = (qs[0], qs[1], ks[0], ks[1], vs[0], vs[1])
    if (
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
    rank_width = triton.next_power_of_2(local * width)
    block = max(
        1, min(64 if rows % 64 == 0 else 32, 8192 // max(1, rank_width))
    )
    programs = max(1, world_size * triton.cdiv(rows, block))
    if factor > 1:
        a, b, c, target = (
            q.view(torch.int64),
            k.view(torch.int64),
            v.view(torch.int64),
            out.view(torch.int64),
        )
        strides = tuple((s // factor for s in strides))
    else:
        a, b, c, target = (q, k, v, out)
    if (
        block >= 32
        and rows % block == 0
        and (local * width == rank_width)
        and all((t.is_contiguous() for t in (q, k, v)))
    ):
        _pack[programs,](
            a,
            b,
            c,
            target,
            (rows, heads, width, local) + strides + (block,),
            num_warps=4,
            groups_per_cluster=64 if block >= 64 else 32,
        )
    elif total:
        temp = torch.empty(
            (rows, heads, 3 * width), dtype=a.dtype, device=device
        )
        bh = min(
            triton.next_power_of_2(heads),
            max(1, 4096 // triton.next_power_of_2(width)),
        )
        _stage[rows * triton.cdiv(heads, bh),](
            a,
            b,
            c,
            temp,
            rows,
            heads,
            width,
            *strides,
            bh,
            triton.next_power_of_2(width),
            groups_per_cluster=32,
        )
        columns = local * 3 * width
        bc = min(512, triton.next_power_of_2(columns))
        bm = max(1, min(32, 16384 // bc))
        _transpose[
            world_size * triton.cdiv(rows, bm), triton.cdiv(columns, bc)
        ](
            temp,
            target,
            rows,
            world_size,
            columns,
            bm,
            bc,
            groups_per_cluster=32,
        )
    if requested is not None and out is not requested:
        _copy_output[max(1, triton.cdiv(3 * total, 1024)),](
            out, requested, 3 * total, 1024, groups_per_cluster=32
        )
        return requested
    return out


__all__ = ["pack_qkv_destination_major"]
