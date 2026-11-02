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
    B: tl.constexpr = META[10]
    x = tl.program_id(0) * B + tl.arange(0, B)
    if R * H * D > 0:
        d = x % D
        h = x // D % L
        r = x // (D * L) % R
        rank = x // (D * L * R)
        gh = rank * L + h
        mask = x < R * H * D
        q = tl.load(Q + r * Q0 + gh * Q1 + d, mask)
        k = tl.load(K + r * K0 + gh * K1 + d, mask)
        v = tl.load(V + r * V0 + gh * V1 + d, mask)
        dst = x // D * (3 * D) + d
        tl.store(Out + dst, q, mask)
        tl.store(Out + dst + D, k, mask)
        tl.store(Out + dst + 2 * D, v, mask)


@triton.jit
def _copy_output(X, Y, N: tl.constexpr, B: tl.constexpr):
    x = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + x, x < N)
    tl.store(Y + x, value, x < N)


def pack_qkv_destination_major(q, k, v, world_size, out):
    if q.ndim != 3 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v must have matching 3D shapes")
    rows, heads, size = q.shape
    if (
        not isinstance(world_size, int)
        or world_size <= 0
        or heads % world_size
    ):
        raise ValueError("world_size must be positive and divide global_heads")
    if (
        k.dtype != q.dtype
        or v.dtype != q.dtype
        or k.device != q.device
        or (v.device != q.device)
    ):
        raise ValueError("q, k, v must have matching dtype and device")
    if q.stride(2) != 1 or k.stride(2) != 1 or v.stride(2) != 1:
        raise ValueError("q, k, v must have final stride 1")
    local = heads // world_size
    shape = (world_size, rows, local, 3 * size)
    requested = out
    if out is None:
        out = torch.empty(shape, dtype=q.dtype, device=q.device)
    elif (
        out.shape != shape
        or out.dtype != q.dtype
        or out.device != q.device
        or (not out.is_contiguous())
    ):
        raise ValueError(
            "out must be contiguous with matching shape, dtype and device"
        )
    if requested is not None and rows * heads * size:
        begin = requested.data_ptr()
        end = begin + requested.numel() * requested.element_size()
        if any(
            (
                t.data_ptr() < end
                and begin
                < t.data_ptr()
                + t.element_size()
                * (
                    1
                    + (rows - 1) * t.stride(0)
                    + (heads - 1) * t.stride(1)
                    + size
                    - 1
                )
                for t in (q, k, v)
            )
        ):
            out = torch.empty(shape, dtype=q.dtype, device=q.device)
    block = 512 if rows >= 256 else 1024
    _pack.run(
        q,
        k,
        v,
        out,
        (
            rows,
            heads,
            size,
            local,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            v.stride(0),
            v.stride(1),
            block,
        ),
        num_warps=4,
        grid=(max(1, (rows * heads * size + block - 1) // block),),
        warmup=False,
    )
    if requested is not None and out is not requested:
        total = 3 * rows * heads * size
        _copy_output.run(
            out,
            requested,
            total,
            1024,
            grid=(max(1, (total + 1024 - 1) // 1024),),
            warmup=False,
        )
        return requested
    return out


__all__ = ["pack_qkv_destination_major"]
