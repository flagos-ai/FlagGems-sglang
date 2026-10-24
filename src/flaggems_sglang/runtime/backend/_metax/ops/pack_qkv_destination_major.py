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
    block = 512 if rows >= 256 else 1024
    meta = (
        rows,
        heads,
        size,
        local,
        qs[0],
        qs[1],
        ks[0],
        ks[1],
        vs[0],
        vs[1],
        block,
    )
    _pack[max(1, triton.cdiv(total, block)),](q, k, v, out, meta, num_warps=2)
    if requested is not None and out is not requested:
        _copy_output[max(1, triton.cdiv(3 * total, 1024)),](
            out, requested, 3 * total, 1024
        )
        return requested
    return out


__all__ = ["pack_qkv_destination_major"]
