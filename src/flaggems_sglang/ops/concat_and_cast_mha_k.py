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
def _copy(
    N,
    R,
    OUT,
    SIZE: tl.constexpr,
    H: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    NS0: tl.constexpr,
    NS1: tl.constexpr,
    NS2: tl.constexpr,
    RS0: tl.constexpr,
    RS2: tl.constexpr,
    PACK: tl.constexpr,
    B: tl.constexpr,
):
    if PACK:
        N = N.to(tl.pointer_type(tl.uint64))
        R = R.to(tl.pointer_type(tl.uint64))
        OUT = OUT.to(tl.pointer_type(tl.uint64))
    for tile in range(tl.program_id(0), tl.cdiv(SIZE, B), tl.num_programs(0)):
        i = tile * B + tl.arange(0, B)
        col = i % (DN + DR)
        row = i // (DN + DR)
        token = row // H
        head = row % H
        n = tl.load(
            N + token * NS0 + head * NS1 + col * NS2,
            (i < SIZE) & (col < DN),
            other=0,
        )
        r = tl.load(
            R + token * RS0 + (col - DN) * RS2,
            (i < SIZE) & (col >= DN),
            other=0,
        )
        tl.store(OUT + i, tl.where(col < DN, n, r), i < SIZE)


def concat_and_cast_mha_k(k, k_nope, k_rope):
    t, h, dn = k_nope.shape
    dr = k_rope.shape[-1]
    out = torch.empty((t, h, dn + dr), dtype=k.dtype, device=k.device)
    size = t * h * (dn + dr)
    if size:
        packed = (
            k.dtype == k_nope.dtype == k_rope.dtype == torch.bfloat16
            and dn % 4 == dr % 4 == 0
            and k_nope.stride(2) == k_rope.stride(2) == 1
            and k_nope.stride(0) % 4 == k_nope.stride(1) % 4 == 0
            and k_rope.stride(0) % 4 == 0
            and k_nope.storage_offset() % 4 == k_rope.storage_offset() % 4 == 0
        )
        factor = 4 if packed else 1
        block = 512
        _copy[(triton.cdiv(size // factor, block),)](
            k_nope,
            k_rope,
            out,
            size // factor,
            h,
            dn // factor,
            dr // factor,
            k_nope.stride(0) // factor,
            k_nope.stride(1) // factor,
            k_nope.stride(2),
            k_rope.stride(0) // factor,
            k_rope.stride(2),
            packed,
            block,
            num_warps=4,
        )
    return out


__all__ = ["concat_and_cast_mha_k"]
