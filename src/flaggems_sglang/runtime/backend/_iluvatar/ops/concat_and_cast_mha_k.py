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
def _segments(
    N,
    R,
    OUT,
    H: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    NS0: tl.constexpr,
    NS1: tl.constexpr,
    NS2: tl.constexpr,
    RS0: tl.constexpr,
    RS2: tl.constexpr,
    BH: tl.constexpr,
    BC: tl.constexpr,
    PACK: tl.constexpr,
):
    if PACK:
        N = N.to(tl.pointer_type(tl.uint64))
        R = R.to(tl.pointer_type(tl.uint64))
        OUT = OUT.to(tl.pointer_type(tl.uint64))
    ns: tl.constexpr = (DN + BC - 1) // BC
    rs: tl.constexpr = (DR + BC - 1) // BC
    group = tl.program_id(0) // (ns + rs)
    segment = tl.program_id(0) % (ns + rs)
    groups: tl.constexpr = (H + BH - 1) // BH
    token = group // groups
    head = group % groups * BH + tl.arange(0, BH)
    c = tl.arange(0, BC)
    base = OUT + (token * H + head[:, None]) * (DN + DR)
    if segment < ns:
        col = segment * BC + c
        nv = tl.load(
            N + token * NS0 + head[:, None] * NS1 + col[None, :] * NS2,
            (head[:, None] < H) & (col[None, :] < DN),
            other=0,
        )
        tl.store(
            base + col[None, :],
            nv,
            (head[:, None] < H) & (col[None, :] < DN),
        )
    else:
        col = (segment - ns) * BC + c
        rv = tl.load(R + token * RS0 + col * RS2, col < DR, other=0)
        tl.store(
            base + DN + col[None, :],
            tl.broadcast_to(rv[None, :], (BH, BC)),
            (head[:, None] < H) & (col[None, :] < DR),
        )


def concat_and_cast_mha_k(k, k_nope, k_rope):
    t, h, dn = k_nope.shape
    dr = k_rope.shape[2]
    out = torch.empty((t, h, dn + dr), dtype=k.dtype, device=k.device)
    if t * h * (dn + dr):
        packed = (
            k.dtype == k_nope.dtype == k_rope.dtype == torch.bfloat16
            and dn % 4 == dr % 4 == 0
            and k_nope.stride(2) == k_rope.stride(2) == 1
            and k_nope.stride(0) % 4 == k_nope.stride(1) % 4 == 0
            and k_rope.stride(0) % 4 == 0
            and k_nope.storage_offset() % 4 == k_rope.storage_offset() % 4 == 0
        )
        factor = 4 if packed else 1
        bc = min(64 // factor, triton.next_power_of_2(max(dn, dr) // factor))
        bh = min(triton.next_power_of_2(h), 1024 // (bc * factor))
        segments = triton.cdiv(dn // factor, bc) + triton.cdiv(
            dr // factor, bc
        )
        _segments[(t * triton.cdiv(h, bh) * segments,)](
            k_nope,
            k_rope,
            out,
            h,
            dn // factor,
            dr // factor,
            k_nope.stride(0) // factor,
            k_nope.stride(1) // factor,
            k_nope.stride(2),
            k_rope.stride(0) // factor,
            k_rope.stride(2),
            bh,
            bc,
            packed,
            num_warps=4,
        )
    return out


__all__ = ["concat_and_cast_mha_k"]
