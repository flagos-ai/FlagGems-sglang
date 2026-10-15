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
def _bf16(value):
    bits = value.to(tl.float32).to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    rounded = tl.where(
        (bits & 0x7FFFFFFF) > 0x7F800000, (bits >> 16) | 0x40, rounded
    )
    return rounded.to(tl.uint16)


@triton.jit
def _rows(
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
    BN: tl.constexpr,
    BR: tl.constexpr,
    PACK: tl.constexpr,
):
    if PACK:
        N = N.to(tl.pointer_type(tl.uint64))
        R = R.to(tl.pointer_type(tl.uint64))
        OUT = OUT.to(tl.pointer_type(tl.uint64))
    groups: tl.constexpr = triton.cdiv(H, BH)
    token = tl.program_id(0) // groups
    head = tl.program_id(0) % groups * BH + tl.arange(0, BH)
    nc = tl.arange(0, BN)
    rc = tl.arange(0, BR)
    nv = tl.load(
        N + token * NS0 + head[:, None] * NS1 + nc[None, :] * NS2,
        (head[:, None] < H) & (nc[None, :] < DN),
        other=0,
    )
    rv = tl.load(R + token * RS0 + rc * RS2, rc < DR, other=0)
    if not PACK and OUT.dtype.element_ty == tl.bfloat16:
        nv = _bf16(nv)
        rv = _bf16(rv)
        OUT = OUT.to(tl.pointer_type(tl.uint16))
    base = OUT + (token * H + head[:, None]) * (DN + DR)
    tl.store(base + nc[None, :], nv, (head[:, None] < H) & (nc[None, :] < DN))
    tl.store(
        base + DN + rc[None, :],
        tl.broadcast_to(rv[None, :], (BH, BR)),
        (head[:, None] < H) & (rc[None, :] < DR),
    )


def concat_and_cast_mha_k(k, k_nope, k_rope):
    t, h, dn = k_nope.shape
    dr = k_rope.shape[-1]
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
        bn = triton.next_power_of_2(dn // factor)
        br = triton.next_power_of_2(dr // factor)
        bh = min(triton.next_power_of_2(h), max(1, 4096 // max(bn, br)))
        if not packed:
            bh = 1
        _rows[(t * triton.cdiv(h, bh),)](
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
            bn,
            br,
            packed,
            groups_per_cluster=32,
        )
    return out


__all__ = ["concat_and_cast_mha_k"]
