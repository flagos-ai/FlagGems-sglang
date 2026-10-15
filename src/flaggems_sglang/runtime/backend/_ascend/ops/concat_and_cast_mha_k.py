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
    WORK: tl.constexpr,
):
    groups: tl.constexpr = tl.cdiv(H, BH)
    nc = tl.arange(0, BN)
    rc = tl.arange(0, BR)
    for work in range(tl.program_id(0), WORK, tl.num_programs(0)):
        token = work // groups
        head = work % groups * BH + tl.arange(0, BH)
        if H % BH == 0:
            valid = tl.full((BH,), True, tl.int1)
        else:
            valid = head < H
        nv = tl.load(
            N + token * NS0 + head[:, None] * NS1 + nc[None, :] * NS2,
            valid[:, None] & (nc[None, :] < DN),
            other=0,
        )
        rv = tl.load(R + token * RS0 + rc * RS2, rc < DR, other=0)
        base = OUT + (token * H + head[:, None]) * (DN + DR)
        tl.store(base + nc[None, :], nv, valid[:, None] & (nc[None, :] < DN))
        tl.store(
            base + DN + rc[None, :],
            tl.broadcast_to(rv[None, :], (BH, BR)),
            valid[:, None] & (rc[None, :] < DR),
        )


def concat_and_cast_mha_k(k, k_nope, k_rope):
    t, h, dn = k_nope.shape
    dr = k_rope.shape[-1]
    out = torch.empty((t, h, dn + dr), dtype=k.dtype, device=k.device)
    if t * h * (dn + dr):
        bn = triton.next_power_of_2(dn)
        br = triton.next_power_of_2(dr)
        bh = min(triton.next_power_of_2(h), max(1, 4096 // max(bn, br)))
        work = t * triton.cdiv(h, bh)
        _copy[(min(work, 40),)](
            k_nope,
            k_rope,
            out,
            h,
            dn,
            dr,
            k_nope.stride(0),
            k_nope.stride(1),
            k_nope.stride(2),
            k_rope.stride(0),
            k_rope.stride(2),
            bh,
            bn,
            br,
            work,
            num_warps=1,
        )
    return out


__all__ = ["concat_and_cast_mha_k"]
