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

__all__ = ["concat_mla_k"]


@triton.jit
def _copy(
    N,
    R,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    DN: tl.constexpr,
    DR: tl.constexpr,
    NS0: tl.constexpr,
    NS1: tl.constexpr,
    NS2: tl.constexpr,
    RS0: tl.constexpr,
    RS2: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    groups = tl.cdiv(H, BH)
    token = tl.program_id(0) // groups * BT + tl.arange(0, BT)
    head = tl.program_id(0) % groups * BH + tl.arange(0, BH)
    nc = tl.arange(0, BN)
    rc = tl.arange(0, BR)
    nv = tl.load(
        N
        + token[:, None, None] * NS0
        + head[None, :, None] * NS1
        + nc[None, None, :] * NS2,
        (token[:, None, None] < T)
        & (head[None, :, None] < H)
        & (nc[None, None, :] < DN),
        other=0,
    )
    rv = tl.load(
        R + token[:, None] * RS0 + rc[None, :] * RS2,
        (token[:, None] < T) & (rc[None, :] < DR),
        other=0,
    )
    base = OUT + (token[:, None, None] * H + head[None, :, None]) * (DN + DR)
    tl.store(
        base + nc[None, None, :],
        nv,
        (token[:, None, None] < T)
        & (head[None, :, None] < H)
        & (nc[None, None, :] < DN),
    )
    tl.store(
        base + DN + rc[None, None, :],
        tl.broadcast_to(rv[:, None, :], (BT, BH, BR)),
        (token[:, None, None] < T)
        & (head[None, :, None] < H)
        & (rc[None, None, :] < DR),
    )


def concat_mla_k(k, k_nope, k_rope):
    t, h, dn = k_nope.shape
    dr = k_rope.shape[2]
    out = (
        k
        if k.is_contiguous()
        else torch.empty(k.shape, dtype=k.dtype, device=k.device)
    )
    if t * h * (dn + dr):
        bn = triton.next_power_of_2(dn)
        br = triton.next_power_of_2(dr)
        bh = min(triton.next_power_of_2(h), max(1, 16384 // max(bn, br)))
        bt = min(triton.next_power_of_2(t), 4)
        if 128 < t <= 256:
            bt = 16
        elif 256 < t < 1024:
            bt = 32
        warps = 2 if 1 < t <= 32 else 1
        _copy[triton.cdiv(t, bt) * triton.cdiv(h, bh),](
            k_nope,
            k_rope,
            out,
            t,
            h,
            dn,
            dr,
            k_nope.stride(0),
            k_nope.stride(1),
            k_nope.stride(2),
            k_rope.stride(0),
            k_rope.stride(2),
            bt,
            bh,
            bn,
            br,
            num_warps=warps,
        )
    return out
