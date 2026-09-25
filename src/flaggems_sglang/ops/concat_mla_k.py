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
def _heads(
    A,
    R,
    O,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    AS2: tl.constexpr,
    RS0: tl.constexpr,
    RS2: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
    NP: tl.constexpr,
    F: tl.constexpr,
):
    groups = tl.cdiv(H, BH)
    for task in range(tl.program_id(0), T * groups, NP):
        token = task // groups
        heads = task % groups * BH + tl.arange(0, BH)
        a = tl.arange(0, BN)
        r = tl.arange(0, BR)
        if N > 0:
            v = tl.load(
                A + token * AS0 + heads[:, None] * AS1 + a[None, :] * AS2,
                (heads[:, None] < H) & (a[None, :] < N),
                other=0,
            )
            tl.store(
                O + (token * H + heads[:, None]) * (N + D) + a[None, :],
                v,
                (heads[:, None] < H) & (a[None, :] < N),
            )
        if D > 0:
            v = tl.load(R + token * RS0 + r * RS2, r < D, other=0)
            tl.store(
                O + (token * H + heads[:, None]) * (N + D) + N + r[None, :],
                tl.broadcast_to(v[None, :], (BH, BR)),
                (heads[:, None] < H) & (r[None, :] < D),
            )


ROWS = 8


@triton.jit
def _cat(
    NOPE,
    ROPE,
    OUT,
    H: tl.constexpr,
    N: tl.constexpr,
    P: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    cn = tl.arange(0, N)
    cp = tl.arange(0, P)
    tl.store(
        OUT + rows[:, None] * (N + P) + cn[None, :],
        tl.load(NOPE + rows[:, None] * N + cn[None, :]),
    )
    r = tl.load(ROPE + pid * R // H * P + cp)
    tl.store(
        OUT + rows[:, None] * (N + P) + N + cp[None, :],
        tl.broadcast_to(r[None, :], (R, P)),
    )


def concat_mla_k(k, k_nope, k_rope):
    tokens, heads, dim = k.shape
    rope = k_rope.shape[2]
    nope = dim - rope
    if (
        nope == 0
        or rope == 0
        or nope & nope - 1
        or rope & rope - 1
        or heads % ROWS
        or (not k.is_contiguous())
        or (not k_nope.is_contiguous())
        or (not k_rope.is_contiguous())
    ):
        out = (
            k
            if k.is_contiguous()
            else torch.empty(k.shape, dtype=k.dtype, device=k.device)
        )
        if tokens * heads * (nope + rope):
            bh = min(8, triton.next_power_of_2(heads))
            grid = tokens * triton.cdiv(heads, bh)
            _heads.run(
                k_nope,
                k_rope,
                out,
                tokens,
                heads,
                nope,
                rope,
                k_nope.stride(0),
                k_nope.stride(1),
                k_nope.stride(2),
                k_rope.stride(0),
                k_rope.stride(2),
                bh,
                triton.next_power_of_2(max(nope, 1)),
                triton.next_power_of_2(max(rope, 1)),
                grid,
                1,
                grid=(grid,),
                warmup=False,
                num_warps=4,
                num_stages=1,
            )
        return out
    blocks = tokens * (heads // ROWS)
    if blocks:
        _cat[blocks,](
            k_nope,
            k_rope,
            k,
            heads,
            nope,
            rope,
            ROWS,
            num_warps=8,
            num_stages=1,
        )
    return k
