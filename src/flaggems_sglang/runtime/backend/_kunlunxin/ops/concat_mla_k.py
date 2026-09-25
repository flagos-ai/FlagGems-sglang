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


@triton.jit
def _linear(
    A,
    R,
    OUT,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    B: tl.constexpr,
    F: tl.constexpr,
    MODE: tl.constexpr,
):
    A = A.to(tl.pointer_type(tl.uint64))
    R = R.to(tl.pointer_type(tl.uint64))
    OUT = OUT.to(tl.pointer_type(tl.uint64))
    p = tl.program_id(0)
    v = tl.arange(0, B)
    i = p * B + v
    if T * H * D % B == 0:
        s = i
    else:
        s = tl.minimum(i, T * H * D - 1)
    row = s // D
    col = s % D
    a = tl.load(A + row * N + col)
    b = tl.load(A + row * N + D + col)
    r = tl.load(R + row // H * D + col)
    tl.store(OUT + row * (N + D) + col, a, i < T * H * D)
    tl.store(OUT + row * (N + D) + D + col, b, i < T * H * D)
    tl.store(OUT + row * (N + D) + N + col, r, i < T * H * D)


def _fallback(k, a, r):
    t, h, n = a.shape
    d = r.shape[2]
    out = (
        k
        if k.is_contiguous()
        else torch.empty(k.shape, dtype=k.dtype, device=k.device)
    )
    if t * h * (n + d):
        bh = min(8, triton.next_power_of_2(h))
        grid = t * triton.cdiv(h, bh)
        _heads.run(
            a,
            r,
            out,
            t,
            h,
            n,
            d,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            r.stride(0),
            r.stride(2),
            bh,
            triton.next_power_of_2(max(n, 1)),
            triton.next_power_of_2(max(d, 1)),
            grid,
            1,
            grid=(grid,),
            warmup=False,
            num_warps=4,
            num_stages=1,
        )
    return out


def concat_mla_k(k, k_nope, k_rope):
    a, r = (k_nope, k_rope)
    t, h, n = a.shape
    d = r.shape[2]
    eligible = (
        t * h * d > 0
        and n == 2 * d
        and (d % 4 == 0)
        and k.is_contiguous()
        and a.is_contiguous()
        and r.is_contiguous()
        and (k.dtype == a.dtype == r.dtype)
        and (k.dtype in (torch.bfloat16, torch.float16))
        and (
            k.storage_offset() % 4
            == a.storage_offset() % 4
            == r.storage_offset() % 4
            == 0
        )
    )
    if not eligible:
        return _fallback(k, a, r)
    n //= 4
    d //= 4
    total = t * h * d
    b = min(16384, triton.next_power_of_2(total))
    while total % b:
        b //= 2
    _linear.run(
        a,
        r,
        k,
        t,
        h,
        n,
        d,
        b,
        4,
        2,
        grid=(triton.cdiv(total, b),),
        warmup=False,
        num_warps=4,
        num_stages=1,
    )
    return k
