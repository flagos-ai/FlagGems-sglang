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


@triton.jit(do_not_specialize=["A", "R", "O"])
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


def _native(fn, pointers, constants, programs, warps):
    fn[(programs, 1, 1)](
        *pointers,
        *constants,
        num_warps=warps,
        num_stages=1,
    )


def _fallback(k, k_nope, k_rope):
    a, r = (k_nope, k_rope)
    t, h, n = a.shape
    d = r.shape[2]
    out = (
        k
        if k.is_contiguous()
        else torch.empty(k.shape, dtype=k.dtype, device=k.device)
    )
    if t * h * (n + d) == 0:
        return out
    bn = triton.next_power_of_2(max(n, 1))
    br = triton.next_power_of_2(max(d, 1))
    bh = min(triton.next_power_of_2(h), max(1, 16384 // max(bn, br)))
    grid = t * triton.cdiv(h, bh)
    _native(
        _heads,
        (a, r, out),
        (
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
            bn,
            br,
            grid,
            1,
        ),
        grid,
        4,
    )
    return out


@triton.jit(do_not_specialize=["NOPE", "ROPE", "OUT"])
def _compact(
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
    a, r = (k_nope, k_rope)
    t, h, n = a.shape
    d = r.shape[2]
    if not (
        t * h * n * d
        and n & n - 1 == 0
        and (d & d - 1 == 0)
        and (h % 64 == 0)
        and k.is_contiguous()
        and a.is_contiguous()
        and r.is_contiguous()
    ):
        return _fallback(k, a, r)
    at = (
        "*bf16"
        if a.dtype == torch.bfloat16
        else (
            "*fp16"
            if a.dtype == torch.float16
            else "*fp32" if a.dtype == torch.float32 else None
        )
    )
    rt = (
        "*bf16"
        if r.dtype == torch.bfloat16
        else (
            "*fp16"
            if r.dtype == torch.float16
            else "*fp32" if r.dtype == torch.float32 else None
        )
    )
    kt = (
        "*bf16"
        if k.dtype == torch.bfloat16
        else (
            "*fp16"
            if k.dtype == torch.float16
            else "*fp32" if k.dtype == torch.float32 else None
        )
    )
    if at is None or rt is None or kt is None:
        return _fallback(k, a, r)
    rows = 64 if t <= 64 or h % 128 else 128
    grid = t * (h // rows)
    _compact[(grid, 1, 1)](a, r, k, h, n, d, rows, num_warps=4, num_stages=1)
    return k
