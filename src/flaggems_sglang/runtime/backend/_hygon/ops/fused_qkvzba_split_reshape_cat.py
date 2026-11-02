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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl


def _align(n):
    return (n + 7) & ~7


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["B", "NQK", "R", "HQ", "HV"],
)
@triton.jit
def _qkvzba_split_reshape_cat_kernel(
    qkvz_ptr,
    ba_ptr,
    qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    B,
    NQK: tl.constexpr,
    R: tl.constexpr,
    HQ: tl.constexpr,
    HV: tl.constexpr,
    G: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_R: tl.constexpr,
    EVEN_Q: tl.constexpr,
    EVEN_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    g0 = tl.program_id(1) * G

    per_head: tl.constexpr = 2 * HQ + 2 * R * HV
    src = qkvz_ptr + row * (NQK * per_head) + g0 * per_head
    dst = qkv_ptr + row * (NQK * (2 * HQ + R * HV))

    hs = tl.arange(0, G)[:, None]

    if EVEN_Q:
        es = tl.arange(0, HQ)[None, :]
        q = tl.load(src + hs * per_head + es)
        k = tl.load(src + hs * per_head + HQ + es)
        tl.store(dst + g0 * HQ + hs * HQ + es, q)
        tl.store(dst + (NQK + g0) * HQ + hs * HQ + es, k)
    else:
        es = tl.arange(0, BLOCK_HQ)[None, :]
        mq = es < HQ
        q = tl.load(src + hs * per_head + es, mask=mq, other=0.0)
        k = tl.load(src + hs * per_head + HQ + es, mask=mq, other=0.0)
        tl.store(dst + g0 * HQ + hs * HQ + es, q, mask=mq)
        tl.store(dst + (NQK + g0) * HQ + hs * HQ + es, k, mask=mq)

    if EVEN_V:
        ev = tl.arange(0, R * HV)[None, :]
        v = tl.load(src + hs * per_head + 2 * HQ + ev)
        z = tl.load(src + hs * per_head + 2 * HQ + R * HV + ev)
        tl.store(dst + 2 * NQK * HQ + g0 * R * HV + hs * (R * HV) + ev, v)
        tl.store(
            z_ptr + row * (NQK * R * HV) + g0 * R * HV + hs * (R * HV) + ev, z
        )
    else:
        ev = tl.arange(0, BLOCK_V)[None, :]
        mv = ev < R * HV
        v = tl.load(src + hs * per_head + 2 * HQ + ev, mask=mv, other=0.0)
        z = tl.load(
            src + hs * per_head + 2 * HQ + R * HV + ev, mask=mv, other=0.0
        )
        tl.store(
            dst + 2 * NQK * HQ + g0 * R * HV + hs * (R * HV) + ev, v, mask=mv
        )
        tl.store(
            z_ptr + row * (NQK * R * HV) + g0 * R * HV + hs * (R * HV) + ev,
            z,
            mask=mv,
        )

    offs_r = tl.arange(0, BLOCK_R)
    mask_r = offs_r < G * R
    hh = offs_r // R
    jj = offs_r - hh * R
    ba_src = ba_ptr + row * (NQK * 2 * R) + g0 * 2 * R
    bv = tl.load(ba_src + hh * (2 * R) + jj, mask=mask_r, other=0.0)
    av = tl.load(ba_src + hh * (2 * R) + R + jj, mask=mask_r, other=0.0)
    ba_dst = row * (NQK * R) + g0 * R
    tl.store(b_ptr + ba_dst + offs_r, bv, mask=mask_r)
    tl.store(a_ptr + ba_dst + offs_r, av, mask=mask_r)


def _heads_per_program(b, nqk):
    if b >= 2048:
        g = 16
    elif b >= 256:
        g = 8
    elif b >= 32:
        g = 4
    elif b >= 4:
        g = 2
    else:
        g = 1
    while g > 1 and nqk % g != 0:
        g //= 2
    return max(1, min(g, nqk))


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    b = mixed_qkvz.shape[0]
    r = num_heads_v // num_heads_qk

    if not mixed_qkvz.is_contiguous():
        mixed_qkvz = mixed_qkvz.contiguous()
    if not mixed_ba.is_contiguous():
        mixed_ba = mixed_ba.contiguous()

    dtype = mixed_qkvz.dtype
    dev = mixed_qkvz.device

    qkv_row = num_heads_qk * 2 * head_qk + num_heads_v * head_v
    n_qkv = b * qkv_row
    n_z = b * num_heads_v * head_v
    n_ba = b * num_heads_v

    if mixed_ba.dtype == dtype:
        off_z = _align(n_qkv)
        off_b = _align(off_z + n_z)
        off_a = _align(off_b + n_ba)
        flat = torch.empty(off_a + n_ba, dtype=dtype, device=dev)
        mixed_qkv = flat[:n_qkv].view(b, qkv_row)
        z = flat[off_z : off_z + n_z].view(b, num_heads_v, head_v)
        b_out = flat[off_b : off_b + n_ba].view(b, num_heads_v)
        a_out = flat[off_a:].view(b, num_heads_v)
    else:
        mixed_qkv = torch.empty((b, qkv_row), dtype=dtype, device=dev)
        z = torch.empty((b, num_heads_v, head_v), dtype=dtype, device=dev)
        b_out = torch.empty((b, num_heads_v), dtype=mixed_ba.dtype, device=dev)
        a_out = torch.empty((b, num_heads_v), dtype=mixed_ba.dtype, device=dev)

    if b > 0:
        g = _heads_per_program(b, num_heads_qk)
        even_q = (head_qk & (head_qk - 1)) == 0
        even_v = (r * head_v) & (r * head_v - 1) == 0

        grid = (b, num_heads_qk // g)
        _qkvzba_split_reshape_cat_kernel[grid](
            mixed_qkvz,
            mixed_ba,
            mixed_qkv,
            z,
            b_out,
            a_out,
            b,
            NQK=num_heads_qk,
            R=r,
            HQ=head_qk,
            HV=head_v,
            G=g,
            BLOCK_HQ=triton.next_power_of_2(head_qk),
            BLOCK_V=triton.next_power_of_2(r * head_v),
            BLOCK_R=triton.next_power_of_2(g * r),
            EVEN_Q=even_q,
            EVEN_V=even_v,
        )
    return mixed_qkv, z, b_out, a_out


__all__ = ["fused_qkvzba_split_reshape_cat"]
