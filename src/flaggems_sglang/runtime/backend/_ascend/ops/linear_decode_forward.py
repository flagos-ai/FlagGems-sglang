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


@triton.jit
def _ldf_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    cache_ptr,
    slope_ptr,
    slot_ptr,
    out_ptr,
    H,
    D_LEN,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    slot = tl.load(slot_ptr + b)
    if slot < 0:
        return

    decay = tl.exp(-tl.load(slope_ptr + h).to(tl.float32))

    kv_off = (b * H + h) * D_LEN
    k_base = k_ptr + kv_off
    v_base = v_ptr + kv_off
    q_base = q_ptr + kv_off
    c_base = cache_ptr + (slot.to(tl.int64) * H + h) * D_LEN * D_LEN
    o_base = out_ptr + b * H * D_LEN + h * D_LEN

    dv = tl.arange(0, DV)
    if EVEN:
        vv = tl.load(v_base + dv).to(tl.float32)
    else:
        vm = dv < D_LEN
        vv = tl.load(v_base + dv, mask=vm, other=0.0).to(tl.float32)

    acc = tl.zeros([DV], dtype=tl.float32)

    for ki in range(0, DK, BK):
        kr = ki + tl.arange(0, BK)
        if EVEN:
            kk = tl.load(k_base + kr).to(tl.float32)
            qq = tl.load(q_base + kr).to(tl.float32)
            st = tl.load(c_base + kr[:, None] * DV + dv[None, :]).to(
                tl.float32
            )
            st = kk[:, None] * vv[None, :] + decay * st
            tl.store(
                c_base + kr[:, None] * DV + dv[None, :],
                st.to(cache_ptr.dtype.element_ty),
            )
        else:
            km = kr < D_LEN
            kk = tl.load(k_base + kr, mask=km, other=0.0).to(tl.float32)
            qq = tl.load(q_base + kr, mask=km, other=0.0).to(tl.float32)
            cm = km[:, None] & vm[None, :]
            st = tl.load(
                c_base + kr[:, None] * D_LEN + dv[None, :], mask=cm, other=0.0
            ).to(tl.float32)
            st = kk[:, None] * vv[None, :] + decay * st
            tl.store(
                c_base + kr[:, None] * D_LEN + dv[None, :],
                st.to(cache_ptr.dtype.element_ty),
                mask=cm,
            )
        acc += tl.sum(qq[:, None] * st, axis=0)

    if EVEN:
        tl.store(o_base + dv, acc.to(out_ptr.dtype.element_ty))
    else:
        tl.store(o_base + dv, acc.to(out_ptr.dtype.element_ty), mask=vm)


def linear_decode_forward(
    q, k, v, kv_caches, slope_rate, slot_idx, BLOCK_SIZE
):
    b, h, _, d = q.shape
    if b == 0 or h == 0:
        return (
            torch.empty((b, h * d), dtype=q.dtype, device=q.device),
            kv_caches,
        )

    DK = 1
    while DK < d:
        DK *= 2
    DV = DK
    even = DK == d

    bk = 32
    while bk < min(DK, 128):
        bk *= 2
    while bk < DK:
        bk *= 2

    out = torch.empty((b, h * d), dtype=q.dtype, device=q.device)
    _ldf_kernel[(b * h,)](
        q,
        k,
        v,
        kv_caches,
        slope_rate,
        slot_idx,
        out,
        h,
        d,
        DK=DK,
        DV=DV,
        BK=bk,
        EVEN=even,
        num_warps=4,
    )
    return out, kv_caches


__all__ = ["linear_decode_forward"]
