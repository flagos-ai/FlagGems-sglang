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
def _qkvzba_split_kernel(
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
    HB: tl.constexpr,
    QKP2: tl.constexpr,
    VP2: tl.constexpr,
    RP2: tl.constexpr,
    MASK_QK: tl.constexpr,
    MASK_V: tl.constexpr,
    MASK_R: tl.constexpr,
    WIDE: tl.constexpr,
):
    SEG: tl.constexpr = 2 * HQ + 2 * R * HV
    NV: tl.constexpr = NQK * R
    RV: tl.constexpr = R * HV

    pid_h = tl.program_id(0)
    row = tl.program_id(1)
    hb0 = pid_h * HB

    if row < B:
        if WIDE:
            in_base = row.to(tl.int64) * (NQK * SEG) + hb0.to(tl.int64) * SEG
            qkv_row = row.to(tl.int64) * (NQK * 2 * HQ + NV * HV)
            z_row = row.to(tl.int64) * (NV * HV)
            ba_row = row.to(tl.int64) * (NQK * 2 * R)
            bao_row = row.to(tl.int64) * NV
        else:
            in_base = row * (NQK * SEG) + hb0 * SEG
            qkv_row = row * (NQK * 2 * HQ + NV * HV)
            z_row = row * (NV * HV)
            ba_row = row * (NQK * 2 * R)
            bao_row = row * NV

        offs_h = tl.arange(0, HB)[:, None]
        offs_qk = tl.arange(0, QKP2)[None, :]
        offs_v = tl.arange(0, VP2)[None, :]
        offs_r = tl.arange(0, RP2)[None, :]

        KSEL: tl.constexpr = (NQK - 1) * HQ
        rm = (offs_r < R) if MASK_R else None
        vmask = (offs_v < RV) if MASK_V else None
        qk_mask = (offs_qk < 2 * HQ) if MASK_QK else None

        qk = tl.load(qkvz_ptr + in_base + offs_h * SEG + offs_qk, mask=qk_mask)
        v = tl.load(
            qkvz_ptr + in_base + offs_h * SEG + (2 * HQ + offs_v), mask=vmask
        )
        z = tl.load(
            qkvz_ptr + in_base + offs_h * SEG + (2 * HQ + RV + offs_v),
            mask=vmask,
        )
        bb = tl.load(
            ba_ptr + ba_row + (hb0 + offs_h) * (2 * R) + offs_r, mask=rm
        )
        aa = tl.load(
            ba_ptr + ba_row + (hb0 + offs_h) * (2 * R) + R + offs_r, mask=rm
        )

        tl.store(
            qkv_ptr
            + qkv_row
            + hb0 * HQ
            + offs_h * HQ
            + offs_qk
            + KSEL * (offs_qk >= HQ),
            qk,
            mask=qk_mask,
        )
        tl.store(
            qkv_ptr + qkv_row + 2 * NQK * HQ + hb0 * RV + offs_h * RV + offs_v,
            v,
            mask=vmask,
        )
        tl.store(
            z_ptr + z_row + hb0 * RV + offs_h * RV + offs_v, z, mask=vmask
        )
        dst = bao_row + hb0 * R + offs_h * R + offs_r
        tl.store(b_ptr + dst, bb, mask=rm)
        tl.store(a_ptr + dst, aa, mask=rm)


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    b = mixed_qkvz.shape[0]
    num_heads_qk = int(num_heads_qk)
    num_heads_v = int(num_heads_v)
    head_qk = int(head_qk)
    head_v = int(head_v)
    r = num_heads_v // num_heads_qk

    qkv_dim = num_heads_qk * 2 * head_qk + num_heads_v * head_v
    mixed_qkv = torch.empty(
        (b, qkv_dim), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
    )
    z = torch.empty(
        (b, num_heads_v, head_v),
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b_out = torch.empty(
        (b, num_heads_v), dtype=mixed_ba.dtype, device=mixed_ba.device
    )
    a_out = torch.empty(
        (b, num_heads_v), dtype=mixed_ba.dtype, device=mixed_ba.device
    )

    if b == 0 or num_heads_qk == 0:
        return mixed_qkv, z, b_out, a_out

    wide = (
        mixed_qkvz.numel() >= (1 << 31)
        or mixed_qkv.numel() >= (1 << 31)
        or z.numel() >= (1 << 31)
        or mixed_ba.numel() >= (1 << 31)
        or b_out.numel() >= (1 << 31)
    )

    if b <= 4:
        HB = 1
        for cand in (4, 2):
            if num_heads_qk % cand == 0:
                HB = cand
                break
        warps = 8 if HB > 1 else 2
    else:
        HB, warps = 1, 2

    _qkvzba_split_kernel[(num_heads_qk // HB, b)](
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
        HB=HB,
        QKP2=triton.next_power_of_2(2 * head_qk),
        VP2=triton.next_power_of_2(r * head_v),
        RP2=triton.next_power_of_2(r),
        MASK_QK=(2 * head_qk) & (2 * head_qk - 1) != 0,
        MASK_V=(r * head_v) & (r * head_v - 1) != 0,
        MASK_R=r & (r - 1) != 0,
        WIDE=wide,
        num_warps=warps,
    )
    return mixed_qkv, z, b_out, a_out


__all__ = ["fused_qkvzba_split_reshape_cat"]
