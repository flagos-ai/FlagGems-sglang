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
    nqk,
    QK: tl.constexpr,  # head_qk
    R: tl.constexpr,  # num_heads_v // num_heads_qk
    HV: tl.constexpr,  # head_v
    S: tl.constexpr,  # 2*QK + 2*R*HV, segment width in mixed_qkvz
    RP2: tl.constexpr,  # next_power_of_2(R)
    BS: tl.constexpr,  # segments (rows) per tile
    BC: tl.constexpr,  # columns per tile
):
    pid = tl.program_id(0)
    row = tl.program_id(1)
    NCB: tl.constexpr = (S + BC - 1) // BC
    ps = pid // NCB
    pc = pid % NCB

    seg = ps * BS + tl.arange(0, BS)[:, None]
    cc = pc * BC + tl.arange(0, BC)[None, :]
    smask = seg < nqk
    mask = smask & (cc < S)

    val = tl.load(
        qkvz_ptr + row * (nqk * S) + seg * S + cc, mask=mask, other=0
    )

    is_q = cc < QK
    is_k = (cc >= QK) & (cc < 2 * QK)
    v_col = cc - 2 * QK
    is_v = (v_col >= 0) & (v_col < R * HV)
    z_col = v_col - R * HV
    is_z = (z_col >= 0) & (z_col < R * HV)

    qkv_off = tl.where(
        is_q,
        seg * QK + cc,
        tl.where(
            is_k,
            nqk * QK + seg * QK + (cc - QK),
            2 * nqk * QK + seg * R * HV + v_col,
        ),
    )
    tl.store(
        qkv_ptr + row * (nqk * (2 * QK + R * HV)) + qkv_off,
        val,
        mask=mask & (is_q | is_k | is_v),
    )
    tl.store(
        z_ptr + row * (nqk * R * HV) + seg * R * HV + z_col,
        val,
        mask=mask & is_z,
    )

    # First column block of each segment tile also splits the b/a pair.
    if pc == 0:
        j = tl.arange(0, RP2)[None, :]
        jm = (j < R) & smask
        src = ba_ptr + row * (nqk * 2 * R) + seg * 2 * R + j
        tl.store(
            b_ptr + row * (nqk * R) + seg * R + j,
            tl.load(src, mask=jm, other=0),
            mask=jm,
        )
        tl.store(
            a_ptr + row * (nqk * R) + seg * R + j,
            tl.load(src + R, mask=jm, other=0),
            mask=jm,
        )


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    bsz = mixed_qkvz.shape[0]
    r = num_heads_v // num_heads_qk

    qkvz = mixed_qkvz.contiguous()
    ba = mixed_ba.contiguous()
    dev = qkvz.device
    dt = qkvz.dtype

    seg = 2 * head_qk + 2 * r * head_v
    qkv_w = num_heads_qk * (2 * head_qk + r * head_v)

    mixed_qkv = torch.empty((bsz, qkv_w), dtype=dt, device=dev)
    z = torch.empty((bsz, num_heads_v, head_v), dtype=dt, device=dev)
    b_out = torch.empty((bsz, num_heads_v), dtype=ba.dtype, device=dev)
    a_out = torch.empty((bsz, num_heads_v), dtype=ba.dtype, device=dev)
    if bsz == 0:
        return mixed_qkv, z, b_out, a_out

    BS, BC = 16, 256
    grid = (triton.cdiv(num_heads_qk, BS) * triton.cdiv(seg, BC), bsz)
    _qkvzba_split_kernel[grid](
        qkvz,
        ba,
        mixed_qkv,
        z,
        b_out,
        a_out,
        num_heads_qk,
        QK=head_qk,
        R=r,
        HV=head_v,
        S=seg,
        RP2=triton.next_power_of_2(r),
        BS=BS,
        BC=BC,
        num_warps=8,
    )
    return mixed_qkv, z, b_out, a_out


__all__ = ["fused_qkvzba_split_reshape_cat"]
