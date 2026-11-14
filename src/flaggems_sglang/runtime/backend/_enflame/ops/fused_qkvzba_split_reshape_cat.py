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
    QKP2: tl.constexpr,
    VZP2: tl.constexpr,
    BAP2: tl.constexpr,
    MASK_QK: tl.constexpr,
    MASK_VZ: tl.constexpr,
    MASK_BA: tl.constexpr,
    WIDE: tl.constexpr,
):
    SEG: tl.constexpr = 2 * HQ + 2 * R * HV
    RV: tl.constexpr = R * HV
    QKVW: tl.constexpr = NQK * 2 * HQ + NQK * RV

    head = tl.program_id(1)
    row = tl.program_id(0)
    n_row_progs = tl.num_programs(0)

    offs_qk = tl.arange(0, QKP2)
    offs_vz = tl.arange(0, VZP2)
    offs_ba = tl.arange(0, BAP2)
    qk_lmask = (offs_qk < 2 * HQ) if MASK_QK else None
    vz_lmask = (offs_vz < 2 * RV) if MASK_VZ else None
    ba_lmask = (offs_ba < 2 * R) if MASK_BA else None

    q_smask = offs_qk < HQ
    k_smask = (offs_qk >= HQ) & (offs_qk < 2 * HQ)
    v_smask = offs_vz < RV
    z_smask = (offs_vz >= RV) & (offs_vz < 2 * RV)
    b_smask = offs_ba < R
    a_smask = (offs_ba >= R) & (offs_ba < 2 * R)

    for r in range(row, B, n_row_progs):
        if WIDE:
            in_base = r.to(tl.int64) * (NQK * SEG) + head.to(tl.int64) * SEG
            qkv_row = r.to(tl.int64) * QKVW
            z_row = r.to(tl.int64) * (NQK * RV)
            ba_row = r.to(tl.int64) * (NQK * 2 * R)
            bao_row = r.to(tl.int64) * (NQK * R)
            q_base = head.to(tl.int64) * HQ
            k_base = q_base + NQK * HQ
            v_base = head.to(tl.int64) * RV + 2 * NQK * HQ
            z_base = head.to(tl.int64) * RV
            ba_base = head.to(tl.int64) * (2 * R)
            bao_base = head.to(tl.int64) * R
        else:
            in_base = r * (NQK * SEG) + head * SEG
            qkv_row = r * QKVW
            z_row = r * (NQK * RV)
            ba_row = r * (NQK * 2 * R)
            bao_row = r * NQK * R
            q_base = head * HQ
            k_base = q_base + NQK * HQ
            v_base = head * RV + 2 * NQK * HQ
            z_base = head * RV
            ba_base = head * (2 * R)
            bao_base = head * R

        qk = tl.load(qkvz_ptr + in_base + offs_qk, mask=qk_lmask)
        vz = tl.load(qkvz_ptr + in_base + 2 * HQ + offs_vz, mask=vz_lmask)
        ba = tl.load(ba_ptr + ba_row + ba_base + offs_ba, mask=ba_lmask)

        tl.store(qkv_ptr + qkv_row + q_base + offs_qk, qk, mask=q_smask)
        tl.store(qkv_ptr + qkv_row + k_base + (offs_qk - HQ), qk, mask=k_smask)
        tl.store(qkv_ptr + qkv_row + v_base + offs_vz, vz, mask=v_smask)
        tl.store(z_ptr + z_row + z_base + (offs_vz - RV), vz, mask=z_smask)
        tl.store(b_ptr + bao_row + bao_base + offs_ba, ba, mask=b_smask)
        tl.store(a_ptr + bao_row + bao_base + (offs_ba - R), ba, mask=a_smask)


@triton.jit
def _qkvzba_split_3d_kernel(
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
    H: tl.constexpr,
    U: tl.constexpr,
):
    # Row-batch 3D form: one program covers U rows x H heads. Loads use 3D
    # affine addressing (row, head, col) into the head-interleaved input;
    # stores use 3D addressing whose inner (head, col) plane is contiguous
    # in the packed outputs (head stride == field width), so each store
    # row lands as one wide contiguous segment. All addresses stay pure
    # affine iota -- no div/mod, no gathers.
    RV: tl.constexpr = R * HV
    QKVW: tl.constexpr = NQK * 2 * HQ + NQK * RV
    SEG: tl.constexpr = 2 * HQ + 2 * RV

    rb = tl.program_id(0)
    hg = tl.program_id(1)
    rows = rb * U + tl.arange(0, U)
    rmask = rows < B
    heads = hg * H + tl.arange(0, H)
    offc_q = tl.arange(0, HQ)
    offc_v = tl.arange(0, RV)
    offc_b = tl.arange(0, R)

    in_base = rows[:, None, None] * (NQK * SEG) + heads[None, :, None] * SEG
    m3 = rmask[:, None, None]
    q = tl.load(qkvz_ptr + in_base + offc_q[None, None, :], mask=m3)
    k = tl.load(qkvz_ptr + in_base + HQ + offc_q[None, None, :], mask=m3)
    v = tl.load(qkvz_ptr + in_base + 2 * HQ + offc_v[None, None, :], mask=m3)
    z = tl.load(qkvz_ptr + in_base + 2 * HQ + RV + offc_v[None, None, :], mask=m3)
    ba_row = rows[:, None, None] * (NQK * 2 * R) + heads[None, :, None] * (2 * R)
    bv = tl.load(ba_ptr + ba_row + offc_b[None, None, :], mask=m3)
    av = tl.load(ba_ptr + ba_row + R + offc_b[None, None, :], mask=m3)

    qkv_row = rows[:, None, None] * QKVW
    z_row = rows[:, None, None] * (NQK * RV)
    bao_row = rows[:, None, None] * (NQK * R)
    tl.store(
        qkv_ptr + qkv_row + heads[None, :, None] * HQ + offc_q[None, None, :], q, mask=m3
    )
    tl.store(
        qkv_ptr
        + qkv_row
        + NQK * HQ
        + heads[None, :, None] * HQ
        + offc_q[None, None, :],
        k,
        mask=m3,
    )
    tl.store(
        qkv_ptr
        + qkv_row
        + 2 * NQK * HQ
        + heads[None, :, None] * RV
        + offc_v[None, None, :],
        v,
        mask=m3,
    )
    tl.store(
        z_ptr + z_row + heads[None, :, None] * RV + offc_v[None, None, :], z, mask=m3
    )
    tl.store(
        b_ptr + bao_row + heads[None, :, None] * R + offc_b[None, None, :], bv, mask=m3
    )
    tl.store(
        a_ptr + bao_row + heads[None, :, None] * R + offc_b[None, None, :], av, mask=m3
    )


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

    # Row-batch 3D fast path. Gated on power-of-two field widths (tl.arange
    # bounds), a clean head-group split, and int32-safe sizes; everything
    # else (including WIDE) keeps the 1D persistent kernel.
    use_3d = (
        (not wide)
        and b >= 16
        and head_qk & (head_qk - 1) == 0
        and r & (r - 1) == 0
        and (r * head_v) & (r * head_v - 1) == 0
        and num_heads_qk % min(num_heads_qk, 16) == 0
    )
    if use_3d:
        h = min(num_heads_qk, 16)
        while h > 1 and (num_heads_qk % h != 0 or h & (h - 1) != 0):
            h //= 2
        elem = mixed_qkvz.element_size()
        load_bytes = (2 * head_qk + 2 * r * head_v + 2 * r) * elem
        uh_cap = 1
        while uh_cap * 2 * load_bytes <= (1 << 19) and uh_cap < 256:
            uh_cap *= 2
        if h >= 2 and uh_cap >= h:
            u = max(
                1,
                min(
                    uh_cap // h,
                    triton.next_power_of_2(max(1, b // 16)),
                    32,
                ),
            )
            _qkvzba_split_3d_kernel[(triton.cdiv(b, u), num_heads_qk // h)](
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
                H=h,
                U=u,
                num_warps=1,
            )
            return mixed_qkv, z, b_out, a_out

    _qkvzba_split_kernel[(min(b, 12), num_heads_qk)](
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
        QKP2=triton.next_power_of_2(2 * head_qk),
        VZP2=triton.next_power_of_2(2 * r * head_v),
        BAP2=triton.next_power_of_2(2 * r),
        MASK_QK=(2 * head_qk) & (2 * head_qk - 1) != 0,
        MASK_VZ=(2 * r * head_v) & (2 * r * head_v - 1) != 0,
        MASK_BA=(2 * r) & (2 * r - 1) != 0,
        WIDE=wide,
        num_warps=2,
    )
    return mixed_qkv, z, b_out, a_out


__all__ = ["fused_qkvzba_split_reshape_cat"]
