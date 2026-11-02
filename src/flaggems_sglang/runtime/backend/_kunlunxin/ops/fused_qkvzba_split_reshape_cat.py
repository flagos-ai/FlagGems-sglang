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
def _qkvzba_split_qkv_kernel(
    src_ptr,
    dst_ptr,
    qkvz_row: tl.constexpr,
    HQ: tl.constexpr,
    HV: tl.constexpr,
    NQK: tl.constexpr,
    R: tl.constexpr,
    L_QKV: tl.constexpr,
    BLK: tl.constexpr,
    P: tl.constexpr,
):
    # src_ptr/dst_ptr are views of the bf16 tensors packed P elements per word
    # (P==1 => plain bf16). qkvz_row and all strides below are in packed words.
    seg = tl.program_id(1)
    row = tl.program_id(2)
    offs = tl.program_id(0) * BLK + tl.arange(0, BLK)
    HQW: tl.constexpr = HQ // P
    VW: tl.constexpr = R * HV // P
    STRIDEW: tl.constexpr = (2 * HQ + 2 * R * HV) // P
    LQKVW: tl.constexpr = L_QKV // P
    if seg == 0:
        m = offs < NQK * HQW
        head = offs // HQW
        o = offs - head * HQW
        val = tl.load(
            src_ptr + row * qkvz_row + head * STRIDEW + o, mask=m, other=0
        )
        tl.store(dst_ptr + row * LQKVW + offs, val, mask=m)
    if seg == 1:
        m = offs < NQK * HQW
        head = offs // HQW
        o = offs - head * HQW
        val = tl.load(
            src_ptr + row * qkvz_row + head * STRIDEW + HQW + o,
            mask=m,
            other=0,
        )
        tl.store(dst_ptr + row * LQKVW + NQK * HQW + offs, val, mask=m)
    if seg == 2:
        m = offs < NQK * VW
        head = offs // VW
        o = offs - head * VW
        val = tl.load(
            src_ptr + row * qkvz_row + head * STRIDEW + 2 * HQW + o,
            mask=m,
            other=0,
        )
        tl.store(dst_ptr + row * LQKVW + 2 * NQK * HQW + offs, val, mask=m)


@triton.jit
def _qkvzba_split_z_kernel(
    src_ptr,
    dst_ptr,
    qkvz_row: tl.constexpr,
    HQ: tl.constexpr,
    HV: tl.constexpr,
    NQK: tl.constexpr,
    R: tl.constexpr,
    L_Z: tl.constexpr,
    BLK: tl.constexpr,
    P: tl.constexpr,
):
    row = tl.program_id(1)
    offs = tl.program_id(0) * BLK + tl.arange(0, BLK)
    HQW: tl.constexpr = HQ // P
    VW: tl.constexpr = R * HV // P
    STRIDEW: tl.constexpr = (2 * HQ + 2 * R * HV) // P
    LZW: tl.constexpr = L_Z // P
    m = offs < NQK * VW
    head = offs // VW
    o = offs - head * VW
    val = tl.load(
        src_ptr + row * qkvz_row + head * STRIDEW + 2 * HQW + VW + o,
        mask=m,
        other=0,
    )
    tl.store(dst_ptr + row * LZW + offs, val, mask=m)


@triton.jit
def _qkvzba_split_ba_kernel(
    src_ptr,
    b_ptr,
    a_ptr,
    n_rows,
    NQK: tl.constexpr,
    R: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    u = pid * BLK + tl.arange(0, BLK)
    m = u < n_rows
    row = u // (NQK * R)
    idx = u - row * (NQK * R)
    head = idx // R
    w = idx - head * R
    bsrc = row * (NQK * 2 * R) + head * (2 * R) + w
    asrc = row * (NQK * 2 * R) + head * (2 * R) + R + w
    bv = tl.load(src_ptr + bsrc, mask=m, other=0.0)
    av = tl.load(src_ptr + asrc, mask=m, other=0.0)
    tl.store(b_ptr + u, bv, mask=m)
    tl.store(a_ptr + u, av, mask=m)


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    B = mixed_qkvz.shape[0]
    r = num_heads_v // num_heads_qk
    L_qkv = num_heads_qk * (2 * head_qk + r * head_v)
    L_z = num_heads_v * head_v
    mixed_qkv = torch.empty(
        (B, L_qkv), device=mixed_qkvz.device, dtype=mixed_qkvz.dtype
    )
    z = torch.empty(
        (B, num_heads_v, head_v),
        device=mixed_qkvz.device,
        dtype=mixed_qkvz.dtype,
    )
    b_out = torch.empty(
        (B, num_heads_v), device=mixed_qkvz.device, dtype=mixed_ba.dtype
    )
    a_out = torch.empty(
        (B, num_heads_v), device=mixed_ba.device, dtype=mixed_ba.dtype
    )
    if B == 0:
        return mixed_qkv, z, b_out, a_out

    L_src = mixed_qkvz.shape[1]
    row_stride = L_src // 4  # pack width in words
    # Pack 4 bf16 elements per int64 word to widen the affine per-head runs.
    # Requires every packed extent to be a multiple of 4.
    wide = (
        B >= 256
        and L_src % 4 == 0
        and L_qkv % 4 == 0
        and L_z % 4 == 0
        and head_qk % 4 == 0
        and (r * head_v) % 4 == 0
        and (2 * head_qk + 2 * r * head_v) % 4 == 0
    )
    if wide:
        P = 4
        src = mixed_qkvz.view(torch.int64)
        dqv = mixed_qkv.view(torch.int64)
        zv = z.reshape(B, num_heads_v, head_v).view(torch.int64)
        blk = 1024
    else:
        P = 1
        src = mixed_qkvz
        dqv = mixed_qkv
        zv = z.view(B, num_heads_v * head_v)
        blk = 2048
        row_stride = L_src

    seg_max = max(
        num_heads_qk * (head_qk // P), num_heads_qk * (r * head_v) // P
    )
    _qkvzba_split_qkv_kernel[(triton.cdiv(seg_max, blk), 3, B)](
        src,
        dqv,
        qkvz_row=row_stride,
        HQ=head_qk,
        HV=head_v,
        NQK=num_heads_qk,
        R=r,
        L_QKV=L_qkv,
        BLK=blk,
        P=P,
        num_warps=4,
        num_stages=1,
    )
    zblk = min(
        blk, triton.next_power_of_2(max(num_heads_qk * (r * head_v) // P, 1))
    )
    _qkvzba_split_z_kernel[(triton.cdiv(L_z // P, zblk), B)](
        src,
        zv,
        qkvz_row=row_stride,
        HQ=head_qk,
        HV=head_v,
        NQK=num_heads_qk,
        R=r,
        L_Z=L_z,
        BLK=zblk,
        P=P,
        num_warps=4,
        num_stages=1,
    )
    ba_total = B * num_heads_v
    ba_blk = min(512, triton.next_power_of_2(max(ba_total, 1)))
    _qkvzba_split_ba_kernel[(triton.cdiv(ba_total, ba_blk),)](
        mixed_ba,
        b_out,
        a_out,
        ba_total,
        NQK=num_heads_qk,
        R=r,
        BLK=ba_blk,
        num_warps=2,
        num_stages=1,
    )
    return mixed_qkv, z, b_out, a_out


__all__ = ["fused_qkvzba_split_reshape_cat"]
