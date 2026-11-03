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

"""Fused RMS normalization with per-batch scale and shift."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_scale_shift_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    sh_ptr,
    out_ptr,
    eps,
    x_rs,
    m_rs,
    o_rs,
    seq,
    D: tl.constexpr,
    BD: tl.constexpr,
    MASKED: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BD)
    if MASKED:
        cmask = cols < D

    row0 = pid * R
    same_b = (row0 // seq) == ((row0 + R - 1) // seq)
    mb = (row0 // seq).to(tl.int64) * m_rs
    if MASKED:
        s = tl.load(s_ptr + mb + cols, mask=cmask, other=0.0)
        sh = tl.load(sh_ptr + mb + cols, mask=cmask, other=0.0)
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0)
    else:
        s = tl.load(s_ptr + mb + cols)
        sh = tl.load(sh_ptr + mb + cols)
        w = tl.load(w_ptr + cols)

    # Modulation operands hoisted out of the row loop in storage dtype;
    # convert once to fp32 for the per-op-rounding chain below.
    t1 = (1.0 + s.to(tl.float32)).to(out_ptr.dtype.element_ty)
    t1f = t1.to(tl.float32)
    shf = sh.to(tl.float32)
    wf = w.to(tl.float32)

    for i in tl.static_range(R):
        row = row0 + i
        if same_b:
            s_row = s
            sh_row = sh
        else:
            # Boundary straddler: re-derive this row's modulation vector.
            mrb = (row // seq).to(tl.int64) * m_rs
            if MASKED:
                s_row = tl.load(s_ptr + mrb + cols, mask=cmask, other=0.0)
                sh_row = tl.load(sh_ptr + mrb + cols, mask=cmask, other=0.0)
            else:
                s_row = tl.load(s_ptr + mrb + cols)
                sh_row = tl.load(sh_ptr + mrb + cols)
        if same_b:
            t1_row = t1
            t1f_row = t1f
            shf_row = shf
        else:
            t1_row = (1.0 + s_row.to(tl.float32)).to(out_ptr.dtype.element_ty)
            t1f_row = t1_row.to(tl.float32)
            shf_row = sh_row.to(tl.float32)
        xb = row.to(tl.int64) * x_rs
        if MASKED:
            x = tl.load(x_ptr + xb + cols, mask=cmask, other=0.0).to(
                tl.float32
            )
        else:
            x = tl.load(x_ptr + xb + cols).to(tl.float32)

        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)

        # Norm: fp32 chain, one rounding to the storage dtype (RTNE), exactly
        # like the reference.
        normed = (x * rstd * wf).to(out_ptr.dtype.element_ty)
        t2 = (normed.to(tl.float32) * t1f_row).to(out_ptr.dtype.element_ty)
        y = (t2.to(tl.float32) + shf_row).to(out_ptr.dtype.element_ty)
        ob = row.to(tl.int64) * o_rs
        if MASKED:
            tl.store(out_ptr + ob + cols, y, mask=cmask)
        else:
            tl.store(out_ptr + ob + cols, y)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _rows_per_program(rows):
    """Pick R: divides `rows` (straddle-safe indexing handles seq %."""
    if rows == 2048 and rows % 2 == 0:
        return 2
    if rows >= 4096 and rows % 4 == 0:
        return 4
    if rows >= 512 and rows % 2 == 0:
        return 2
    return 1


def rmsnorm_scale_shift(x, weight, scale, shift, eps):
    """Apply RMSNorm and per-batch affine modulation."""
    dim = x.shape[-1]
    x2 = x.reshape(-1, dim)
    s2 = scale.reshape(-1, dim)
    sh2 = shift.reshape(-1, dim)
    seq = x2.shape[0] // s2.shape[0]

    if not x2.is_contiguous():
        x2 = x2.contiguous()
    if not s2.is_contiguous():
        s2 = s2.contiguous()
    if not sh2.is_contiguous():
        sh2 = sh2.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    out = torch.empty_like(x)
    o2 = out.reshape(-1, dim)

    rows = x2.shape[0]
    bd = _next_pow2(dim)
    masked = bd != dim

    rpp = _rows_per_program(rows)

    nw = 2 if (rpp >= 4 or (rpp == 2 and rows >= 2048)) else 4

    _rmsnorm_scale_shift_kernel[(rows // rpp,)](
        x2,
        weight,
        s2,
        sh2,
        o2,
        eps,
        x2.stride(0),
        s2.stride(0),
        o2.stride(0),
        seq,
        dim,
        bd,
        masked,
        rpp,
        num_warps=nw,
    )
    return out


__all__ = ["rmsnorm_scale_shift"]
