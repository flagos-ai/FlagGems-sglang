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

_ROWS_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=1),
    triton.Config({}, num_warps=16, num_stages=1),
]

# Two-pass fallback for rows wider than the fast path; any BLOCK_D is correct
# here because both passes loop over the full row.
_LOOP_CONFIGS = [
    triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_D": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
]

# Largest row width kept in registers in the single-pass path.
_MAX_BLOCK_D = 4096

_ROWS_PER_PROG = 4
_ROWS_PER_PROG_SMALL = 2
_SINGLE_ROW_MAX = 256
_SMALL_GRID_ROWS = 2048


@triton.autotune(
    configs=_ROWS_CONFIGS,
    key=["D", "BLOCK_D", "R"],
)
@triton.jit
def _rmsnorm_ss_batched_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    eps,
    S,
    D,
    stride_xb,
    stride_xs,
    R: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_R: tl.constexpr,
):
    """Single-pass kernel: each program handles R rows of ONE batch."""
    per_b = (S + R - 1) // R
    b = tl.program_id(0) // per_b
    s0 = (tl.program_id(0) % per_b) * R

    offs = tl.arange(0, BLOCK_D)
    el_ty = out_ptr.dtype.element_ty

    wf = tl.load(w_ptr + offs).to(tl.float32)
    b64 = b.to(tl.int64)
    sb = b64 * D
    sc = tl.load(scale_ptr + sb + offs).to(tl.float32)
    sh = tl.load(shift_ptr + sb + offs).to(tl.float32)
    one_plus = (1.0 + sc).to(el_ty)

    for i in range(R):
        if EVEN_R or (s0 + i < S):
            s = s0 + i
            xb = b64 * stride_xb + s.to(tl.int64) * stride_xs
            xf = tl.load(x_ptr + xb + offs).to(tl.float32)
            rstd = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / D + eps)
            normed = (xf * rstd * wf).to(el_ty)
            t = (normed.to(tl.float32) * one_plus.to(tl.float32)).to(el_ty)
            out = (t.to(tl.float32) + sh).to(el_ty)
            tl.store(out_ptr + xb + offs, out)


@triton.autotune(configs=_LOOP_CONFIGS, key=["D"])
@triton.jit
def _rmsnorm_ss_loop_kernel(
    x_ptr,
    w_ptr,
    scale_ptr,
    shift_ptr,
    out_ptr,
    eps,
    S,
    D,
    stride_xb,
    stride_xs,
    BLOCK_D: tl.constexpr,
):
    """Two-pass fallback for D wider than the register-resident fast path."""
    row = tl.program_id(0)
    b = row // S
    s = row % S

    xb = b.to(tl.int64) * stride_xb + s.to(tl.int64) * stride_xs
    sb = b.to(tl.int64) * D

    offs = tl.arange(0, BLOCK_D)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        idx = off + offs
        mask = idx < D
        xf = tl.load(x_ptr + xb + idx, mask=mask, other=0.0).to(tl.float32)
        acc += xf * xf
    rstd = tl.math.rsqrt(tl.sum(acc, axis=0) / D + eps)

    el_ty = out_ptr.dtype.element_ty
    for off in range(0, D, BLOCK_D):
        idx = off + offs
        mask = idx < D
        xf = tl.load(x_ptr + xb + idx, mask=mask, other=0.0).to(tl.float32)
        wf = tl.load(w_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        normed = (xf * rstd * wf).to(el_ty)

        sc = tl.load(scale_ptr + sb + idx, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(shift_ptr + sb + idx, mask=mask, other=0.0).to(tl.float32)
        one_plus = (1.0 + sc).to(el_ty)
        t = (normed.to(tl.float32) * one_plus.to(tl.float32)).to(el_ty)
        out = (t.to(tl.float32) + sh).to(el_ty)
        tl.store(out_ptr + xb + idx, out, mask=mask)


def rmsnorm_scale_shift(x, weight, scale, shift, eps):
    if x.stride(-1) != 1:
        x = x.contiguous()
    B, S, D = x.shape
    rows = B * S

    out = torch.empty_like(x)
    if rows == 0 or D == 0:
        return out

    if scale.stride(-1) != 1:
        scale = scale.contiguous()
    if shift.stride(-1) != 1:
        shift = shift.contiguous()
    scale2 = scale.view(B, D)
    shift2 = shift.view(B, D)
    weight = weight.contiguous()

    block_d = triton.next_power_of_2(D)
    if block_d > _MAX_BLOCK_D:
        _rmsnorm_ss_loop_kernel[(rows,)](
            x,
            weight,
            scale2,
            shift2,
            out,
            eps,
            S,
            D,
            x.stride(0),
            x.stride(1),
        )
    else:
        if rows <= _SINGLE_ROW_MAX:
            r = 1
        elif rows <= _SMALL_GRID_ROWS:
            r = _ROWS_PER_PROG_SMALL
        else:
            r = _ROWS_PER_PROG
        grid = (B * ((S + r - 1) // r),)
        _rmsnorm_ss_batched_kernel[grid](
            x,
            weight,
            scale2,
            shift2,
            out,
            eps,
            S,
            D,
            x.stride(0),
            x.stride(1),
            R=r,
            BLOCK_D=block_d,
            EVEN_R=(r == 1) or (S % r == 0),
        )
    return out


__all__ = ["rmsnorm_scale_shift"]
