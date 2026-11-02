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
    sc_ptr,
    sh_ptr,
    out_ptr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    EPS: tl.constexpr,
    MASK_S: tl.constexpr,
    MASK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)

    rows = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    cols = tl.arange(0, BLOCK_D)

    base = pid_b * S * D + rows[:, None] * D + cols[None, :]
    # Broadcastable (1, BLOCK_D) row; one offset serves the whole batch column.
    mod = pid_b * D + cols[None, :]

    # All predicates are compile-time constant: the common production shapes
    # (D a power of two, S a multiple of BLOCK_S) take the mask-free path.
    if MASK_S == 0 and MASK_D == 0:
        x = tl.load(x_ptr + base).to(tl.float32)
    elif MASK_D == 0:
        m = (rows < S)[:, None]
        x = tl.load(x_ptr + base, mask=m, other=0.0).to(tl.float32)
    else:
        m = (rows < S)[:, None] & (cols < D)[None, :]
        x = tl.load(x_ptr + base, mask=m, other=0.0).to(tl.float32)

    # fp32 reduction — the one place precision is contractual.
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)

    if MASK_D == 0:
        w = tl.load(w_ptr + cols).to(tl.float32)
        sc = tl.load(sc_ptr + mod)
        sh = tl.load(sh_ptr + mod)
    else:
        cm = cols < D
        w = tl.load(w_ptr + cols, mask=cm, other=0.0).to(tl.float32)
        sc = tl.load(sc_ptr + mod, mask=cm, other=0.0)
        sh = tl.load(sh_ptr + mod, mask=cm, other=0.0)

    et = x_ptr.dtype.element_ty
    normed = (x * rstd[:, None] * w[None, :]).to(et)
    one_sc = (1.0 + sc.to(tl.float32)).to(et)
    y = normed * one_sc + sh

    if MASK_S == 0 and MASK_D == 0:
        tl.store(out_ptr + base, y)
    elif MASK_D == 0:
        tl.store(out_ptr + base, y, mask=m)
    else:
        tl.store(out_ptr + base, y, mask=m)


def _next_pow2(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _launch_config(batch, seq, d):
    """(BLOCK_S, num_warps, num_stages) measured on the target."""
    block_d = _next_pow2(d)
    if block_d >= 4096:
        return (4, 4, 1)
    if seq < 150:
        return (4, 4, 1)
    if seq < 2048:
        return (8, 2, 1)
    return (8, 2, 2)


def rmsnorm_scale_shift(x, weight, scale, shift, eps):
    B, S, D = x.shape

    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if not shift.is_contiguous():
        shift = shift.contiguous()

    out = torch.empty_like(x)
    if B == 0 or S == 0 or D == 0:
        return out

    block_d = _next_pow2(D)
    block_s, num_warps, num_stages = _launch_config(B, S, D)

    grid_s = (S + block_s - 1) // block_s
    mask_s = 1 if S % block_s else 0
    mask_d = 0 if D == block_d else 1
    _rmsnorm_scale_shift_kernel[(grid_s, B)](
        x,
        weight,
        scale,
        shift,
        out,
        S,
        D,
        block_d,
        block_s,
        eps,
        mask_s,
        mask_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


__all__ = ["rmsnorm_scale_shift"]
