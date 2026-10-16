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

_WIDE_MAX_H = 8192


@triton.jit
def _rmsnorm_hf_wide_kernel(
    X,
    W,
    Y,
    M,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    COL_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)

    if COL_MASK:
        cmask = cols < H
        mask = (rows < M)[:, None] & cmask[None, :]
        w = tl.load(W + cols, mask=cmask, other=0.0)
    else:
        mask = (rows < M)[:, None]
        w = tl.load(W + cols)

    offs = rows[:, None] * H + cols[None, :]
    xb = tl.load(X + offs, mask=mask, other=0.0)

    xf = xb.to(tl.float32)
    ss = tl.sum(xf * xf, axis=1)
    inv = tl.rsqrt(ss / H + EPS)

    out = (xf * inv[:, None] * w.to(tl.float32)[None, :]).to(
        Y.dtype.element_ty
    )

    tl.store(Y + offs, out, mask=mask)


@triton.jit
def _rmsnorm_hf_chunk_kernel(
    X,
    W,
    Y,
    stride_xm,
    stride_ym,
    M,
    H: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    row_off = pid * stride_xm
    out_off = pid * stride_ym

    ss = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        cmask = (h0 + cols) < H
        x = tl.load(X + row_off + h0 + cols, mask=cmask, other=0.0).to(
            tl.float32
        )
        ss += x * x

    inv = tl.rsqrt(tl.sum(ss) / H + EPS)

    for h0 in range(0, H, BLOCK_H):
        cmask = (h0 + cols) < H
        x = tl.load(X + row_off + h0 + cols, mask=cmask, other=0.0).to(
            tl.float32
        )
        w = tl.load(W + h0 + cols, mask=cmask, other=0.0).to(tl.float32)
        out = (x * inv * w).to(Y.dtype.element_ty)
        tl.store(Y + out_off + h0 + cols, out, mask=cmask)


def rmsnorm_hf(input, weight, eps):
    x = input if input.is_contiguous() else input.contiguous()
    w = weight if weight.is_contiguous() else weight.contiguous()

    H = x.shape[-1]
    M = x.numel() // H if H else 0
    out = torch.empty_like(x)

    if M == 0:
        return out

    xv = x.reshape(M, H) if x.dim() != 2 else x
    ov = out.reshape(M, H) if out.dim() != 2 else out

    if H <= _WIDE_MAX_H:
        block_h = 1 << (H - 1).bit_length()
        col_mask = block_h != H
        if block_h <= 128:
            if M >= 2048:
                block_m, num_warps = 8, 2
            elif M >= 256:
                block_m, num_warps = 4, 1
            elif M >= 32:
                block_m, num_warps = 4, 4
            else:
                block_m, num_warps = 2, 2
        else:
            block_m = 1
            num_warps = 8 if M <= 64 else 4
        grid = ((M + block_m - 1) // block_m,)
        _rmsnorm_hf_wide_kernel[grid](
            xv,
            w,
            ov,
            M,
            H=H,
            EPS=eps,
            BLOCK_H=block_h,
            BLOCK_M=block_m,
            COL_MASK=col_mask,
            num_warps=num_warps,
        )
    else:
        _rmsnorm_hf_chunk_kernel[(M,)](
            xv,
            w,
            ov,
            xv.stride(0),
            ov.stride(0),
            M,
            H=H,
            EPS=eps,
            BLOCK_H=2048,
            num_warps=8,
        )
    return out


__all__ = ["rmsnorm_hf"]
