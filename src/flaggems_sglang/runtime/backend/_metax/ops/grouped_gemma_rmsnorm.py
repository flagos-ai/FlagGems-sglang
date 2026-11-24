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
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    EPS: tl.constexpr,
    GS: tl.constexpr,
    G: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // G
    pid_g = pid - pid_m * G
    k = tl.arange(0, BLOCK_G)
    col = pid_g * GS + k
    if BLOCK_G == GS:
        offs = pid_m * (G * GS) + col
        xf = tl.load(x_ptr + offs, eviction_policy="evict_first").to(
            tl.float32
        )
        ss = tl.sum(xf * xf, axis=0)
        rstd = tl.rsqrt(ss * (1.0 / GS) + EPS)
        w = tl.load(w_ptr + col).to(tl.float32)
        t = xf * rstd
        tl.store(
            out_ptr + offs,
            (t * w + t).to(out_ptr.dtype.element_ty),
            eviction_policy="evict_first",
        )
    else:
        m = k < GS
        offs = pid_m * (G * GS) + col
        xf = tl.load(
            x_ptr + offs, mask=m, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        ss = tl.sum(xf * xf, axis=0)
        rstd = tl.rsqrt(ss * (1.0 / GS) + EPS)
        w = tl.load(w_ptr + col, mask=m, other=0.0).to(tl.float32)
        t = xf * rstd
        tl.store(
            out_ptr + offs,
            (t * w + t).to(out_ptr.dtype.element_ty),
            mask=m,
            eviction_policy="evict_first",
        )


@triton.jit
def _grouped_gemma_rmsnorm_hoist_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    EPS: tl.constexpr,
    GS: tl.constexpr,
    G: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // G
    pid_g = pid - pid_m * G
    k = tl.arange(0, BLOCK_G)
    col = pid_g * GS + k
    if BLOCK_G == GS:
        offs = pid_m * (G * GS) + col
        xf = tl.load(x_ptr + offs, eviction_policy="evict_first").to(
            tl.float32
        )
        w = tl.load(w_ptr + col).to(tl.float32)
        ss = tl.sum(xf * xf, axis=0)
        rstd = tl.rsqrt(ss * (1.0 / GS) + EPS)
        t = xf * rstd
        tl.store(
            out_ptr + offs,
            (t * w + t).to(out_ptr.dtype.element_ty),
            eviction_policy="evict_first",
        )
    else:
        m = k < GS
        offs = pid_m * (G * GS) + col
        xf = tl.load(
            x_ptr + offs, mask=m, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        w = tl.load(w_ptr + col, mask=m, other=0.0).to(tl.float32)
        ss = tl.sum(xf * xf, axis=0)
        rstd = tl.rsqrt(ss * (1.0 / GS) + EPS)
        t = xf * rstd
        tl.store(
            out_ptr + offs,
            (t * w + t).to(out_ptr.dtype.element_ty),
            mask=m,
            eviction_policy="evict_first",
        )


@triton.jit
def _grouped_gemma_rmsnorm_strided_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    H,
    GS: tl.constexpr,
    slast,
    nd,
    meta_ptr,
    EPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    MAXD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)
    acc = pid_m.to(tl.int64)
    src = acc * 0
    for i in tl.static_range(MAXD):
        if i < nd - 1:
            cum = tl.load(meta_ptr + i)
            st = tl.load(meta_ptr + MAXD + i)
            q = acc // cum
            src += q * st
            acc -= q * cum
    src += acc * tl.load(meta_ptr + MAXD + nd - 2)
    k = tl.arange(0, BLOCK_G)
    col = pid_g * GS + k
    m = k < GS
    xf = tl.load(x_ptr + src + col.to(tl.int64) * slast, mask=m, other=0.0).to(
        tl.float32
    )
    ss = tl.sum(xf * xf, axis=0)
    rstd = tl.rsqrt(ss * (1.0 / GS) + EPS)
    w = tl.load(w_ptr + col, mask=m, other=0.0).to(tl.float32)
    t = xf * rstd
    tl.store(
        out_ptr + pid_m * H + col,
        (t * w + t).to(out_ptr.dtype.element_ty),
        mask=m,
    )


def grouped_gemma_rmsnorm(x, weight, group_size, eps=1e-6):
    numel = x.numel()
    H = x.shape[-1]
    gs = group_size
    G = H // gs
    block_g = 1 << (gs - 1).bit_length()
    progs = numel // gs
    if x.is_contiguous():
        out = torch.empty_like(x)
        if numel == 0:
            return out
        if block_g <= 1024:
            num_warps = 1
        else:
            num_warps = 0
        kernel = (
            _grouped_gemma_rmsnorm_hoist_kernel
            if (block_g <= 1024 or progs <= 128 or block_g == gs)
            else _grouped_gemma_rmsnorm_kernel
        )
        if num_warps:
            kernel[(progs,)](
                x,
                weight,
                out,
                eps,
                gs,
                G,
                block_g,
                num_warps=num_warps,
            )
        else:
            kernel[(progs,)](
                x,
                weight,
                out,
                eps,
                gs,
                G,
                block_g,
            )
        return out
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if numel == 0:
        return out
    M = numel // H
    num_warps = 1 if block_g <= 1024 else 4
    maxd = 8
    nd = x.dim()
    cums = [1] * (nd - 1)
    for i in range(nd - 3, -1, -1):
        cums[i] = cums[i + 1] * int(x.shape[i + 1])
    rstrides = [int(s) for s in x.stride()[: nd - 1]]
    meta = cums + [0] * (maxd - len(cums)) + rstrides
    meta += [0] * (2 * maxd - len(meta))
    meta_t = torch.tensor(meta, dtype=torch.int64, device=x.device)
    _grouped_gemma_rmsnorm_strided_kernel[(M, G)](
        x,
        weight,
        out,
        H,
        gs,
        int(x.stride()[-1]),
        nd,
        meta_t,
        EPS=eps,
        BLOCK_G=block_g,
        MAXD=maxd,
        num_warps=num_warps,
    )
    return out


__all__ = ["grouped_gemma_rmsnorm"]
