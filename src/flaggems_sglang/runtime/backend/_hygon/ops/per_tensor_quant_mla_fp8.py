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

_INV_448 = tl.constexpr(0.002232142857142857)

_BLOCK1 = 4096
_PPE1 = _BLOCK1
_INLINE_NC = 1024
_INLINE_RB = 1024
_SMALL_N1 = 8192
_SMALL_N2 = 16384
_SBLOCK1 = 8192
_SBLOCK2 = 16384


@triton.jit
def _to_f32(v):
    if v.dtype == tl.bfloat16:
        u = v.to(tl.uint16, bitcast=True).to(tl.uint32)
        return (u << 16).to(tl.float32, bitcast=True)
    else:
        return v.to(tl.float32)


@triton.jit
def _e4m3_bytes(xb):
    sgn = (xb >> 24) & 0x80
    mag = xb & 0x7FFFFFFF
    is_nan = mag > 0x7F800000
    mag = tl.minimum(mag, 0x43E00000)
    rn = ((mag + 0x7FFFF + ((mag >> 20) & 1) - 0x3C000000) >> 20).to(tl.int32)
    mf = mag.to(tl.float32, bitcast=True)
    sub = ((mf + 24576.0).to(tl.uint32, bitcast=True) - 0x46C00000).to(
        tl.int32
    )
    code = tl.where(mag >= 0x3C800000, rn, sub)
    code = tl.where(is_nan, 0x7F, code)
    return (code.to(tl.uint32) | sgn).to(tl.uint8)


@triton.jit
def _elem_addr(idx, srow, scol, D, dims_ptr, strs_ptr, ND, MODE: tl.constexpr):
    if MODE == 0:
        return idx
    elif MODE == 1:
        idx = idx.to(tl.int64)
        row = idx // D
        return row * srow + (idx - row * D) * scol
    else:
        rem = idx.to(tl.int64)
        off = rem * 0
        for d in range(0, ND):
            sz = tl.load(dims_ptr + d)
            st = tl.load(strs_ptr + d)
            off += (rem % sz) * st
            rem = rem // sz
        return off


@triton.jit
def _small_quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    N,
    EPS,
    srow,
    scol,
    D,
    dims_ptr,
    strs_ptr,
    ND,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
    SPE: tl.constexpr,
    FULL: tl.constexpr,
):
    run = 0.0
    nan_any = 0
    for j in tl.static_range(SPE):
        offs = j * BLOCK + tl.arange(0, BLOCK)
        if FULL:
            v = tl.load(
                x_ptr
                + _elem_addr(offs, srow, scol, D, dims_ptr, strs_ptr, ND, MODE)
            )
        else:
            m = offs < N
            v = tl.load(
                x_ptr
                + _elem_addr(
                    offs, srow, scol, D, dims_ptr, strs_ptr, ND, MODE
                ),
                mask=m,
                other=0.0,
            )
        f = _to_f32(v)
        run = tl.maximum(run, tl.max(tl.abs(f), axis=0))
        nan_any = nan_any | tl.max((f != f).to(tl.int32), axis=0)
    scale = tl.where(
        nan_any != 0, float("nan"), tl.maximum(run, EPS) * _INV_448
    )
    tl.store(s_ptr, scale)
    rcp = 1.0 / scale
    for j in tl.static_range(SPE):
        offs = j * BLOCK + tl.arange(0, BLOCK)
        if FULL:
            v = tl.load(
                x_ptr
                + _elem_addr(offs, srow, scol, D, dims_ptr, strs_ptr, ND, MODE)
            )
            q = _to_f32(v) * rcp
            code = _e4m3_bytes(q.to(tl.uint32, bitcast=True))
            tl.store(
                y_ptr + offs, code.to(y_ptr.dtype.element_ty, bitcast=True)
            )
        else:
            m = offs < N
            v = tl.load(
                x_ptr
                + _elem_addr(
                    offs, srow, scol, D, dims_ptr, strs_ptr, ND, MODE
                ),
                mask=m,
                other=0.0,
            )
            q = _to_f32(v) * rcp
            code = _e4m3_bytes(q.to(tl.uint32, bitcast=True))
            tl.store(
                y_ptr + offs,
                code.to(y_ptr.dtype.element_ty, bitcast=True),
                mask=m,
            )


@triton.jit
def _absmax_kernel(
    x_ptr,
    part_ptr,
    N,
    srow,
    scol,
    D,
    dims_ptr,
    strs_ptr,
    ND,
    MODE: tl.constexpr,
    I64: tl.constexpr,
    BF16: tl.constexpr,
    BLOCK: tl.constexpr,
    SPE: tl.constexpr,
    FULL: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK * SPE)
    if I64:
        base = base.to(tl.int64)
    if BF16:
        run = tl.full((), 0, tl.int32)
        for j in tl.static_range(SPE):
            offs = base + j * BLOCK + tl.arange(0, BLOCK)
            if FULL:
                v = tl.load(
                    x_ptr
                    + _elem_addr(
                        offs, srow, scol, D, dims_ptr, strs_ptr, ND, MODE
                    )
                )
            else:
                m = offs < N
                v = tl.load(
                    x_ptr
                    + _elem_addr(
                        offs, srow, scol, D, dims_ptr, strs_ptr, ND, MODE
                    ),
                    mask=m,
                    other=0.0,
                )
            u = v.to(tl.uint16, bitcast=True).to(tl.int32) & 0x7FFF
            run = tl.maximum(run, tl.max(u, axis=0))
        out = tl.where(
            run > 0x7F80,
            float("nan"),
            (run << 16)
            .to(tl.uint32, bitcast=True)
            .to(tl.float32, bitcast=True),
        )
    else:
        run = 0.0
        nan_any = 0
        for j in tl.static_range(SPE):
            offs = base + j * BLOCK + tl.arange(0, BLOCK)
            if FULL:
                v = tl.load(
                    x_ptr
                    + _elem_addr(
                        offs, srow, scol, D, dims_ptr, strs_ptr, ND, MODE
                    )
                )
            else:
                m = offs < N
                v = tl.load(
                    x_ptr
                    + _elem_addr(
                        offs, srow, scol, D, dims_ptr, strs_ptr, ND, MODE
                    ),
                    mask=m,
                    other=0.0,
                )
            f = _to_f32(v)
            run = tl.maximum(run, tl.max(tl.abs(f), axis=0))
            nan_any = nan_any | tl.max((f != f).to(tl.int32), axis=0)
        out = tl.where(nan_any != 0, float("nan"), run)
    tl.store(part_ptr + pid, out)


@triton.jit
def _scale_kernel(part_ptr, out_ptr, NC, EPS, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mv = tl.full((BLOCK,), 0.0, tl.float32)
    nan_any = 0
    for start in range(0, NC, BLOCK):
        pv = tl.load(
            part_ptr + start + offs, mask=start + offs < NC, other=0.0
        )
        mv = tl.maximum(mv, pv)
        nan_any = nan_any | tl.max((pv != pv).to(tl.int32), axis=0)
    amax = tl.max(mv, axis=0)
    tl.store(
        out_ptr,
        tl.where(nan_any != 0, float("nan"), tl.maximum(amax, EPS) * _INV_448),
    )


@triton.jit
def _quant_kernel(
    x_ptr,
    y_ptr,
    s_ptr,
    part_ptr,
    N,
    NC,
    EPS,
    srow,
    scol,
    D,
    dims_ptr,
    strs_ptr,
    ND,
    MODE: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
    SPE: tl.constexpr,
    FULL: tl.constexpr,
    INLINE: tl.constexpr,
    RB: tl.constexpr,
    WARPS8: tl.constexpr = False,
):
    pid = tl.program_id(0)
    base = pid * (BLOCK * SPE)
    if I64:
        base = base.to(tl.int64)
    if INLINE:
        offs_r = tl.arange(0, RB)
        mv = tl.full((RB,), 0.0, tl.float32)
        nan_any = 0
        for start in range(0, NC, RB):
            pv = tl.load(
                part_ptr + start + offs_r, mask=start + offs_r < NC, other=0.0
            )
            mv = tl.maximum(mv, pv)
            nan_any = nan_any | tl.max((pv != pv).to(tl.int32), axis=0)
        amax = tl.max(mv, axis=0)
        scale = tl.where(
            nan_any != 0, float("nan"), tl.maximum(amax, EPS) * _INV_448
        )
        if pid == 0:
            tl.store(s_ptr, scale)
        rcp = 1.0 / scale
    else:
        rcp = 1.0 / tl.load(s_ptr)
    for j in tl.static_range(SPE):
        offs = base + j * BLOCK + tl.arange(0, BLOCK)
        if FULL:
            if WARPS8:
                tl.debug_barrier()
            v = tl.load(
                x_ptr
                + _elem_addr(offs, srow, scol, D, dims_ptr, strs_ptr, ND, MODE)
            )
            q = _to_f32(v) * rcp
            code = _e4m3_bytes(q.to(tl.uint32, bitcast=True))
            tl.store(
                y_ptr + offs, code.to(y_ptr.dtype.element_ty, bitcast=True)
            )
        else:
            m = offs < N
            v = tl.load(
                x_ptr
                + _elem_addr(
                    offs, srow, scol, D, dims_ptr, strs_ptr, ND, MODE
                ),
                mask=m,
                other=0.0,
            )
            q = _to_f32(v) * rcp
            code = _e4m3_bytes(q.to(tl.uint32, bitcast=True))
            tl.store(
                y_ptr + offs,
                code.to(y_ptr.dtype.element_ty, bitcast=True),
                mask=m,
            )


def _next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _merge_leading(sizes, strides):
    dims = []
    for sz, st in zip(sizes, strides):
        if sz == 1:
            continue
        if dims and dims[-1][1] == st * sz:
            dims[-1] = (dims[-1][0] * sz, st)
        else:
            dims.append((sz, st))
    return dims


def per_tensor_quant_mla_fp8(x, eps=1e-12):
    shape = x.shape
    dev = x.device
    y = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)
    s = torch.empty(1, dtype=torch.float32, device=dev)
    n = x.numel()
    epsf = float(eps)
    if n == 0:
        _scale_kernel[(1,)](s, s, 0, epsf, BLOCK=4096, num_warps=8)
        return y, s

    dim = x.dim()
    d_last = shape[-1] if dim > 0 else 1
    full1 = n % _PPE1 == 0
    nc = (n + _PPE1 - 1) // _PPE1
    i64 = n > 0x7FFFFFFF
    scol1 = x.stride(-1) if dim > 0 else 1
    scol = scol1
    if x.is_contiguous():
        mode = 0
        srow = 0
        d_last = 1
        dims_t = s
        strs_t = s
        nd = 0
    elif scol1 == 1:
        lead = _merge_leading(list(shape)[:-1], list(x.stride())[:-1])
        d_last = shape[-1]
        dims_t = s
        strs_t = s
        nd = 0
        if len(lead) == 1:
            mode = 1
            srow = lead[0][1]
        else:
            mode = 2
            srow = 0
            d_last = 1
            dims_l = []
            strs_l = []
            for d in range(dim - 1, -1, -1):
                if shape[d] != 1:
                    dims_l.append(shape[d])
                    strs_l.append(x.stride(d))
            dims_t = torch.tensor(dims_l, dtype=torch.int64, device=dev)
            strs_t = torch.tensor(strs_l, dtype=torch.int64, device=dev)
            nd = len(dims_l)
    else:
        mode = 2
        srow = 0
        d_last = 1
        dims_l = []
        strs_l = []
        for d in range(dim - 1, -1, -1):
            if shape[d] != 1:
                dims_l.append(shape[d])
                strs_l.append(x.stride(d))
        if not dims_l:
            dims_l = [1]
            strs_l = [0]
        dims_t = torch.tensor(dims_l, dtype=torch.int64, device=dev)
        strs_t = torch.tensor(strs_l, dtype=torch.int64, device=dev)
        nd = len(dims_l)

    if n <= _SMALL_N2:
        if n <= _SMALL_N1:
            _small_quant_kernel[(1,)](
                x,
                y.view(torch.uint8),
                s,
                n,
                epsf,
                srow,
                scol,
                d_last,
                dims_t,
                strs_t,
                nd,
                MODE=mode,
                BLOCK=_SBLOCK1,
                SPE=1,
                FULL=(n % _SBLOCK1 == 0),
                num_warps=4,
            )
        else:
            _small_quant_kernel[(1,)](
                x,
                y.view(torch.uint8),
                s,
                n,
                epsf,
                srow,
                scol,
                d_last,
                dims_t,
                strs_t,
                nd,
                MODE=mode,
                BLOCK=_SBLOCK2,
                SPE=1,
                FULL=(n % _SBLOCK2 == 0),
                num_warps=8,
            )
        return y, s

    parts = torch.empty(nc, dtype=torch.float32, device=dev)
    _absmax_kernel[(nc,)](
        x,
        parts,
        n,
        srow,
        scol,
        d_last,
        dims_t,
        strs_t,
        nd,
        MODE=mode,
        I64=i64,
        BF16=(x.dtype == torch.bfloat16),
        BLOCK=_BLOCK1,
        SPE=1,
        FULL=full1,
        num_warps=4 if n <= 16 * 1024 * 1024 else 8,
        num_stages=3,
    )
    if nc <= _INLINE_NC:
        if nc <= 256:
            blk2, w2, st2 = 4096, 4, 3
        else:
            blk2, w2, st2 = 8192, 4, 4
        nc2 = (n + blk2 - 1) // blk2
        _quant_kernel[(nc2,)](
            x,
            y.view(torch.uint8),
            s,
            parts,
            n,
            nc,
            epsf,
            srow,
            scol,
            d_last,
            dims_t,
            strs_t,
            nd,
            MODE=mode,
            I64=i64,
            BLOCK=blk2,
            SPE=1,
            FULL=(n % blk2 == 0),
            INLINE=True,
            RB=min(_next_pow2(nc), _INLINE_RB),
            WARPS8=(blk2 == 8192),
            num_warps=w2,
            num_stages=st2,
        )
    else:
        scb = min(_next_pow2(nc), 32768)
        if scb < 4096:
            scb = 4096
        _scale_kernel[(1,)](parts, s, nc, epsf, BLOCK=scb, num_warps=8)
        blk2, w2 = 8192, 8
        nc2 = (n + blk2 - 1) // blk2
        _quant_kernel[(nc2,)](
            x,
            y.view(torch.uint8),
            s,
            parts,
            n,
            nc,
            epsf,
            srow,
            scol,
            d_last,
            dims_t,
            strs_t,
            nd,
            MODE=mode,
            I64=i64,
            BLOCK=blk2,
            SPE=1,
            FULL=(n % blk2 == 0),
            INLINE=False,
            RB=_INLINE_RB,
            num_warps=w2,
            num_stages=3,
        )
    return y, s


__all__ = ["per_tensor_quant_mla_fp8"]
