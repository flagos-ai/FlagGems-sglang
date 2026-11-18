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

_HEAD_DIM = 128
_SMALL_LIMIT = 32768

_SMALL_NOMASK = [
    triton.Config({"BLOCK_M": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_M": 8}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_M": 8}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 16}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_M": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 128}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 128}, num_warps=16, num_stages=1),
]

_SMALL_MASKED = [
    triton.Config({"BLOCK_M": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_M": 8}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_SMALL_NOMASK, key=["total_rows"])
@triton.jit
def _small_nomask(
    X, W, Y, total_rows, eps, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * (BLOCK_M * HEAD_DIM)
    offs = base + tl.arange(0, BLOCK_M * HEAD_DIM)
    x = tl.load(X + offs)
    xf = tl.reshape(x, (BLOCK_M, HEAD_DIM)).to(tl.float32)
    ms = tl.sum(xf * xf, axis=1) * (1.0 / HEAD_DIM)
    rms = tl.rsqrt(ms + eps)
    w = tl.load(W + tl.arange(0, HEAD_DIM)).to(tl.float32)
    out = (xf * rms[:, None] * w[None, :]).to(x.dtype)
    tl.store(Y + offs, tl.reshape(out, (BLOCK_M * HEAD_DIM,)))


@triton.autotune(configs=_SMALL_MASKED, key=["total_rows"])
@triton.jit
def _small_masked(
    X, W, Y, total_rows, eps, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
):
    pid = tl.program_id(0)
    base = pid * (BLOCK_M * HEAD_DIM)
    offs = base + tl.arange(0, BLOCK_M * HEAD_DIM)
    total = total_rows * HEAD_DIM
    mask = offs < total
    x = tl.load(X + offs, mask=mask, other=0.0)
    xf = tl.reshape(x, (BLOCK_M, HEAD_DIM)).to(tl.float32)
    ms = tl.sum(xf * xf, axis=1) * (1.0 / HEAD_DIM)
    rms = tl.rsqrt(ms + eps)
    w = tl.load(W + tl.arange(0, HEAD_DIM)).to(tl.float32)
    out = (xf * rms[:, None] * w[None, :]).to(x.dtype)
    tl.store(Y + offs, tl.reshape(out, (BLOCK_M * HEAD_DIM,)), mask=mask)


@triton.jit
def _small_strided_exact(
    X,
    W,
    Y,
    stride_xb,
    stride_xs,
    S: tl.constexpr,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    base = pid_b * stride_xb + pid_s * stride_xs
    y_base = (pid_b * S + pid_s) * (H * HEAD_DIM)
    d = tl.arange(0, HEAD_DIM)
    w = tl.load(W + d).to(tl.float32)
    for h_start in tl.static_range(0, H, BLOCK_H):
        elem = h_start * HEAD_DIM + tl.arange(0, BLOCK_H * HEAD_DIM)
        x = tl.load(X + base + elem)
        xf = tl.reshape(x, (BLOCK_H, HEAD_DIM)).to(tl.float32)
        ms = tl.sum(xf * xf, axis=1) * (1.0 / HEAD_DIM)
        rms = tl.rsqrt(ms + eps)
        out = (xf * rms[:, None] * w[None, :]).to(x.dtype)
        tl.store(Y + y_base + elem, tl.reshape(out, (BLOCK_H * HEAD_DIM,)))


@triton.jit
def _small_strided_masked(
    X,
    W,
    Y,
    stride_xb,
    stride_xs,
    S: tl.constexpr,
    H: tl.constexpr,
    eps,
    BLOCK_H: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_b = tl.program_id(1)
    base = pid_b * stride_xb + pid_s * stride_xs
    y_base = (pid_b * S + pid_s) * (H * HEAD_DIM)
    d = tl.arange(0, HEAD_DIM)
    w = tl.load(W + d).to(tl.float32)
    for i in tl.static_range(0, NUM_BLOCKS):
        h_start = i * BLOCK_H
        elem = h_start * HEAD_DIM + tl.arange(0, BLOCK_H * HEAD_DIM)
        mask = elem < H * HEAD_DIM
        x = tl.load(X + base + elem, mask=mask, other=0.0)
        xf = tl.reshape(x, (BLOCK_H, HEAD_DIM)).to(tl.float32)
        ms = tl.sum(xf * xf, axis=1) * (1.0 / HEAD_DIM)
        rms = tl.rsqrt(ms + eps)
        out = (xf * rms[:, None] * w[None, :]).to(x.dtype)
        tl.store(
            Y + y_base + elem,
            tl.reshape(out, (BLOCK_H * HEAD_DIM,)),
            mask=mask,
        )


@triton.jit
def _small_strided_general(
    X,
    W,
    Y,
    stride_xb,
    stride_xs,
    stride_xh,
    stride_xd,
    S,
    H,
    eps,
    BLOCK_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = pid_tok // S
    s = pid_tok - b * S
    base = b * stride_xb + s * stride_xs
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offs < H
    d = tl.arange(0, HEAD_DIM)
    x_ptrs = X + base + h_offs[:, None] * stride_xh + d[None, :] * stride_xd
    x = tl.load(x_ptrs, mask=mask_h[:, None], other=0.0)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=1) * (1.0 / HEAD_DIM)
    rms = tl.rsqrt(ms + eps)
    w = tl.load(W + d).to(tl.float32)
    out = (xf * rms[:, None] * w[None, :]).to(x.dtype)
    y_base = pid_tok * H * HEAD_DIM
    y_ptrs = Y + y_base + h_offs[:, None] * HEAD_DIM + d[None, :]
    tl.store(y_ptrs, out, mask=mask_h[:, None])


@triton.jit
def _big_rmsnorm(
    x,
    weight,
    out,
    eps,
    S: tl.constexpr,
    H: tl.constexpr,
    NROWS: tl.constexpr,
    BSTRIDE: tl.constexpr,
    TSTRIDE: tl.constexpr,
    ROWS: tl.constexpr,
):
    cols = tl.arange(0, 128)
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    if TSTRIDE == H * 128 and BSTRIDE == S * TSTRIDE:
        base = rows * 128
    else:
        tokens = rows // H
        if BSTRIDE == S * TSTRIDE:
            base = tokens * TSTRIDE + (rows % H) * 128
        else:
            base = (
                (tokens // S) * BSTRIDE
                + (tokens % S) * TSTRIDE
                + (rows % H) * 128
            )
    if NROWS % ROWS == 0:
        values = tl.load(x + base[:, None] + cols[None, :]).to(tl.float32)
    else:
        values = tl.load(
            x + base[:, None] + cols[None, :],
            mask=rows[:, None] < NROWS,
            other=0,
        ).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=1) / 128 + eps)
    scaled = (values * inv_rms[:, None]).to(tl.int32, bitcast=True)
    normalized = ((scaled + 0x8000) & -65536).to(tl.float32, bitcast=True)
    w = tl.load(weight + cols).to(tl.float32)
    bits = (normalized * w[None, :]).to(tl.int32, bitcast=True)
    result = ((bits + 0x8000) >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
    if NROWS % ROWS == 0:
        tl.store(out + rows[:, None] * 128 + cols[None, :], result)
    else:
        tl.store(
            out + rows[:, None] * 128 + cols[None, :],
            result,
            mask=rows[:, None] < NROWS,
        )


@triton.jit
def _big_fast(x, weight, out, eps, NROWS: tl.constexpr):
    cols = tl.arange(0, 128)
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    offsets = rows[:, None] * 128 + cols[None, :]
    if NROWS % 16 == 0:
        values = tl.load(x + offsets).to(tl.float32)
    else:
        values = tl.load(x + offsets, mask=rows[:, None] < NROWS, other=0).to(
            tl.float32
        )
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=1) / 128 + eps)
    scaled = (values * inv_rms[:, None]).to(tl.int32, bitcast=True)
    normalized = ((scaled + 0x8000) & -65536).to(tl.float32, bitcast=True)
    result = normalized * tl.load(weight + cols).to(tl.float32)[None, :]
    bits = result.to(tl.int32, bitcast=True)
    result = ((bits + 0x8000) >> 16).to(tl.int16).to(tl.bfloat16, bitcast=True)
    if NROWS % 16 == 0:
        tl.store(out + offsets, result)
    else:
        tl.store(out + offsets, result, mask=rows[:, None] < NROWS)


def _small_path(x, weight, out, eps, b, s, h, d, total_rows):
    if x.is_contiguous():
        if total_rows % 128 == 0:

            def grid(meta):
                return (total_rows // meta["BLOCK_M"],)

            _small_nomask[grid](
                x, weight, out, total_rows, eps, HEAD_DIM=_HEAD_DIM
            )
        else:

            def grid(meta):
                return (triton.cdiv(total_rows, meta["BLOCK_M"]),)

            _small_masked[grid](
                x, weight, out, total_rows, eps, HEAD_DIM=_HEAD_DIM
            )
        return out
    stride_xb = x.stride(0)
    stride_xs = x.stride(1)
    stride_xh = x.stride(2)
    stride_xd = x.stride(3)
    if stride_xd == 1 and stride_xh == _HEAD_DIM:
        bh = 0
        for cand in (32, 16, 8, 4):
            if h % cand == 0:
                bh = cand
                break
        if bh != 0:
            nw = min(max(bh // 4, 1), 16)
            _small_strided_exact[(s, b)](
                x,
                weight,
                out,
                stride_xb,
                stride_xs,
                S=s,
                H=h,
                eps=eps,
                BLOCK_H=bh,
                HEAD_DIM=_HEAD_DIM,
                num_warps=nw,
                num_stages=1,
            )
        else:
            bh = 8
            num_blocks = (h + bh - 1) // bh
            _small_strided_masked[(s, b)](
                x,
                weight,
                out,
                stride_xb,
                stride_xs,
                S=s,
                H=h,
                eps=eps,
                BLOCK_H=bh,
                NUM_BLOCKS=num_blocks,
                HEAD_DIM=_HEAD_DIM,
                num_warps=2,
                num_stages=1,
            )
        return out
    n_tokens = b * s
    block_h = 32
    _small_strided_general[(n_tokens, triton.cdiv(h, block_h))](
        x,
        weight,
        out,
        stride_xb,
        stride_xs,
        stride_xh,
        stride_xd,
        s,
        h,
        eps,
        BLOCK_H=block_h,
        HEAD_DIM=_HEAD_DIM,
        num_warps=4,
        num_stages=1,
    )
    return out


def _big_path(x, weight, out, eps, b, s, h, d, total_rows):
    if x.is_contiguous():
        blocks = (total_rows + 15) // 16
        _big_fast[(blocks,)](x, weight, out, eps, total_rows)
        return out
    _big_rmsnorm[(triton.cdiv(total_rows, 16),)](
        x,
        weight,
        out,
        eps,
        s,
        h,
        total_rows,
        x.stride(0),
        x.stride(1),
        ROWS=16,
        num_warps=4,
        enable_fp_fusion=True,
    )
    return out


def zimage_qk_rmsnorm(x, weight, eps):
    if x.ndim != 4 or x.shape[-1] != 128:
        return None
    if not weight.is_contiguous() or weight.shape != (128,):
        return None
    if x.stride(3) != 1:
        return None
    b, s, h, d = x.shape
    out = torch.empty((b, s, h, d), dtype=x.dtype, device=x.device)
    if x.numel() == 0:
        return out
    total_rows = b * s * h
    if total_rows <= _SMALL_LIMIT or x.dtype is not torch.bfloat16:
        return _small_path(x, weight, out, eps, b, s, h, d, total_rows)
    return _big_path(x, weight, out, eps, b, s, h, d, total_rows)


__all__ = ["zimage_qk_rmsnorm"]
