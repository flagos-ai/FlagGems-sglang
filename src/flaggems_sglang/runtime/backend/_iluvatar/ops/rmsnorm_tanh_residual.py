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

"""Fused RMS normalization, tanh gating, and residual addition."""

import torch
import triton
import triton.language as tl

# --- launch configuration (tuned on the target part; no vendor-specific ops)
# --
# Row widths at/below this keep the one-pass resident-row critical path.
_ONE_PASS_MAX_D = 2048
_STREAM_NARROW_MIN_ROWS = 64
_STREAM_WIDE_MIN_ROWS = 32
# Streaming chunk widths. Narrow rows stream a 1024-wide chunk at 8 warps;
# wide rows a 2048-wide chunk at 16.
_STREAM_CHUNK_NARROW = 1024
_STREAM_CHUNK_WIDE = 2048
_NARROW_WARPS = 16  # one-pass, D <= 2048
_WIDE_FEW_WARPS = 32  # one-pass, D > 2048
_STREAM_NARROW_WARPS = 8  # streaming, D <= 2048
_STREAM_WIDE_WARPS = 16  # streaming, D > 2048
# 1/ln(2): exp(-y) == exp2(-y * log2(e)) with one SFU instruction.
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _rmsnorm_tanh_residual_onepass_kernel(
    x_ptr,  # [x_rows, D] bf16, row-major
    g_ptr,  # [gate_rows, D] bf16, row-major
    r_ptr,  # [x_rows, D] bf16, row-major (residual)
    w_ptr,  # [D] bf16, contiguous
    o_ptr,  # [x_rows, D] bf16, contiguous
    eps,
    repeat,  # x rows covered by one gate row (>= 1)
    x_row_stride,
    g_row_stride,
    r_row_stride,
    D: tl.constexpr,
    BLOCK: tl.constexpr,  # >= D: the whole row lives in registers
):
    row = tl.program_id(0)
    grow = row // repeat
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # All loads issue before the reduction so their latencies overlap.
    xv = tl.load(x_ptr + row * x_row_stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    wv = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gv = tl.load(g_ptr + grow * g_row_stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    rv = tl.load(r_ptr + row * r_row_stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )

    inv = tl.rsqrt(tl.sum(xv * xv, axis=0) / D + eps)

    # Replicate the reference's rounding points exactly: norm -> bf16, * w ->
    # bf16, tanh(gate) -> bf16 (evaluated in fp32), product -> bf16, + r ->
    # bf16.
    normed = (xv * inv).to(tl.bfloat16).to(tl.float32) * wv
    normed = normed.to(tl.bfloat16).to(tl.float32)
    e2 = tl.math.exp2(gv * (-2.0 * _LOG2E))
    gated = (2.0 / (e2 + 1.0) - 1.0).to(tl.bfloat16).to(tl.float32)
    out = rv + (gated * normed).to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + row * D + offs, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _rmsnorm_tanh_residual_streaming_kernel(
    x_ptr,  # [x_rows, D] bf16, row-major
    g_ptr,  # [gate_rows, D] bf16, row-major
    r_ptr,  # [x_rows, D] bf16, row-major (residual)
    w_ptr,  # [D] bf16, contiguous
    o_ptr,  # [x_rows, D] bf16, contiguous
    eps,
    repeat,  # x rows covered by one gate row (>= 1)
    x_row_stride,
    g_row_stride,
    r_row_stride,
    D: tl.constexpr,
    BLOCK: tl.constexpr,  # chunk width
):
    row = tl.program_id(0)
    grow = row // repeat
    x_row = x_ptr + row * x_row_stride
    g_row = g_ptr + grow * g_row_stride
    r_row = r_ptr + row * r_row_stride
    offs = tl.arange(0, BLOCK)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, D, BLOCK):
        cmask = (start + offs) < D
        xv = tl.load(x_row + start + offs, mask=cmask, other=0.0).to(
            tl.float32
        )
        acc += xv * xv
    inv = tl.rsqrt(tl.sum(acc, axis=0) / D + eps)

    # Same rounding chain (and the same exp2-based tanh) as the one-pass
    # kernel.
    for start in range(0, D, BLOCK):
        cmask = (start + offs) < D
        xv = tl.load(x_row + start + offs, mask=cmask, other=0.0).to(
            tl.float32
        )
        wv = tl.load(w_ptr + start + offs, mask=cmask, other=0.0).to(
            tl.float32
        )
        gv = tl.load(g_row + start + offs, mask=cmask, other=0.0).to(
            tl.float32
        )
        rv = tl.load(r_row + start + offs, mask=cmask, other=0.0).to(
            tl.float32
        )
        normed = (xv * inv).to(tl.bfloat16).to(tl.float32) * wv
        normed = normed.to(tl.bfloat16).to(tl.float32)
        e2 = tl.math.exp2(gv * (-2.0 * _LOG2E))
        gated = (2.0 / (e2 + 1.0) - 1.0).to(tl.bfloat16).to(tl.float32)
        out = rv + (gated * normed).to(tl.bfloat16).to(tl.float32)
        tl.store(
            o_ptr + row * D + start + offs, out.to(tl.bfloat16), mask=cmask
        )


def rmsnorm_tanh_residual(x, gate, residual, weight, eps):
    if (
        x.dtype is not torch.bfloat16
        or gate.dtype is not torch.bfloat16
        or residual.dtype is not torch.bfloat16
        or weight.dtype is not torch.bfloat16
    ):
        return None
    if not (x.device == gate.device == residual.device == weight.device):
        return None
    if residual.shape != x.shape:
        return None
    if weight.dim() != 1 or not weight.is_contiguous():
        return None

    dim = x.shape[-1]
    if dim == 0 or x.numel() == 0:
        return torch.empty_like(x)

    xf = x.reshape(-1, dim)
    gf = gate.reshape(-1, dim)
    rf = residual.reshape(-1, dim)
    x_rows, g_rows = xf.shape[0], gf.shape[0]
    if g_rows == 0 or x_rows % g_rows != 0:
        return None
    repeat = x_rows // g_rows

    out = torch.empty((x_rows, dim), dtype=torch.bfloat16, device=x.device)

    # Narrow rows stream once there are enough CTAs; wide rows stream from 32
    # rows up. Otherwise the one-pass resident-row program runs.
    if dim <= _ONE_PASS_MAX_D:
        use_stream = x_rows >= _STREAM_NARROW_MIN_ROWS
    else:
        use_stream = x_rows >= _STREAM_WIDE_MIN_ROWS
    if use_stream:
        if dim <= _ONE_PASS_MAX_D:
            chunk, warps = _STREAM_CHUNK_NARROW, _STREAM_NARROW_WARPS
        else:
            chunk, warps = _STREAM_CHUNK_WIDE, _STREAM_WIDE_WARPS
        _rmsnorm_tanh_residual_streaming_kernel[(x_rows,)](
            xf,
            gf,
            rf,
            weight,
            out,
            eps,
            repeat,
            xf.stride(0),
            gf.stride(0),
            rf.stride(0),
            D=dim,
            BLOCK=chunk,
            num_warps=warps,
        )
    else:
        num_warps = (
            _NARROW_WARPS if dim <= _ONE_PASS_MAX_D else _WIDE_FEW_WARPS
        )
        _rmsnorm_tanh_residual_onepass_kernel[(x_rows,)](
            xf,
            gf,
            rf,
            weight,
            out,
            eps,
            repeat,
            xf.stride(0),
            gf.stride(0),
            rf.stride(0),
            D=dim,
            BLOCK=triton.next_power_of_2(dim),
            num_warps=num_warps,
        )
    return out.view(x.shape)


__all__ = ["rmsnorm_tanh_residual"]
