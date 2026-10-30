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

"""Operator: fla/kda_gate_chunk_cumsum"""

import torch
import triton
import triton.language as tl

_MAX_TILE_T = 64


@triton.autotune(
    configs=[
        # Small T is launch-bound, so many thin S-tiles (more programs) win.
        triton.Config({"BLOCK_S": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_S": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_S": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_S": 32}, num_warps=4, num_stages=1),
        # Mid T.
        triton.Config({"BLOCK_S": 64}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_S": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=1),
        # Large T is bandwidth-bound: the widest valid tile wins.
        triton.Config({"BLOCK_S": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=1),
    ],
    key=["S", "CS", "T"],
)
@triton.jit
def _kda_gate_chunk_cumsum_kernel(
    g_ptr,
    a_log_ptr,
    bias_ptr,
    out_ptr,
    T,
    S,
    CS: tl.constexpr,
    scale_val,
    lb_val,
    stride_gb,
    stride_gt,
    stride_gh,
    stride_gs,
    H,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_SAFE: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # b * H + h (fastest-varying for L2 locality)
    pid_c = tl.program_id(1)  # chunk index along T
    pid_s = tl.program_id(2)  # S tile index
    b = pid_bh // H
    h = pid_bh % H

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = offs_s < S

    a = tl.exp(tl.load(a_log_ptr + h))

    g_base = g_ptr + b * stride_gb + h * stride_gh
    # out is contiguous [B, T, H, S] fp32
    o_base = out_ptr + (b * T * H + h) * S

    offs_t = pid_c * CS + tl.arange(0, CS)
    tmask = offs_t < T
    m2d = tmask[:, None] & smask[None, :]

    x = tl.load(
        g_base + offs_t[:, None] * stride_gt + offs_s[None, :] * stride_gs,
        mask=m2d,
        other=0.0,
    ).to(tl.float32)
    if HAS_BIAS:
        bv = tl.load(bias_ptr + h * S + offs_s, mask=smask, other=0.0)
        x = x + bv[None, :]

    if IS_SAFE:
        gate = lb_val * tl.sigmoid(a * x)
    else:
        sp = tl.log(1.0 + tl.exp(x))
        gate = -a * sp
    gate = tl.where(tmask[:, None], gate, 0.0)

    c = tl.cumsum(gate, axis=0)
    if HAS_SCALE:
        c = c * scale_val
    tl.store(
        o_base + offs_t[:, None] * (H * S) + offs_s[None, :],
        c,
        mask=m2d,
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_S": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_S": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_S": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_S": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_S": 256}, num_warps=8, num_stages=2),
    ],
    key=["S", "CS", "BT"],
)
@triton.jit
def _kda_gate_chunk_cumsum_blocked_kernel(
    g_ptr,
    a_log_ptr,
    bias_ptr,
    out_ptr,
    T,
    S,
    CS: tl.constexpr,
    scale_val,
    lb_val,
    stride_gb,
    stride_gt,
    stride_gh,
    stride_gs,
    H,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_SAFE: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # b * H + h (fastest-varying for L2 locality)
    pid_c = tl.program_id(1)  # chunk index along T
    pid_s = tl.program_id(2)  # S tile index
    b = pid_bh // H
    h = pid_bh % H

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    smask = offs_s < S

    a = tl.exp(tl.load(a_log_ptr + h))

    g_base = g_ptr + b * stride_gb + h * stride_gh
    o_base = out_ptr + (b * T * H + h) * S

    rows = tl.arange(0, BT)
    t0 = pid_c * CS
    carry = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for i in range(0, CS, BT):
        offs_t = t0 + i + rows
        tmask = offs_t < T
        m2d = tmask[:, None] & smask[None, :]

        x = tl.load(
            g_base + offs_t[:, None] * stride_gt + offs_s[None, :] * stride_gs,
            mask=m2d,
            other=0.0,
        ).to(tl.float32)
        if HAS_BIAS:
            bv = tl.load(bias_ptr + h * S + offs_s, mask=smask, other=0.0)
            x = x + bv[None, :]

        if IS_SAFE:
            gate = lb_val * tl.sigmoid(a * x)
        else:
            sp = tl.log(1.0 + tl.exp(x))
            gate = -a * sp
        gate = tl.where(tmask[:, None], gate, 0.0)

        c = tl.cumsum(gate, axis=0) + carry[None, :]
        # Row-sum of this block is independent of the cumsum result, so the
        # next iteration's loads/activation can overlap the cumsum above.
        carry = carry + tl.sum(gate, axis=0)

        o_ptrs = o_base + offs_t[:, None] * (H * S) + offs_s[None, :]
        if HAS_SCALE:
            c = c * scale_val
        tl.store(o_ptrs, c, mask=m2d)


def kda_gate_chunk_cumsum(g, A_log, chunk_size, scale, dt_bias, lower_bound):
    b, t, h, s = g.shape
    g = g.contiguous()
    out = torch.empty((b, t, h, s), dtype=torch.float32, device=g.device)
    if t == 0 or h == 0 or s == 0 or b == 0:
        return out

    if chunk_size <= _MAX_TILE_T:

        def grid(meta):
            return (
                b * h,
                triton.cdiv(t, chunk_size),
                triton.cdiv(s, meta["BLOCK_S"]),
            )

        _kda_gate_chunk_cumsum_kernel[grid](
            g,
            A_log,
            dt_bias if dt_bias is not None else g,
            out,
            t,
            s,
            chunk_size,
            float(scale) if scale is not None else 0.0,
            float(lower_bound) if lower_bound is not None else 0.0,
            g.stride(0),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            h,
            HAS_BIAS=dt_bias is not None,
            HAS_SCALE=scale is not None,
            IS_SAFE=lower_bound is not None,
        )
    else:
        # BT must be a power of two dividing chunk_size (both are
        # powers of two).
        bt = min(chunk_size, 64)

        def grid(meta):
            return (
                b * h,
                triton.cdiv(t, chunk_size),
                triton.cdiv(s, meta["BLOCK_S"]),
            )

        _kda_gate_chunk_cumsum_blocked_kernel[grid](
            g,
            A_log,
            dt_bias if dt_bias is not None else g,
            out,
            t,
            s,
            chunk_size,
            float(scale) if scale is not None else 0.0,
            float(lower_bound) if lower_bound is not None else 0.0,
            g.stride(0),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            h,
            HAS_BIAS=dt_bias is not None,
            HAS_SCALE=scale is not None,
            IS_SAFE=lower_bound is not None,
            BT=bt,
        )
    return out


__all__ = ["kda_gate_chunk_cumsum"]
