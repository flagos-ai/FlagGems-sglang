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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _one_pass_rms_norm_kernel(
    X,
    W,
    Y,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_COLS)
    col_mask = cols < N_COLS
    mask = (rows[:, None] < N_ROWS) & col_mask[None, :]
    offsets = rows[:, None] * N_COLS + cols[None, :]
    values = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    if N_COLS != BLOCK_COLS:
        values = tl.where(col_mask[None, :], values, 0.0)
    weights = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(values * values, axis=1) / N_COLS + EPS)
    result = values * rstd[:, None] * weights[None, :]
    tl.store(Y + offsets, result.to(Y.dtype.element_ty), mask=mask)


def one_pass_rms_norm(x, w, eps):
    values = x.contiguous()
    weights = w.contiguous()
    out = torch.empty_like(values)
    cols = values.shape[-1]
    rows = values.numel() // cols
    block_cols = triton.next_power_of_2(cols)
    block_rows = min(2, triton.next_power_of_2(max(1, rows // 512)))
    num_warps = 4
    if rows >= 4096 and block_cols > 2048:
        block_rows = 1
        num_warps = 2
    if rows <= 64:
        num_warps = 2
    if rows >= 4096:
        block_rows, num_warps = (2, 1) if block_cols <= 2048 else (1, 2)
        kernel = _linear_tile if block_cols <= 2048 else _late_weight
        kernel.run(
            values,
            weights,
            out,
            rows,
            cols,
            eps,
            block_rows,
            block_cols,
            grid=(triton.cdiv(rows, block_rows),),
            warmup=False,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _one_pass_rms_norm_kernel.run(
            values,
            weights,
            out,
            N_ROWS=rows,
            N_COLS=cols,
            EPS=eps,
            BLOCK_ROWS=block_rows,
            BLOCK_COLS=block_cols,
            grid=(triton.cdiv(rows, block_rows),),
            warmup=False,
            num_warps=num_warps,
            num_stages=1,
        )
    return out


@triton.jit
def _linear_tile(
    X,
    W,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    linear = tl.arange(0, R * B)
    row = tl.program_id(0) * R + linear // B
    col = linear % B
    valid = (row < M) & (col < D)
    x = tl.reshape(tl.load(X + row * D + col, valid, 0).to(tl.float32), (R, B))
    w = tl.load(W + tl.arange(0, B), tl.arange(0, B) < D, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / D + E)
    y = x * rstd[:, None] * w[None, :]
    tl.store(Y + row * D + col, tl.reshape(y, (R * B,)), valid)


@triton.jit
def _late_weight(
    X,
    W,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, B)
    offsets = rows[:, None] * D + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < D)
    x = tl.load(X + offsets, valid, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / D + E)
    normalized = x * rstd[:, None]
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
    tl.store(Y + offsets, normalized * w[None, :], valid)


__all__ = ["one_pass_rms_norm"]
