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
def _row(X, T, Y, INNER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    mask = cols < INNER
    tau = tl.load(T + row).to(tl.float32)
    x = tl.load(X + row * INNER + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        Y + row * INNER + cols,
        (x * tau).to(Y.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _multirow(
    X,
    T,
    Y,
    rows,
    INNER: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(INNER, BLOCK_N)
    tile_m = pid // num_n
    tile_n = pid - tile_m * num_n
    row = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (row[:, None] < rows) & (col[None, :] < INNER)
    tau = tl.load(T + row, mask=row < rows, other=0.0).to(tl.float32)
    x = tl.load(
        X + row[:, None] * INNER + col[None, :],
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    tl.store(
        Y + row[:, None] * INNER + col[None, :],
        (x * tau[:, None]).to(Y.dtype.element_ty),
        mask=mask,
    )


def log_scaling_tau(x, tau):
    source = x.contiguous()
    rows = source.shape[0]
    out = torch.empty_like(source)
    if rows == 0:
        return out
    inner = source.numel() // rows
    if inner == 0:
        return out
    scale = tau.contiguous()
    if rows >= 256 and inner >= 1024:
        block_n = inner if inner in (1024, 2048, 4096, 8192) else 1024
        block_m = 2
        grid = (triton.cdiv(rows, block_m) * triton.cdiv(inner, block_n),)
        _multirow[grid](
            source,
            scale,
            out,
            rows,
            INNER=inner,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=2,
            num_stages=1,
        )
        return out
    if inner in (512, 1024, 2048, 4096, 8192):
        block, warps = (inner, 2 if inner <= 2048 else 4)
    elif inner <= 64:
        block, warps = (triton.next_power_of_2(max(1, inner)), 1)
    elif inner <= 256:
        block, warps = (triton.next_power_of_2(inner), 2)
    else:
        block, warps = (1024, 4)
    _row[rows, triton.cdiv(inner, block)](
        source,
        scale,
        out,
        INNER=inner,
        BLOCK=block,
        num_warps=warps,
        num_stages=1,
    )
    return out


__all__ = ["log_scaling_tau"]
