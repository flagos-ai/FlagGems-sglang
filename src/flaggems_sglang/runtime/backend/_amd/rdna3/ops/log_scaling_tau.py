# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""log_scaling_tau for RDNA3 (gfx1100) — scales every row of a tensor by its
own factor (Triton).

The operator only reads and writes, so the launch is chosen for bandwidth: a
program takes one slice of one row, sized so that every lane of the block
moves a single 32-bit word, and consecutive programs take consecutive slices
of the same row. The loads bypass the per-CU caches, which nothing reads
again.

This kernel is written in portable Triton only. It must NOT call any
pre-compiled / vendor-specific cached operator — the whole computation is done
inside the Triton kernel.
"""

import torch
import triton
import triton.language as tl

# 512 elements a program, and one 32-bit word a lane, is what the W7900D
# reaches its copy bandwidth with: wider slices leave the device waiting on
# fewer, longer requests (1.5x at 32 MiB), narrower ones waste lanes.
_BLOCK = 512
_WORD_BITS = 32
_LANES_PER_WARP = 32
_MAX_WARPS = 16


@triton.jit
def _scale(
    X,
    T,
    Y,
    INNER: tl.constexpr,
    BLOCK: tl.constexpr,
    TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // TILES
    cols = (pid % TILES) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < INNER
    tau = tl.load(T + row).to(tl.float32)
    x = tl.load(
        X + row * INNER + cols,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
    ).to(tl.float32)
    tl.store(
        Y + row * INNER + cols,
        (x * tau).to(Y.dtype.element_ty),
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
    block = min(_BLOCK, triton.next_power_of_2(inner))
    tiles = triton.cdiv(inner, block)
    words = block * source.element_size() * 8 // _WORD_BITS
    warps = max(1, min(_MAX_WARPS, words // _LANES_PER_WARP))
    _scale[(rows * tiles,)](
        source,
        tau.contiguous(),
        out,
        INNER=inner,
        BLOCK=block,
        TILES=tiles,
        num_warps=warps,
        num_stages=1,
    )
    return out


__all__ = ["log_scaling_tau"]
