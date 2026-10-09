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

"""tanh-approximated gated GELU -- Ascend NPU specialization.
"""

import torch
import triton
import triton.language as tl

_INT32_MAX = 2**31 - 1

# gelu_tanh(v) = v * sigmoid(v * (+2*sqrt(2/pi) + (+2*sqrt(2/pi)*0.044715) * v^2)),
# the sigmoid identity of 0.5*v*(1 + tanh(sqrt(2/pi)*(v + 0.044715*v^3))).
# Self-saturating with no branch: the denominator saturates for large |t| and
# argument 0 (masked lanes) gives exactly 0.5.
_GELU_TANH_A = 1.5957691216057308  # 2*sqrt(2/pi)
_GELU_TANH_B = 0.07133681239524435  # 2*sqrt(2/pi)*0.044715

# (ROWS, P, num_warps, num_stages) candidates; BLOCK_N is fixed per shape by
# the launcher. The sweep covers two regimes: low warp counts with a moderate
# grid cap (best for the large, bandwidth-bound shapes — many CTAs each
# streaming a wide tile) and high warp counts with a large cap (best for tiny
# inputs, where the natural grid is already small and per-CTA launch cost
# dominates). num_stages varies the software pipelining depth of the
# grid-stride row-block loop.
_TUNE_CFGS = (
    # low-warp / medium-cap regime
    (2, 32, 1, 1),
    (2, 32, 1, 2),
    (2, 32, 1, 3),
    (4, 64, 1, 1),
    (4, 64, 1, 2),
    (4, 64, 1, 3),
    (4, 64, 2, 1),
    (4, 64, 2, 2),
    (8, 64, 1, 1),
    (8, 64, 1, 2),
    (8, 128, 1, 2),
    (8, 128, 1, 3),
    (8, 256, 1, 2),
    (8, 512, 1, 2),
    (8, 128, 2, 2),
    (16, 128, 1, 2),
    (16, 128, 1, 3),
    (16, 256, 1, 2),
    (16, 128, 2, 2),
    (4, 128, 1, 2),
    (4, 256, 1, 2),
    (16, 64, 2, 2),
    (16, 64, 2, 3),
    (8, 64, 2, 3),
    # high-warp / wide-grid regime
    (1, 64, 4, 1),
    (1, 64, 4, 2),
    (2, 32, 4, 1),
    (2, 32, 4, 2),
    (4, 128, 4, 2),
    (8, 512, 4, 2),
    (16, 128, 4, 2),
    (32, 64, 4, 2),
    (64, 64, 4, 2),
    (128, 16, 4, 2),
    (128, 64, 4, 2),
    (2, 64, 2, 2),
    (2, 128, 2, 2),
    (8, 128, 4, 2),
    (16, 64, 4, 2),
    (32, 128, 4, 2),
    (64, 128, 4, 2),
)


@triton.autotune(
    configs=[
        triton.Config({"ROWS": r, "P": p}, num_warps=nw, num_stages=ns)
        for r, p, nw, ns in _TUNE_CFGS
    ],
    key=["m", "d", "BLOCK_N"],
)
@triton.jit
def _gelu_tanh_and_mul_i32_kernel(
    x_ptr,
    out_ptr,
    m,
    d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    P: tl.constexpr,
):
    UNMASKED: tl.constexpr = BLOCK_N == d
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    nrb = tl.cdiv(m, ROWS)
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < d
    rows_off = tl.arange(0, ROWS)
    evict: tl.constexpr = "evict_first"

    for rb in range(pid, nrb, nprog):
        r = rb * ROWS + rows_off
        rmask = r < m
        base = r[:, None] * (2 * d) + cols[None, :]
        mask = rmask[:, None] & cmask[None, :]

        if UNMASKED:
            x1 = tl.load(
                x_ptr + base,
                mask=rmask[:, None],
                other=0.0,
                eviction_policy=evict,
            ).to(tl.float32)
            x3 = tl.load(
                x_ptr + base + d,
                mask=rmask[:, None],
                other=0.0,
                eviction_policy=evict,
            ).to(tl.float32)
        else:
            x1 = tl.load(
                x_ptr + base, mask=mask, other=0.0, eviction_policy=evict
            ).to(tl.float32)
            x3 = tl.load(
                x_ptr + base + d, mask=mask, other=0.0, eviction_policy=evict
            ).to(tl.float32)

        # gelu_tanh(v) = v * sigmoid(v * (+2*sqrt(2/pi) + (+2*sqrt(2/pi)*0.044715) * v^2)),
        # the sigmoid identity of 0.5*v*(1 + tanh(sqrt(2/pi)*(v + 0.044715*v^3))).
        # Self-saturating with no branch: the denominator saturates for large
        # |t| and argument 0 (masked lanes) gives exactly 0.5.
        t = x1 * (1.5957691216057308 + 0.07133681239524435 * x1 * x1)
        y = x1 * x3 * tl.sigmoid(t)

        tl.store(
            out_ptr + r[:, None] * d + cols[None, :],
            y.to(out_ptr.dtype.element_ty),
            mask=rmask[:, None] & cmask[None, :],
            eviction_policy=evict,
        )


@triton.jit
def _gelu_tanh_and_mul_i64_kernel(
    x_ptr,
    out_ptr,
    m,
    d: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
):
    # int64-indexed variant for inputs above INT32_MAX elements. One row block
    # per program; the grid is sized by the launcher so no loop is needed.
    UNMASKED: tl.constexpr = BLOCK_N == d
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < m
    cols = tl.arange(0, BLOCK_N)
    cmask = cols < d
    base = rows.to(tl.int64)[:, None] * (2 * d) + cols[None, :]
    mask = rmask[:, None] & cmask[None, :]

    if UNMASKED:
        x1 = tl.load(x_ptr + base, mask=rmask[:, None], other=0.0).to(
            tl.float32
        )
        x3 = tl.load(x_ptr + base + d, mask=rmask[:, None], other=0.0).to(
            tl.float32
        )
    else:
        x1 = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        x3 = tl.load(x_ptr + base + d, mask=mask, other=0.0).to(tl.float32)

    # Sigmoid-form gate, as in the i32 kernel.
    t = x1 * (1.5957691216057308 + 0.07133681239524435 * x1 * x1)
    y = x1 * x3 * tl.sigmoid(t)

    tl.store(
        out_ptr + rows.to(tl.int64)[:, None] * d + cols[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=rmask[:, None] & cmask[None, :],
        eviction_policy="evict_first",
    )


def gelu_tanh_and_mul(input):
    two_d = input.shape[-1]
    d = two_d // 2
    x = input.contiguous()
    out = torch.empty(
        input.shape[:-1] + (d,), dtype=input.dtype, device=input.device
    )
    m = out.numel() // d if d > 0 else 0
    if m == 0 or d == 0:
        return out

    block_n = triton.next_power_of_2(d)
    if x.numel() <= _INT32_MAX:
        _gelu_tanh_and_mul_i32_kernel[
            (lambda META: (min(triton.cdiv(m, META["ROWS"]), META["P"]),))
        ](x, out, m, d, block_n)
    else:
        # int64 variant: fixed 8-row tiles, grid covers all row blocks.
        grid = (triton.cdiv(m, 8),)
        _gelu_tanh_and_mul_i64_kernel[grid](x, out, m, d, block_n, 8)
    return out


__all__ = ["gelu_tanh_and_mul"]
