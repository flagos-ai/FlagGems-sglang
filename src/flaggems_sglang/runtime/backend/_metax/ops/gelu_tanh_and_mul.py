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

"""tanh-approximated gated GELU -- MetaX specialization.
"""

import torch
import triton
import triton.language as tl

_SQRT_2_OVER_PI = tl.constexpr(0.7978845608028654)
_COEFF = tl.constexpr(0.044715)
_LOG2E = tl.constexpr(1.4426950408889634)


@triton.autotune(
    configs=[
        # Narrow blocks with 1-2 warps. The small shapes are CTA-launch bound,
        # so a small CTA keeps the per-CTA start latency down; this region owns
        # every bs1 / bs8 case.
        triton.Config({"BLOCK_D": 256, "RPC": 1}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_D": 256, "RPC": 1}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_D": 256, "RPC": 2}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_D": 256, "RPC": 2}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_D": 256, "RPC": 4}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_D": 512, "RPC": 1}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_D": 512, "RPC": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512, "RPC": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 512, "RPC": 4}, num_warps=4, num_stages=2),
        # Mid blocks: the bs64 family, where a few extra rows per program
        # remove the last of the launch tail without lengthening the CTA.
        triton.Config({"BLOCK_D": 1024, "RPC": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1024, "RPC": 1}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 1024, "RPC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1024, "RPC": 2}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_D": 1024, "RPC": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 1024, "RPC": 4}, num_warps=8, num_stages=2),
        # Wide blocks for the large streaming shapes, where 4-8 warps keeps
        # enough loads in flight per CTA to reach the copy roofline. RPC>1 with
        # stages>1 is the part the runtime row loop makes pipelinable.
        triton.Config({"BLOCK_D": 2048, "RPC": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 2048, "RPC": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 2048, "RPC": 2}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_D": 2048, "RPC": 2}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 2048, "RPC": 2}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_D": 2048, "RPC": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 4096, "RPC": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 4096, "RPC": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 4096, "RPC": 2}, num_warps=8, num_stages=1),
    ],
    key=["D", "NROWS"],
)
@triton.jit
def _gelu_tanh_and_mul_kernel(
    x_ptr,
    out_ptr,
    D: tl.constexpr,
    NROWS,
    BLOCK_D: tl.constexpr,
    RPC: tl.constexpr,
):
    # Decided by the compiler for this (D, BLOCK_D) pair: when BLOCK_D tiles D
    # the column mask is provably all-true and is dropped entirely; otherwise
    # this same config compiles a masked variant rather than being illegal.
    EVEN_D: tl.constexpr = (D % BLOCK_D) == 0
    BLOCKS_PER_ROW: tl.constexpr = (D + BLOCK_D - 1) // BLOCK_D

    pid = tl.program_id(0)
    row0 = (pid // BLOCKS_PER_ROW) * RPC
    blk = pid % BLOCKS_PER_ROW

    cols = blk * BLOCK_D + tl.arange(0, BLOCK_D)

    # Runtime loop: an scf.for the backend can software-pipeline with
    # num_stages, rather than an unrolled straight-line body.
    for i in range(RPC):
        row = row0 + i
        # Only the trailing row group can run past NROWS; the guard is a
        # uniform branch, so the in-range path keeps its unmasked loads.
        if row < NROWS:
            base = row * (2 * D)
            if EVEN_D:
                x1 = tl.load(
                    x_ptr + base + cols, eviction_policy="evict_first"
                )
                x3 = tl.load(
                    x_ptr + base + D + cols, eviction_policy="evict_first"
                )
            else:
                mask = cols < D
                x1 = tl.load(
                    x_ptr + base + cols,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                x3 = tl.load(
                    x_ptr + base + D + cols,
                    mask=mask,
                    other=0.0,
                    eviction_policy="evict_first",
                )

            x1 = x1.to(tl.float32)
            x3 = x3.to(tl.float32)

            # gelu_tanh(v) = v * sigmoid(2 * sqrt(2/pi) * (v + 0.044715 * v^3)),
            # sigmoid(z) = exp2(-log2(1 + 2^(-z * log2(e)))):
            # no fp32 division, just the fast exp2/log2 units.
            inner = _SQRT_2_OVER_PI * (x1 + _COEFF * x1 * x1 * x1)
            e = tl.math.exp2(-2.0 * _LOG2E * inner)
            gate = x1 * tl.math.exp2(-tl.math.log2(1.0 + e))
            val = (gate * x3).to(out_ptr.dtype.element_ty)

            if EVEN_D:
                tl.store(
                    out_ptr + row * D + cols,
                    val,
                    eviction_policy="evict_first",
                )
            else:
                tl.store(
                    out_ptr + row * D + cols,
                    val,
                    mask=cols < D,
                    eviction_policy="evict_first",
                )


def gelu_tanh_and_mul(input):
    shape = input.shape
    d = shape[-1] // 2
    out = torch.empty(
        shape[:-1] + (d,), dtype=input.dtype, device=input.device
    )
    if d == 0 or input.numel() == 0:
        return out

    inp = input if input.is_contiguous() else input.contiguous()
    rows = inp.numel() // (2 * d)

    def grid(meta):
        return (
            triton.cdiv(rows, meta["RPC"]) * triton.cdiv(d, meta["BLOCK_D"]),
        )

    _gelu_tanh_and_mul_kernel[grid](inp, out, d, rows)
    return out


__all__ = ["gelu_tanh_and_mul"]
