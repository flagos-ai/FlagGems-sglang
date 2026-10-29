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

"""Fused RMS normalization with per-batch scale and shift."""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": 1}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_R": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_R": 1}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_R": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_R": 2}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_R": 2}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_R": 2}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_R": 2}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_R": 4}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_R": 4}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_R": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_R": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_R": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_R": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_R": 8}, num_warps=8, num_stages=3),
    ],
    key=["B", "S", "D"],
)
@triton.jit
def _rmsnorm_scale_shift_kernel(
    x_ptr,  # [B*S, D]  (flattened rows)
    w_ptr,  # [D]
    scale_ptr,  # [B, D]    (flattened from [B, 1, D])
    shift_ptr,  # [B, D]
    out_ptr,  # [B*S, D]
    B,  # batch size (autotune key only)
    S: tl.constexpr,  # sequence length (rows per batch element)
    D: tl.constexpr,  # hidden size (power of 2 on all real shapes)
    eps,
    BLOCK_R: tl.constexpr,  # rows per program
    WIDE: tl.constexpr,  # True when offsets need int64 arithmetic
):
    b = tl.program_id(1)
    c = tl.program_id(0)

    cols = tl.arange(0, D)
    rids = c * BLOCK_R + tl.arange(0, BLOCK_R)

    # Compile-time row-mask elision: exact when BLOCK_R divides S.
    EVEN_R: tl.constexpr = S % BLOCK_R == 0

    if WIDE:
        off = (b.to(tl.int64) * S + rids)[:, None] * D + cols[None, :]
        moff = b.to(tl.int64) * D + cols
    else:
        off = (b * S + rids)[:, None] * D + cols[None, :]
        moff = b * D + cols

    # Issue the wide x tile first so its long-latency gather overlaps the
    # narrow (weight, scale, shift) vector loads that follow.
    if EVEN_R:
        x = tl.load(x_ptr + off).to(tl.float32)
    else:
        x = tl.load(x_ptr + off, mask=(rids < S)[:, None], other=0.0).to(
            tl.float32
        )

    # RMS statistics in fp32 (masked lanes contribute 0 to the sum).
    var = tl.sum(x * x, axis=1) / D
    rstd = tl.math.rsqrt(var + eps)

    # One (weight, scale, shift) triple per program, shared by all rows.
    w = tl.load(w_ptr + cols).to(tl.float32)
    s = tl.load(scale_ptr + moff) + 1.0
    sh = tl.load(shift_ptr + moff)

    normed = (x * rstd[:, None] * w[None, :]).to(out_ptr.dtype.element_ty)

    y = tl.math.fma(normed.to(tl.float32), s[None, :], sh[None, :]).to(
        out_ptr.dtype.element_ty
    )

    if EVEN_R:
        tl.store(out_ptr + off, y)
    else:
        tl.store(out_ptr + off, y, mask=(rids < S)[:, None])


def rmsnorm_scale_shift(x, weight, scale, shift, eps):
    B, S, D = x.shape

    # Core compute stays in Triton; contiguous() only normalizes layout.
    xc = x if x.is_contiguous() else x.contiguous()
    wc = weight if weight.is_contiguous() else weight.contiguous()
    sc = scale if scale.is_contiguous() else scale.contiguous()
    shc = shift if shift.is_contiguous() else shift.contiguous()

    out = torch.empty_like(xc)
    if B * S == 0:
        return out

    wide = xc.numel() >= 2**31

    # Row tile on grid axis 0 (fastest-varying across scheduled programs)
    # so consecutive programs stream adjacent rows of one batch element.
    def grid(meta):
        return (triton.cdiv(S, meta["BLOCK_R"]), B)

    _rmsnorm_scale_shift_kernel[grid](
        xc,
        wc,
        sc,
        shc,
        out,
        B,
        S,
        D,
        eps,
        WIDE=wide,
    )
    return out


__all__ = ["rmsnorm_scale_shift"]
