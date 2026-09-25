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

"""tanh-approximated gated GELU -- Hygon DCU specialization.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _libdevice

# -2 * sqrt(2/pi) * log2(e): the fused constant of the sigmoid-as-exp2 fold.
_K2 = tl.constexpr(-2.302208198144325)
_COEFF = tl.constexpr(0.044715)


@triton.jit
def _gelu_tanh_and_mul_kernel(
    x_ptr,
    out_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    n_cb: tl.constexpr = (D + BLOCK_D - 1) // BLOCK_D
    colb = pid % n_cb
    row = (pid // n_cb).to(tl.int64)

    offs = colb * BLOCK_D + tl.arange(0, BLOCK_D)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK_D), BLOCK_D)
    row_off = row * (2 * D)

    if D % BLOCK_D == 0:
        x1 = tl.load(x_ptr + row_off + offs).to(tl.float32)
        x3 = tl.load(x_ptr + row_off + D + offs).to(tl.float32)
    else:
        mask = offs < D
        x1 = tl.load(x_ptr + row_off + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        x3 = tl.load(x_ptr + row_off + D + offs, mask=mask, other=0.0).to(
            tl.float32
        )

    # gelu_tanh(v) = v * sigmoid(-2 * sqrt(2/pi) * log2(e) * (v + 0.044715 v^3))
    # evaluated through exp2, fp32.
    k = _K2 * (x1 + _COEFF * x1 * x1 * x1)
    sig = _libdevice.fast_dividef(1.0, 1.0 + tl.exp2(k))
    y = x1 * sig * x3

    if D % BLOCK_D == 0:
        tl.store(out_ptr + row * D + offs, y.to(out_ptr.dtype.element_ty))
    else:
        tl.store(
            out_ptr + row * D + offs, y.to(out_ptr.dtype.element_ty), mask=mask
        )


def _pick_config(n_rows, d):
    """Empirical best (BLOCK_D, num_warps) per size class.

    Small problems ride the platform's launch/allocation floor and any config
    below ~8.5k elements/thread-width ties, so they get one narrow wave.
    Mid-size working sets (up to ~2M output elements) gain memory-level
    parallelism from more elements per thread — narrow blocks at 2 warps
    (2048-wide @ 2 warps = 64 elements/thread) win there, while the very
    wide 8192-column rows prefer 4 warps for more concurrent CTAs. The
    largest streaming shapes saturate DRAM at BLOCK_D=1024 / 2 warps
    (~1250 GB/s effective on 8192-column rows); 1024/4 is measurably better
    for d<=1024 where a second CTA per row keeps the wave full.
    """
    if n_rows <= 256:
        return 512, 2
    if n_rows <= 2048:
        if d <= 2048:
            return 1024, 4
        if d <= 4096:
            return 2048, 2
        return 1024, 4
    if d <= 1024:
        return 1024, 4
    return 1024, 2


def gelu_tanh_and_mul(input):
    d = input.shape[-1] // 2
    n_rows = input.numel() // (2 * d) if d > 0 else 0
    out = torch.empty(
        input.shape[:-1] + (d,), dtype=input.dtype, device=input.device
    )
    if n_rows == 0 or d == 0:
        return out
    if not input.is_contiguous():
        input = input.contiguous()

    block_d, num_warps = _pick_config(n_rows, d)
    n_cb = (d + block_d - 1) // block_d
    _gelu_tanh_and_mul_kernel[(n_rows * n_cb,)](
        input, out, d, block_d, num_warps=num_warps
    )
    return out


__all__ = ["gelu_tanh_and_mul"]
