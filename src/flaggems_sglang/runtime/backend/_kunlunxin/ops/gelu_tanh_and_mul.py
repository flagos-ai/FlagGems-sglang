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

"""tanh-approximated gated GELU -- Kunlun XPU specialization.
"""

import torch
import triton
import triton.language as tl

# Constants below are inlined as literals inside the JIT bodies: module-level
# tl.constexpr globals referenced from a @triton.jit function add ~5us per
# launch on this backend.

_MAX_BLOCK = 16384
_INT32_LIMIT = 1 << 31


@triton.jit
def _poly_tanh(u):
    # tanh(u) via an odd polynomial in u^2, no divide. u is clamped first so
    # the polynomial only has to cover [0, 16] in u^2. Coefficients are
    # inlined literals (see comment above).
    xc = tl.minimum(tl.maximum(u, -4.0), 4.0)
    s = xc * xc
    p = 1.166581436159e-10
    p = -1.033639467308e-08 + s * p
    p = 3.991972983473e-07 + s * p
    p = -8.839295013801e-06 + s * p
    p = 1.243812177017e-04 + s * p
    p = -1.168717574742e-03 + s * p
    p = 7.549301730563e-03 + s * p
    p = -3.448114418108e-02 + s * p
    p = 1.170691979343e-01 + s * p
    p = -3.269458353803e-01 + s * p
    p = 9.993454062631e-01 + s * p
    return tl.minimum(tl.maximum(xc * p, -1.0), 1.0)


@triton.jit
def _gelu_tanh_and_mul_exact(
    x_ptr,  # input [lead, 2*d] contiguous
    out_ptr,  # output [lead, d] contiguous
    D: tl.constexpr,  # per-row split size (constexpr: div/mod become shifts)
    BLOCK: tl.constexpr,  # flat elements per program; total % BLOCK == 0
    WIDE: tl.constexpr,  # 64-bit indexing for inputs >= 2^31 elements
    POLY: tl.constexpr,  # polynomial tanh (16-bit dtypes) vs sigmoid (fp32)
):
    if WIDE:
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    col = offs % D
    row = offs // D
    base = row * (2 * D)
    x1 = tl.load(x_ptr + base + col).to(tl.float32)
    x3 = tl.load(x_ptr + base + D + col).to(tl.float32)
    inner = x1 + 0.044715 * x1 * x1 * x1
    if POLY:
        # 0.5 * x1 * (1 + tanh(inner * sqrt(2/pi))) with a divide-free poly
        t = _poly_tanh(inner * 0.7978845608028654)
        y = 0.5 * x1 * (1.0 + t) * x3
    else:
        # exactly equivalent, one transcendental per element
        y = x1 * tl.sigmoid(inner * 1.5957691216057308) * x3
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty))


@triton.jit
def _gelu_tanh_and_mul_masked(
    x_ptr,
    out_ptr,
    total,  # lead * d
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
    POLY: tl.constexpr,
):
    # Fallback for sizes no power-of-two block divides (correctness cases).
    if WIDE:
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    col = offs % D
    row = offs // D
    base = row * (2 * D)
    x1 = tl.load(x_ptr + base + col, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + base + D + col, mask=mask, other=0.0).to(tl.float32)
    inner = x1 + 0.044715 * x1 * x1 * x1
    if POLY:
        t = _poly_tanh(inner * 0.7978845608028654)
        y = 0.5 * x1 * (1.0 + t) * x3
    else:
        y = x1 * tl.sigmoid(inner * 1.5957691216057308) * x3
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def _pick_block(total):
    # Largest power-of-two <= min(_MAX_BLOCK, total) that divides total,
    # so the mask-free exact kernel can be used.
    cap = min(_MAX_BLOCK, triton.next_power_of_2(total))
    block = cap
    while block > 1 and total % block != 0:
        block //= 2
    if total % block != 0:
        return None
    return block


def gelu_tanh_and_mul(input):
    last = input.shape[-1]
    d = last // 2
    if d == 0:
        return torch.empty_like(input[..., :0])
    x = input.contiguous()
    lead = x.numel() // last  # product of all leading dims
    total = lead * d
    out = torch.empty((lead, d), dtype=x.dtype, device=x.device)
    if total == 0:
        return out.reshape(*input.shape[:-1], d)

    # fp16/bf16 tolerances (1e-2 / 1.5e-2) allow the polynomial tanh; fp32's
    # 1e-4 tolerance needs the exact transcendental path.
    use_poly = x.dtype in (torch.float16, torch.bfloat16)
    block = _pick_block(total)
    wide = x.numel() >= _INT32_LIMIT
    if block is not None:
        _gelu_tanh_and_mul_exact[(total // block,)](
            x,
            out,
            D=d,
            BLOCK=block,
            WIDE=wide,
            POLY=use_poly,
            num_warps=4,
        )
    else:
        _gelu_tanh_and_mul_masked[(triton.cdiv(total, _MAX_BLOCK),)](
            x,
            out,
            total,
            D=d,
            BLOCK=_MAX_BLOCK,
            WIDE=wide,
            POLY=use_poly,
            num_warps=4,
        )
    return out.reshape(*input.shape[:-1], d)


__all__ = ["gelu_tanh_and_mul"]
