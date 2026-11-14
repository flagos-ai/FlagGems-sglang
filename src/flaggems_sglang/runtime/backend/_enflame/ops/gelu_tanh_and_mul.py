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

"""tanh-approximated gated GELU -- Enflame GCU specialization.

Flat 1-D streaming rewrite. The previous [ROWS, BLOCK_N] 2-D tile layout
(rank-2 masked load/store + same-base offset pair) can no longer be freshly
compiled on the 2026-11-07 toolchain (memref_ext.slice_pad_start
operandSegmentSizes 8 vs 7, see compiler_issues/vllm_silu_flat_slice_start.md),
and its one-tile-per-CTA dispatch left the large shapes ~3x off the flat
streaming roofline anyway.

The kernel walks the contiguous *output* space; a single ``offs // D`` recovers
the row and ``base = offs + row * D`` addresses the gate half, with ``base + D``
hitting the up half -- both contiguous BLOCK-wide spans, so every load/store is
pure 1-D. The gate value is computed from the tanh identity
``gelu_tanh(v) = v * sigmoid(v * (2*sqrt(2/pi) + 2*sqrt(2/pi)*0.044715*v^2))``
(one native sigmoid, no extern libm). The up half is loaded only after the
gate is ready (probe-verified -8~-11% on the bandwidth-bound shapes).

Launch geometry is dispatched statically (no autotuner wrapper on the hot
path): a persistent grid capped at the 48 GCU SPs iterates BLOCK-sized tiles
grid-stride style for large inputs, while small inputs ride a single wave.
BLOCK/num_warps stay inside the per-warp <=2048-element / per-CTA <=32KB
slice bounds, and BLOCK is additionally capped at the largest power of two
<= 2*D: blocks spanning 3+ D-segments (BLOCK > 2*D) silently compute garbage
on this toolchain, as does BLOCK=16384@num_warps=1.
"""

import torch
import triton
import triton.language as tl

_INT32_MAX = 2**31 - 1

# GCU400 exposes 48 simultaneously resident programs ("Fixed Grid 48");
# torch_device_fn.get_device_properties().multi_processor_count reports 4,
# so the SP count cannot be queried and stays a constant.
_PERSISTENT_CTAS = 48

# gelu_tanh(v) = v * sigmoid(v * (2*sqrt(2/pi) + 2*sqrt(2/pi)*0.044715*v^2)),
# the sigmoid identity of 0.5*v*(1 + tanh(sqrt(2/pi)*(v + 0.044715*v^3))).
# Self-saturating with no branch: the sigmoid saturates for large |v| and
# argument 0 (masked lanes) gives exactly 0.5.
_GELU_TANH_A = 1.5957691216057308  # 2*sqrt(2/pi)
_GELU_TANH_B = 0.07133681239524435  # 2*sqrt(2/pi)*0.044715


@triton.jit
def _gelu_tanh_and_mul_kernel(
    x_ptr,
    out_ptr,
    n,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for start in range(pid * BLOCK, n, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        # Flat output-space addressing: base == row * 2D + col for the gate
        # half and base + D for the up half (each a contiguous BLOCK span).
        row = offs // D
        base = offs + row * D
        if EVEN:
            x1 = tl.load(x_ptr + base).to(tl.float32)
            t = x1 * (1.5957691216057308 + 0.07133681239524435 * x1 * x1)
            gate = x1 * tl.sigmoid(t)
            x3 = tl.load(x_ptr + base + D).to(tl.float32)
            tl.store(out_ptr + offs, (gate * x3).to(out_ptr.dtype.element_ty))
        else:
            msk = offs < n
            x1 = tl.load(x_ptr + base, mask=msk, other=0.0).to(tl.float32)
            t = x1 * (1.5957691216057308 + 0.07133681239524435 * x1 * x1)
            gate = x1 * tl.sigmoid(t)
            x3 = tl.load(x_ptr + base + D, mask=msk, other=0.0).to(tl.float32)
            tl.store(
                out_ptr + offs,
                (gate * x3).to(out_ptr.dtype.element_ty),
                mask=msk,
            )


@triton.jit
def _gelu_tanh_and_mul_i64_kernel(
    x_ptr,
    out_ptr,
    n,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # int64-indexed variant for inputs above INT32_MAX elements; same flat
    # scheme with 64-bit offsets, persistent grid-stride over the 48 SPs.
    pid = tl.program_id(0).to(tl.int64)
    nprog = tl.num_programs(0).to(tl.int64)
    for start in range(pid * BLOCK, n, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK).to(tl.int64)
        row = offs // D
        base = offs + row * D
        if EVEN:
            x1 = tl.load(x_ptr + base).to(tl.float32)
            t = x1 * (1.5957691216057308 + 0.07133681239524435 * x1 * x1)
            gate = x1 * tl.sigmoid(t)
            x3 = tl.load(x_ptr + base + D).to(tl.float32)
            tl.store(out_ptr + offs, (gate * x3).to(out_ptr.dtype.element_ty))
        else:
            msk = offs < n
            x1 = tl.load(x_ptr + base, mask=msk, other=0.0).to(tl.float32)
            t = x1 * (1.5957691216057308 + 0.07133681239524435 * x1 * x1)
            gate = x1 * tl.sigmoid(t)
            x3 = tl.load(x_ptr + base + D, mask=msk, other=0.0).to(tl.float32)
            tl.store(
                out_ptr + offs,
                (gate * x3).to(out_ptr.dtype.element_ty),
                mask=msk,
            )


def _flat_config(n, itemsize, d):
    # Static dispatch (probe-tuned on GCU400, do_bench median, kernel mode):
    # inputs up to 8192 elements are launch-bound and want the narrowest
    # single wave at num_warps=2; larger shapes run persistent BLOCK=8192@nw4
    # over 48 CTAs. BLOCK/num_warps keep per-warp <=2048 elements and
    # per-CTA <=32KB for every dtype width.
    #
    # Miscompile guard: a block whose address pattern spans more than 2
    # D-segments (BLOCK > 2*D) silently computes garbage on this toolchain
    # (compiler_issues/vllm_silu_flat_slice_start.md), so BLOCK is capped at
    # the largest power of two <= 2*D.
    seg2_cap = 1 << ((2 * d).bit_length() - 1)
    if itemsize <= 2:
        wide = min(8192, seg2_cap)
        if n <= 8192:
            return min(triton.next_power_of_2(max(n, 256)), 2048, wide), 2
        return min(triton.next_power_of_2(n), wide), 4
    return min(triton.next_power_of_2(n), 4096, seg2_cap), 4


def gelu_tanh_and_mul(input):
    two_d = input.shape[-1]
    d = two_d // 2
    x = input.contiguous()
    out = torch.empty(
        input.shape[:-1] + (d,), dtype=input.dtype, device=input.device
    )
    n = out.numel()
    if n == 0 or d == 0:
        return out

    if x.numel() <= _INT32_MAX:
        block, warps = _flat_config(n, x.element_size(), d)
        grid = min(_PERSISTENT_CTAS, triton.cdiv(n, block))
        _gelu_tanh_and_mul_kernel[(grid,)](
            x, out, n, d, block, (n % block == 0), num_warps=warps
        )
    else:
        block, _ = _flat_config(n, x.element_size(), d)
        block = min(block, 4096)
        _gelu_tanh_and_mul_i64_kernel[(_PERSISTENT_CTAS,)](
            x, out, n, d, block, (n % block == 0), num_warps=4
        )
    return out


__all__ = ["gelu_tanh_and_mul"]
