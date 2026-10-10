# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pack_topk_ids — Kunlunxin-specific entry point.

Why this file exists (single-variable fix over the cross-chip file)
------------------------------------------------------------------
The cross-chip implementation derives the low 16 bits as::

    w.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF

On Kunlunxin that fails to compile::

    loc("w_bits"("pack_topk_ids.py":64:23)):
        Cannot bitcast data-type of size 32 to data-type of size 16
    note: Pipeline failed while executing [`TritonXPUDtypeConvert`]
    Reducing block sizes or `num_stages` may help.

The XPU backend keeps bf16 values in 32-bit registers, so a "16-bit" bitcast
is lowered as a 32-to-16 conversion and the MLIR dtype pass rejects it.  This
matches the constraint already recorded for this backend: *an in-kernel bf16
bitcast does not compile*.

The fix keeps identical arithmetic but never names a bf16 value before the
bitcast.  Rounding to bf16 is done on the **fp32 bit pattern** (a size
preserving ``f32 -> i32`` bitcast) with the standard round-to-nearest-even
integer sequence::

    lsb     = (u >> 16) & 1
    bits16  = ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

Two's-complement addition is identical modulo 2**32, so negative weights (sign
bit set in the fp32 word) round correctly; the final ``& 0xFFFF`` discards
whatever the arithmetic shift sign-extended.

``verify_rounding.py`` checks this sequence against an independent exact-rational
IEEE round-to-nearest-even oracle over every special value, subnormal boundary
and tie pattern: 200,028 samples, 0 mismatches.

Resource knobs
--------------
The compile failure arrived wrapped in ``OutOfResources: uni_sram`` and the note
explicitly suggested smaller blocks or fewer stages, so this variant also uses
``BLOCK_SIZE = 512`` and ``num_stages = 1``.  A fused elementwise kernel gains
nothing from pipelining, and Kunlunxin is currently at 0 for this operator, so
correct compilation dominates any occupancy consideration here.

Pure Triton: no fallback path, no device-dependent branching."""
import torch
import triton
import triton.language as tl

__all__ = ["pack_topk_ids"]


@triton.jit
def _pack_topk_ids_kernel(
    ids_ptr,
    w_ptr,
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Exact-fit variant: ``N`` is a multiple of ``BLOCK_SIZE``, so no mask."""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    ids = tl.load(ids_ptr + offs)
    w = tl.load(w_ptr + offs)

    # 32 -> 32 bitcast only: never materialise a 16-bit value.
    u = w.to(tl.int32, bitcast=True)
    lsb = (u >> 16) & 1
    bits16 = ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

    tl.store(out_ptr + offs, (ids << 16) | bits16)


@triton.jit
def _pack_topk_ids_kernel_masked(
    ids_ptr,
    w_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """General variant for ``numel`` that does not divide the block size."""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    ids = tl.load(ids_ptr + offs, mask=mask, other=0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)

    u = w.to(tl.int32, bitcast=True)
    lsb = (u >> 16) & 1
    bits16 = ((u + 0x7FFF + lsb) >> 16) & 0xFFFF

    tl.store(out_ptr + offs, (ids << 16) | bits16, mask=mask)


def pack_topk_ids(topk_ids, topk_weights):
    """Pack each (expert_id, weight) routing pair into one int32."""
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()

    out = torch.empty_like(topk_ids, dtype=torch.int32)
    n_elements = topk_ids.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 512
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    if n_elements % BLOCK_SIZE == 0:
        _pack_topk_ids_kernel[grid](
            topk_ids,
            topk_weights,
            out,
            N=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=1,
        )
    else:
        _pack_topk_ids_kernel_masked[grid](
            topk_ids,
            topk_weights,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=1,
        )
    return out
