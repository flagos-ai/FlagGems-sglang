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

"""pack_topk_ids — fused (expert_id, weight) packing into one int32.

Contract (exact, integer)
-------------------------
    out = (topk_ids << 16) | (bf16_bits(topk_weights) & 0xFFFF)

where ``bf16_bits`` is the *bit pattern* of the round-to-nearest-even bf16
value, zero-extended - not a fixed-point rescale.  ``topk_ids`` is int32 and
``topk_weights`` is fp32; both are contiguous and share a shape; the output is
int32 with the same shape.

Why one fused kernel wins
-------------------------
The reference expression materialises several intermediates::

    (topk_weights.to(torch.bfloat16)          # fp32 read, bf16 write
     .view(torch.int16)                       # view (free)
     .to(torch.int32))                        # bf16 read, int32 write
    & 0xFFFF                                  # int32 read+write
    (topk_ids.to(torch.int32) << 16)          # int32 read, shift, write
    ... | ...                                 # read+read, write

so the eager path moves roughly six element-passes plus temporaries.  This
kernel reads the two inputs once and writes the output once: two reads and one
write, the same shape of win that ``add3`` gets, but with an *exact* integer
contract instead of a tolerance.  Three passes is the floor for this operator,
so the remaining headroom is launch and occupancy tuning, not fewer passes.

Bit trick used here
-------------------
``topk_ids << 16`` needs the low 16 bits of the result to be exactly the bf16
pattern.  We obtain it by casting the fp32 weight to bf16 (hardware RNE) and
reinterpreting the 16 bits, then zero-extending so the OR cannot pollute the
high half.

Mask-free fast path
-------------------
The task shapes make ``numel`` an exact multiple of any power-of-two block we
would pick, so ``N`` is passed as a ``constexpr`` and the kernel is launched
without any bounds mask.  ``tl.arange`` then covers whole blocks and every
program is branch-free.  If a shape ever does not divide evenly, the caller
falls back to the masked variant, which is the same arithmetic with a guard -
a compile-time choice, never a runtime device branch, and never a PyTorch
fallback.

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

    # fp32 -> bf16 with hardware RNE, then reinterpret the 16 raw bits.
    # Mirrors the reference exactly: ``.view(int16).to(int32) & 0xFFFF``.
    # int16 -> int32 in Triton sign-extends, so the mask is required; dropping
    # it would leak set bits into the high half for weights whose bf16 sign
    # bit is 1 (i.e. every negative weight).
    w = tl.load(w_ptr + offs)
    w_bits = w.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF

    tl.store(out_ptr + offs, (ids << 16) | w_bits)


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
    w_bits = w.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF

    tl.store(out_ptr + offs, (ids << 16) | w_bits, mask=mask)


def pack_topk_ids(topk_ids, topk_weights):
    """Pack each (expert_id, weight) routing pair into one int32."""
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.contiguous()

    out = torch.empty_like(topk_ids, dtype=torch.int32)
    n_elements = topk_ids.numel()
    if n_elements == 0:
        return out

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    if n_elements % BLOCK_SIZE == 0:
        _pack_topk_ids_kernel[grid](
            topk_ids,
            topk_weights,
            out,
            N=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
    else:
        _pack_topk_ids_kernel_masked[grid](
            topk_ids,
            topk_weights,
            out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
    return out
