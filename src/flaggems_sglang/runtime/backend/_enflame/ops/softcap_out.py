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

# L600 / GCU400 stream processors: the persistent launch is capped at one
# program per SP (DSM per program stays <= 448 KB so grid=48 keeps true
# 48-way concurrency).
_NUM_SPC = 48


@triton.jit
def _softcap_out_enflame_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    two_inv_cap: tl.constexpr,
    two_cap: tl.constexpr,
    cap: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flat persistent softcap: ``out = cap * tanh(x / cap)`` in FP32.

    ``tanh(z) = 2 * sigmoid(2z) - 1`` keeps the hot loop on the native
    ``tops_sigmoid_v2_fp32`` unit (libdevice tanh is an extern call, ~2.1x
    slower end to end), and the constants are folded on the host so the
    loop body is ``two_cap * sigmoid(x * two_inv_cap) - cap``.

    Tiling uses ``tl.make_block_ptr`` with ``boundary_check`` and a
    constexpr ``n_elements``: the block_ptr form streams the bf16 -> fp32
    mixed-width traffic at ~2.95 TB/s (vs ~1.99 TB/s for the previous
    masked arange form, E-1 ladder 2026-11-12), matching the pure
    load+store floor within 3%.  ``n_elements`` must stay constexpr
    because runtime-shaped block_ptr boundary checks trip the
    ``memref_ext.slice_pad_start`` operandSegmentSizes toolchain bug
    (see softcap_inplace_logits); each shape pays one JIT compile.
    """
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    nblocks = tl.cdiv(n_elements, BLOCK)
    for bid in tl.range(pid, nblocks, nprog):
        xblk = tl.make_block_ptr(
            input_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(bid * BLOCK,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        yblk = tl.make_block_ptr(
            output_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(bid * BLOCK,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        x = tl.load(xblk, boundary_check=(0,)).to(tl.float32)
        y = two_cap * tl.sigmoid(x * two_inv_cap) - cap
        tl.store(yblk, y, boundary_check=(0,))


def _pick_block(n_elements: int):
    """Measured dispatch tiers (L600, bf16 in / fp32 out, E-1 2026-11-12).

    n >= 4M  -> BLOCK=65536 (512x128256: 133us = 2.95 TB/s)
    n >= 96K -> BLOCK=8192  (8x128256: 6.97us, 64x32000: 10.3us)
    else     -> BLOCK=2048  (launch-floor band; nw=2 beats nw=4)
    32768-wide tiles measured strictly worse than both neighbours at every
    tier, and num_stages in {1,2,3} is flat, so stages stays 1.
    """
    if n_elements >= 4194304:
        return 65536
    if n_elements >= 98304:
        return 8192
    return 2048


def softcap_out(x, softcap_const, autotune=False):
    """Compute ``softcap * tanh(x / softcap)`` and return an FP32 tensor."""
    if not x.is_contiguous():
        x = x.contiguous()

    output = torch.empty_like(x, dtype=torch.float32)
    n_elements = output.numel()
    if n_elements == 0:
        return output

    cap = float(softcap_const)
    block = _pick_block(n_elements)
    grid_size = min(_NUM_SPC, triton.cdiv(n_elements, block))

    _softcap_out_enflame_kernel[(grid_size,)](
        x,
        output,
        n_elements=n_elements,
        two_inv_cap=2.0 / cap,
        two_cap=2.0 * cap,
        cap=cap,
        BLOCK=block,
        num_warps=2 if n_elements < 98304 else 4,
        num_stages=1,
    )
    return output


__all__ = ["softcap_out"]
