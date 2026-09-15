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

import triton
import triton.language as tl
from triton.language.extra import libdevice

MAX_BLOCK = 32768
MAX_PROGRAMS = 12
NUM_WARPS = 4


@triton.jit
def _direct(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    inv_cap: tl.constexpr,
    cap: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_offset = block_id * BLOCK
    x_blk = tl.make_block_ptr(
        x_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(block_offset,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    out_blk = tl.make_block_ptr(
        out_ptr,
        shape=(n_elements,),
        strides=(1,),
        offsets=(block_offset,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    x = tl.load(x_blk, boundary_check=(0,)).to(tl.float32)
    y = libdevice.tanh(x * inv_cap) * cap
    tl.store(out_blk, y, boundary_check=(0,))


@triton.jit
def _persist(
    x_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    inv_cap: tl.constexpr,
    cap: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    nblocks = tl.cdiv(n_elements, BLOCK)
    for bid in tl.range(pid, nblocks, nprog):
        block_offset = bid * BLOCK
        x_blk = tl.make_block_ptr(
            x_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_offset,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        out_blk = tl.make_block_ptr(
            out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_offset,),
            block_shape=(BLOCK,),
            order=(0,),
        )
        x = tl.load(x_blk, boundary_check=(0,)).to(tl.float32)
        y = libdevice.tanh(x * inv_cap) * cap
        tl.store(out_blk, y, boundary_check=(0,))


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    cap = float(final_logit_softcapping)
    inv_cap = 1.0 / cap
    x = (
        full_logits
        if full_logits.is_contiguous()
        else full_logits.contiguous()
    )
    n = int(x.numel())
    if n == 0:
        return full_logits
    out = x
    block = MAX_BLOCK if n >= MAX_BLOCK else (1 << (n - 1).bit_length())
    nblocks = (n + block - 1) // block

    if nblocks <= MAX_PROGRAMS:
        _direct[(nblocks, 1, 1)](
            x,
            out,
            n_elements=n,
            inv_cap=inv_cap,
            cap=cap,
            BLOCK=block,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
    else:
        _persist[(MAX_PROGRAMS, 1, 1)](
            x,
            out,
            n_elements=n,
            inv_cap=inv_cap,
            cap=cap,
            BLOCK=block,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
    if x is not full_logits:
        full_logits.copy_(x)
    return full_logits


__all__ = ["softcap_inplace_logits"]
