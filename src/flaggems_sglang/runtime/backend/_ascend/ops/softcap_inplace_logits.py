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

NPROG = 40
_MAX_BLOCK = 8192


@triton.jit
def _direct(
    ptr,
    n_elements,
    cap,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(ptr + offs).to(tl.float32)
        y = (2.0 * tl.sigmoid(2.0 * (x / cap)) - 1.0) * cap
        tl.store(ptr + offs, y.to(ptr.dtype.element_ty))
    else:
        mask = offs < n_elements
        x = tl.load(ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (2.0 * tl.sigmoid(2.0 * (x / cap)) - 1.0) * cap
        tl.store(ptr + offs, y.to(ptr.dtype.element_ty), mask=mask)


@triton.jit
def _persist(
    ptr,
    n_elements,
    cap,
    BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
):
    pid = tl.program_id(0)
    nblocks = tl.cdiv(n_elements, BLOCK)
    for bid in tl.range(pid, nblocks, NPROG, num_stages=1):
        offs = bid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        x = tl.load(ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (2.0 * tl.sigmoid(2.0 * (x / cap)) - 1.0) * cap
        tl.store(ptr + offs, y.to(ptr.dtype.element_ty), mask=mask)


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    cap = float(final_logit_softcapping)
    x = (
        full_logits
        if full_logits.is_contiguous()
        else full_logits.contiguous()
    )
    n = int(x.numel())
    if n == 0:
        return full_logits

    if n < 65536:
        block = 1024 if n >= 1024 else 64
        even = n % block == 0
        grid = triton.cdiv(n, block)
        _direct[(grid, 1, 1)](
            x,
            n,
            cap,
            BLOCK=block,
            EVEN=even,
            num_warps=4,
            num_stages=1,
        )
    else:
        block = _MAX_BLOCK
        nprog = NPROG if n >= NPROG * block else max(1, triton.cdiv(n, block))
        _persist[(nprog, 1, 1)](
            x,
            n,
            cap,
            BLOCK=block,
            NPROG=nprog,
            num_warps=4,
            num_stages=1,
        )

    if x is not full_logits:
        full_logits.copy_(x)
    return full_logits


__all__ = ["softcap_inplace_logits"]
