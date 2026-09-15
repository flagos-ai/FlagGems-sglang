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

_MAX_BLOCK = 8192


@triton.jit
def _direct(
    ptr,
    n_elements: tl.constexpr,
    inv_cap: tl.constexpr,
    cap: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(ptr + offs).to(tl.float32)
        y = (2.0 * tl.sigmoid(2.0 * (x * inv_cap)) - 1.0) * cap
        tl.store(ptr + offs, y.to(ptr.dtype.element_ty))
    else:
        mask = offs < n_elements
        x = tl.load(ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (2.0 * tl.sigmoid(2.0 * (x * inv_cap)) - 1.0) * cap
        tl.store(ptr + offs, y.to(ptr.dtype.element_ty), mask=mask)


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
    if n <= _MAX_BLOCK:
        block = 64 if n <= 64 else triton.next_power_of_2(n)
        _direct[(1,)](
            x,
            n_elements=n,
            inv_cap=inv_cap,
            cap=cap,
            BLOCK=block,
            EVEN=n % block == 0,
        )
    else:
        block = _MAX_BLOCK
        _direct[(triton.cdiv(n, block),)](
            x,
            n_elements=n,
            inv_cap=inv_cap,
            cap=cap,
            BLOCK=block,
            EVEN=n % block == 0,
        )
    if x is not full_logits:
        full_logits.copy_(x)
    return full_logits


__all__ = ["softcap_inplace_logits"]
