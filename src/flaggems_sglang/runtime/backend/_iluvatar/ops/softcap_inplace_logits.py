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


@triton.jit
def _softcap_1d(
    ptr,
    two_over_cap: tl.constexpr,
    cap: tl.constexpr,
    n_elements,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, 16), 16)
    if EVEN:
        x = tl.load(ptr + offs).to(tl.float32)
        y = (2.0 * tl.sigmoid(x * two_over_cap) - 1.0) * cap
        tl.store(ptr + offs, y.to(ptr.dtype.element_ty))
    else:
        mask = offs < n_elements
        x = tl.load(ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (2.0 * tl.sigmoid(x * two_over_cap) - 1.0) * cap
        tl.store(ptr + offs, y.to(ptr.dtype.element_ty), mask=mask)


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    cap = float(final_logit_softcapping)
    two_over_cap = 2.0 / cap
    x = (
        full_logits
        if full_logits.is_contiguous()
        else full_logits.contiguous()
    )
    n = int(x.numel())
    if n == 0:
        return full_logits
    if n >= 1024:
        block, warps = 1024, 16
    else:
        block, warps = 256, 4
    _softcap_1d[(triton.cdiv(n, block),)](
        x,
        two_over_cap=two_over_cap,
        cap=cap,
        n_elements=n,
        BLOCK=block,
        EVEN=n % block == 0,
        num_warps=warps,
    )
    if x is not full_logits:
        full_logits.copy_(x)
    return full_logits


__all__ = ["softcap_inplace_logits"]
