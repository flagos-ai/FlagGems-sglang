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
def _softcap_rows(
    ptr,
    cap,
    ncols,
    row_stride,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    base = ptr + row * row_stride
    if EVEN:
        x = tl.load(base + offs).to(tl.float32)
        y = (2.0 * tl.sigmoid(2.0 * (x / cap)) - 1.0) * cap
        tl.store(base + offs, y.to(ptr.dtype.element_ty))
    else:
        mask = offs < ncols
        x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        y = (2.0 * tl.sigmoid(2.0 * (x / cap)) - 1.0) * cap
        tl.store(base + offs, y.to(ptr.dtype.element_ty), mask=mask)


def _block(n):
    if n >= 1024:
        return 1024
    if n >= 256:
        return 256
    if n >= 128:
        return 128
    return 64


def softcap_inplace_logits(full_logits, final_logit_softcapping):
    cap = float(final_logit_softcapping)
    n = full_logits.numel()
    if n == 0:
        return full_logits
    x = (
        full_logits
        if full_logits.is_contiguous()
        else full_logits.contiguous()
    )
    ncols = n
    nrows = 1
    row_stride = n
    block = _block(ncols)
    even = ncols % block == 0
    _softcap_rows[(triton.cdiv(ncols, block), nrows)](
        x,
        cap,
        ncols,
        row_stride,
        BLOCK=block,
        EVEN=even,
        num_warps=8,
        num_stages=1,
    )
    if x is not full_logits:
        full_logits.copy_(x)
    return full_logits


__all__ = ["softcap_inplace_logits"]
