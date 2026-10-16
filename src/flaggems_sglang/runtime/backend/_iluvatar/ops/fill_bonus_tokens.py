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


@triton.jit
def _gather(
    src, lens, out, stride: tl.constexpr, n: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = row < n
    length = tl.load(lens + row, mask=mask, other=1).to(tl.int64)
    idx = row.to(tl.int64) * stride + length - 1
    val = tl.load(src + idx, mask=mask, other=0)
    tl.store(out + row, val.to(out.dtype.element_ty), mask=mask)


def fill_bonus_tokens(accept_tokens, accept_lens, bonus_tokens, accept_stride):
    flat = accept_tokens.reshape(-1)
    n = accept_lens.shape[0]
    out = torch.empty(n, dtype=bonus_tokens.dtype, device=accept_lens.device)
    if n == 0:
        return out
    if not flat.is_contiguous():
        flat = flat.contiguous()
    lens = (
        accept_lens
        if accept_lens.is_contiguous()
        else accept_lens.contiguous()
    )
    _gather[(triton.cdiv(n, 128),)](
        flat, lens, out, int(accept_stride), n, 128, num_warps=2
    )
    return out


__all__ = ["fill_bonus_tokens"]
