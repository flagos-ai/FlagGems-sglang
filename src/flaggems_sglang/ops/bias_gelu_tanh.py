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

"""Triton bias_gelu_tanh (candidate v1)."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _bgt_kernel(X, BI, O, N, RS: tl.constexpr, OS: tl.constexpr,
                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    m = cols < N
    x = tl.load(X + row.to(tl.int64) * RS + cols, mask=m, other=0.0).to(
        tl.float32
    )
    b = tl.load(BI + cols, mask=m, other=0.0).to(tl.float32)
    y = x + b
    c: tl.constexpr = 0.7978845608028654
    inner = c * (y + 0.044715 * y * y * y)
    out = 0.5 * y * (1.0 + libdevice.tanh(inner))
    tl.store(O + row.to(tl.int64) * OS + cols, out, mask=m)


def bias_gelu_tanh(x, bias):
    n = x.shape[-1]
    orig = x.shape
    x2 = x.reshape(-1, n)
    out = torch.empty_like(x2)
    if x2.numel() == 0:
        return out.reshape(orig)
    block = min(2048, max(16, triton.next_power_of_2(n)))
    warps = 4 if block <= 2048 else 8
    grid = (x2.shape[0], triton.cdiv(n, block))
    _bgt_kernel[grid](
        x2,
        bias,
        out,
        n,
        x2.stride(0),
        out.stride(0),
        block,
        num_warps=warps,
    )
    return out.reshape(orig)


__all__ = ["bias_gelu_tanh"]