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
def _iss(
    X,
    SH,
    SC,
    IDX,
    O,
    H: tl.constexpr,
    NH: tl.constexpr,
    SI: tl.constexpr,
    SX: tl.constexpr,
    SV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid // NH
    j = pid % NH
    k = j * BLOCK + tl.arange(0, BLOCK)
    m = k < H
    i = tl.load(IDX + r * SI)
    xf = tl.load(X + r * SX + k, mask=m, other=0.0).to(tl.float32)
    sc = tl.load(SC + i * SV + k, mask=m, other=0.0).to(tl.float32)
    sh = tl.load(SH + i * SV + k, mask=m, other=0.0).to(tl.float32)
    one = (1.0 + sc).to(tl.bfloat16).to(tl.float32)
    v = (xf * one).to(tl.bfloat16).to(tl.float32)
    tl.store(O + r * H + k, (v + sh).to(O.dtype.element_ty), mask=m)


def indexed_scale_shift(x, shift, scale, indices):
    idx = indices
    si = 1
    if idx.dtype == torch.int64:
        idx = idx.view(torch.int32)
        si = 2
    rows, h = x.shape
    block = 256
    if h > 256:
        block = 1024
    if h > 1024:
        block = 2048
    if h > 2048:
        block = 4096
    nh = (h + block - 1) // block
    out = torch.empty((rows, h), dtype=x.dtype, device=x.device)
    _iss[(rows * nh,)](
        x,
        shift,
        scale,
        idx,
        out,
        h,
        nh,
        si,
        x.stride(0),
        shift.stride(0),
        block,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["indexed_scale_shift"]
