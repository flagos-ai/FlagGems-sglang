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
def _scatter(
    X,
    I,
    Y,
    R: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    SI: tl.constexpr,
    N: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(I + row * SI).to(tl.int64)
    cols = tl.arange(0, B)
    value = tl.full((B,), 0, Y.dtype.element_ty)
    if N > 0:
        if idx >= 0:
            if S2 == 1 and S1 == D:
                addr = idx * S0 + cols
            else:
                addr = idx * S0 + cols // D * S1 + cols % D * S2
            if B == H * D:
                value = tl.load(X + addr)
            else:
                value = tl.load(X + addr, mask=cols < H * D, other=0)
    if B == H * D:
        tl.store(Y + row * (H * D) + cols, value)
    else:
        tl.store(Y + row * (H * D) + cols, value, mask=cols < H * D)


def scatter_to_padded(out_unpad, inv_indices, batch_size, seqlen):
    _, h, d = out_unpad.shape
    out = torch.empty(
        (batch_size, seqlen, h, d),
        dtype=out_unpad.dtype,
        device=out_unpad.device,
    )
    rows = batch_size * seqlen
    if rows * h * d:
        _scatter[rows,](
            out_unpad,
            inv_indices,
            out,
            rows,
            h,
            d,
            *out_unpad.stride(),
            inv_indices.stride(0),
            out_unpad.shape[0],
            triton.next_power_of_2(h * d),
            num_warps=8,
        )
    return out


__all__ = ["scatter_to_padded"]
