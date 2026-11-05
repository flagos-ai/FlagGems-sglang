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
def _base(
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
    cols = tl.arange(0, B)
    for row in range(tl.program_id(0).to(tl.int64), R, tl.num_programs(0)):
        idx = tl.load(I + row * SI).to(tl.int64)
        value = tl.full((B,), 0, Y.dtype.element_ty)
        if N > 0:
            if idx >= 0:
                if S2 == 1 and S1 == D:
                    value = tl.load(
                        X + idx * S0 + cols, mask=cols < H * D, other=0
                    )
                else:
                    value = tl.load(
                        X + idx * S0 + cols // D * S1 + cols % D * S2,
                        mask=cols < H * D,
                        other=0,
                    )
        tl.store(Y + row * (H * D) + cols, value, mask=cols < H * D)


@triton.jit
def _unroll4(
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
    cols = tl.arange(0, B)
    npr = tl.num_programs(0)
    for row in range(tl.program_id(0).to(tl.int64), R, npr * 4):
        r1 = row + npr
        r2 = row + npr * 2
        r3 = row + npr * 3
        idx0 = tl.load(I + row * SI).to(tl.int64)
        idx1 = tl.load(I + r1 % R * SI).to(tl.int64)
        idx2 = tl.load(I + r2 % R * SI).to(tl.int64)
        idx3 = tl.load(I + r3 % R * SI).to(tl.int64)
        ok1 = r1 < R
        ok2 = r2 < R
        ok3 = r3 < R
        v0 = tl.full((B,), 0, Y.dtype.element_ty)
        v1 = tl.full((B,), 0, Y.dtype.element_ty)
        v2 = tl.full((B,), 0, Y.dtype.element_ty)
        v3 = tl.full((B,), 0, Y.dtype.element_ty)
        if N > 0:
            if idx0 >= 0:
                v0 = tl.load(X + idx0 * S0 + cols, mask=cols < H * D, other=0)
            if ok1:
                if idx1 >= 0:
                    v1 = tl.load(
                        X + idx1 * S0 + cols, mask=cols < H * D, other=0
                    )
            if ok2:
                if idx2 >= 0:
                    v2 = tl.load(
                        X + idx2 * S0 + cols, mask=cols < H * D, other=0
                    )
            if ok3:
                if idx3 >= 0:
                    v3 = tl.load(
                        X + idx3 * S0 + cols, mask=cols < H * D, other=0
                    )
        tl.store(Y + row * (H * D) + cols, v0, mask=cols < H * D)
        if ok1:
            tl.store(Y + r1 * (H * D) + cols, v1, mask=cols < H * D)
        if ok2:
            tl.store(Y + r2 * (H * D) + cols, v2, mask=cols < H * D)
        if ok3:
            tl.store(Y + r3 * (H * D) + cols, v3, mask=cols < H * D)


def scatter_to_padded(out_unpad, inv_indices, batch_size, seqlen):
    n, h, d = out_unpad.shape
    rows = batch_size * seqlen
    out = torch.empty(
        (batch_size, seqlen, h, d),
        dtype=out_unpad.dtype,
        device=out_unpad.device,
    )
    if rows * h * d:
        kern = (
            _unroll4
            if rows >= 256 and out_unpad.stride() == (h * d, d, 1)
            else _base
        )
        kern[min(rows, 120),](
            out_unpad,
            inv_indices,
            out,
            rows,
            h,
            d,
            *out_unpad.stride(),
            inv_indices.stride(0),
            n,
            triton.next_power_of_2(h * d),
            num_warps=4,
        )
    return out


__all__ = ["scatter_to_padded"]
