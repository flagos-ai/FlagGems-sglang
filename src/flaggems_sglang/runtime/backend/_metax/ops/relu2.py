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

"""One complete row per program; bounded grid for large row counts."""
import torch
import triton
import triton.language as tl


@triton.jit
def _rows(
    X,
    Y,
    R: tl.constexpr,
    D: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
):
    for group in range(triton.cdiv(R, P)):
        row = group * P + tl.program_id(0)
        for start in range(triton.cdiv(D, B)):
            col = start * B + tl.arange(0, B)
            x = tl.load(
                X + row * S0 + col * S1, (row < R) & (col < D), other=0
            ).to(tl.float32)
            z = tl.where(x < 0.0, 0.0, x)
            tl.store(Y + row * D + col, z * z, (row < R) & (col < D))


def relu2(input):
    rows, dim = input.shape
    out = torch.empty((rows, dim), dtype=input.dtype, device=input.device)
    if rows and dim:
        b = min(triton.next_power_of_2(dim), 32768)
        p = min(rows, 65535)
        _rows[(p,)](
            input,
            out,
            rows,
            dim,
            *input.stride(),
            b,
            p,
            num_warps=4,
            num_stages=1,
        )
    return out


__all__ = ["relu2"]
