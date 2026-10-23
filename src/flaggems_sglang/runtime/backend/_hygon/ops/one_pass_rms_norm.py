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
def _tile(
    X,
    W,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, B)
    offsets = rows[:, None] * D + cols[None, :]
    valid = (rows[:, None] < M) & (cols[None, :] < D)
    x = tl.load(X + offsets, valid, 0).to(tl.float32)
    w = tl.load(W + cols, cols < D, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 1) / D + E)
    y = x * rstd[:, None] * w[None, :]
    tl.store(Y + offsets, y, valid)


def one_pass_rms_norm(x, w, eps):
    values = x.contiguous()
    weights = w.contiguous()
    out = torch.empty_like(values)
    cols = values.shape[-1]
    rows = values.numel() // cols
    block = triton.next_power_of_2(cols)
    r, warps = 1, 2
    _tile.run(
        values,
        weights,
        out,
        rows,
        cols,
        eps,
        r,
        block,
        grid=(triton.cdiv(rows, r),),
        warmup=False,
        num_warps=warps,
        num_stages=1,
    )
    return out


__all__ = ["one_pass_rms_norm"]
