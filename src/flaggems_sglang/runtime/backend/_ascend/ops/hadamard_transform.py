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
def _transform(v, N: tl.constexpr):
    c = tl.arange(0, N)
    for stage in tl.static_range(N.bit_length() - 1):
        partner = tl.gather(v, c ^ (1 << stage), 0)
        v = tl.where((c & (1 << stage)) == 0, v + partner, partner - v)
    return v


@triton.jit
def _row(X, Y, SCALE: tl.constexpr, N: tl.constexpr):
    i = tl.program_id(0) * N + tl.arange(0, N)
    v = tl.load(X + i).to(tl.float32)
    v = _transform(v, N)
    tl.store(Y + i, v * SCALE)


@triton.jit
def _strided(
    X,
    Y,
    SCALE: tl.constexpr,
    N: tl.constexpr,
    SH: tl.constexpr,
    ST: tl.constexpr,
):
    row = tl.program_id(0)
    rest = row
    offset = 0
    for axis in tl.static_range(len(SH) - 1, -1, -1):
        offset += (rest % SH[axis]) * ST[axis]
        rest = rest // SH[axis]
    i = tl.arange(0, N)
    v = tl.load(X + offset + i * ST[len(SH)]).to(tl.float32)
    v = _transform(v, N)
    tl.store(Y + row * N + i, v * SCALE)


def hadamard_transform(x, scale):
    n = x.shape[-1]
    size = x.numel()
    if x.is_contiguous():
        out = torch.empty_like(x)
        if size:
            _row.run(
                x,
                out,
                scale,
                n,
                grid=(size // n,),
                warmup=False,
                num_warps=4 if n <= 4096 else 8,
                num_stages=1,
                enable_fp_fusion=False,
            )
    else:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        if size:
            _strided.run(
                x,
                out,
                scale,
                n,
                tuple(x.shape[:-1]),
                tuple(x.stride()),
                grid=(size // n,),
                warmup=False,
                num_warps=4 if n <= 4096 else 8,
                num_stages=1,
                enable_fp_fusion=False,
            )
    return out


__all__ = ["hadamard_transform"]
