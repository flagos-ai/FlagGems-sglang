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

"""Triton timestep_embedding (candidate v1)."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _tse_kernel(T, O, HALF, SHIFT: tl.constexpr, SCALE: tl.constexpr,
                LOG_MAXP: tl.constexpr, FLIP: tl.constexpr, TS: tl.constexpr,
                OS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    m = j < HALF
    t = tl.load(T + row * TS).to(tl.float32)
    hf = HALF.to(tl.int32).to(tl.float32)
    ex = -LOG_MAXP * j.to(tl.float32) / (hf - SHIFT)
    f = libdevice.exp(ex)
    a = t * f * SCALE
    c = libdevice.cos(a)
    s = libdevice.sin(a)
    if FLIP:
        first, second = c, s
    else:
        first, second = s, c
    base = row.to(tl.int64) * OS
    tl.store(O + base + j, first, mask=m)
    tl.store(O + base + HALF + j, second, mask=m)


def timestep_embedding(
    t,
    dim,
    flip_sin_to_cos=True,
    downscale_freq_shift=0.0,
    scale=1.0,
    max_period=10000,
):
    t2 = t.contiguous()
    n = t2.shape[0]
    half = max(dim // 2, 1)
    if dim % 2:
        out = torch.zeros(
            (n, dim), dtype=torch.float32, device=t2.device
        )
    else:
        out = torch.empty(
            (n, dim), dtype=torch.float32, device=t2.device
        )
    if n == 0:
        return out
    block = triton.next_power_of_2(half)
    _tse_kernel[(n,)](
        t2,
        out,
        half,
        downscale_freq_shift,
        scale,
        float(torch.log(torch.tensor(float(max_period))).item()),
        bool(flip_sin_to_cos),
        t2.stride(0),
        out.stride(0),
        block,
        num_warps=4 if block <= 1024 else 8,
    )
    return out


__all__ = ["timestep_embedding"]