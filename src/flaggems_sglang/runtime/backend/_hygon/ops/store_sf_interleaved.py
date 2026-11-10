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


@triton.jit(do_not_specialize=["Y"])
def _scatter(
    X,
    L,
    Y: tl.pointer_type(tl.int32),
    T: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
):
    for start in range(
        tl.program_id(0), tl.cdiv(T * H, B), tl.num_programs(0)
    ):
        i = start * B + tl.arange(0, B)
        t = i // H
        h = i % H
        if (start + 1) * B <= T * H:
            loc = tl.load(L + t).to(tl.int64)
            out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
            v = tl.load(X + i)
            tl.store(Y + out, v, cache_modifier=".cs")
        else:
            loc = tl.load(L + t, t < T, 0).to(tl.int64)
            out = ((loc >> 7) * H + h) * 128 + (loc & 31) * 4 + (loc >> 5 & 3)
            v = tl.load(X + i, i < T * H, 0)
            tl.store(Y + out, v, i < T * H, cache_modifier=".cs")


def store_sf_interleaved(sf_in, sf_out, loc, page_size):
    if page_size != 128:
        raise ValueError("page_size must be 128")
    x = sf_in.contiguous().view(torch.int32)
    locations = loc.contiguous()
    if sf_out.is_contiguous() and sf_out.storage_offset() % 4 == 0:
        out = sf_out
    else:
        out = sf_out.clone(memory_format=torch.contiguous_format)
    y = out.view(torch.int32)
    t, h, _ = sf_in.shape
    if t * h:
        _scatter.run(
            x,
            locations,
            y,
            t,
            h,
            128,
            num_warps=4,
            grid=(min(triton.cdiv(t * h, 128), 4096),),
            warmup=False,
            waves_per_eu=2,
        )
    return out


__all__ = ["store_sf_interleaved"]
