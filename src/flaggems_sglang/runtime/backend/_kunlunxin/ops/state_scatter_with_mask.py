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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import triton
import triton.language as tl


@triton.jit
def _state_scatter_kernel(
    dst_ptr,
    src_ptr,
    dst_idx_ptr,
    step_idx_ptr,
    numel,
    sd0,
    sd1,
    ss0,
    ss1,
    ss2,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    layer_idx = tl.program_id(1)
    step = tl.load(step_idx_ptr + i)
    if step < 0:
        return
    didx = tl.load(dst_idx_ptr + i).to(tl.int64)
    l64 = layer_idx.to(tl.int64)
    i64 = i.to(tl.int64)
    st64 = step.to(tl.int64)
    for off in range(0, numel, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        m = offs < numel
        src_off = l64 * ss0 + i64 * ss1 + st64 * ss2 + offs
        dst_off = l64 * sd0 + didx * sd1 + offs
        v = tl.load(src_ptr + src_off, mask=m)
        tl.store(dst_ptr + dst_off, v, mask=m)


def state_scatter_with_mask(dst, src, dst_indices_raw, step_indices_raw):
    out = dst.clone()
    layers, cache = dst.shape[0], dst.shape[1]
    numel = dst.numel() // (layers * cache)
    requests = src.shape[1]
    grid = (requests, layers)
    _state_scatter_kernel[grid](
        out,
        src,
        dst_indices_raw,
        step_indices_raw,
        numel,
        dst.stride(0),
        dst.stride(1),
        src.stride(0),
        src.stride(1),
        src.stride(2),
        BLOCK=1024,
    )
    return out


__all__ = ["state_scatter_with_mask"]
