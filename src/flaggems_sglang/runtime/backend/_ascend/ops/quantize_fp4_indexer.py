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

_GRID_CAP = 32768


@triton.jit
def _e2m1_code(v):
    ax = tl.abs(v)
    bit2 = ax > 1.75
    bit1 = ax > tl.where(bit2, 3.5, 0.75)
    last = tl.where(bit2, tl.where(bit1, 5.0, 2.5), tl.where(bit1, 1.25, 0.25))
    bit0 = ax > last
    idx = bit2.to(tl.int16) << 2 | bit1.to(tl.int16) << 1 | bit0.to(tl.int16)
    sign = (v < 0).to(tl.int16) << 3
    return tl.where(ax > 0.25, idx | sign, 0)


@triton.jit
def _e2m1_half_code(v):
    v = v.to(tl.float16)
    q025 = tl.full((), 0.25, tl.float16)
    q075 = tl.full((), 0.75, tl.float16)
    q125 = tl.full((), 1.25, tl.float16)
    q175 = tl.full((), 1.75, tl.float16)
    q250 = tl.full((), 2.5, tl.float16)
    q350 = tl.full((), 3.5, tl.float16)
    q500 = tl.full((), 5.0, tl.float16)
    ax = tl.abs(v)
    bit2 = ax > q175
    bit1 = ax > tl.where(bit2, q350, q075)
    last = tl.where(
        bit2, tl.where(bit1, q500, q250), tl.where(bit1, q125, q025)
    )
    bit0 = ax > last
    idx = bit2.to(tl.int16) << 2 | bit1.to(tl.int16) << 1 | bit0.to(tl.int16)
    sign = (v < -q025).to(tl.int16) << 3
    return idx | sign


@triton.jit
def _maximum_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _quantize_fp4_indexer_ascend_kernel(
    x_ptr,
    x_fp4_ptr,
    x_sf_ptr,
    num_groups,
    GPP: tl.constexpr,
    TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    groups = pid * GPP + tl.arange(0, GPP)
    lanes = tl.arange(0, 16)
    even_off = groups[:, None] * 32 + lanes[None, :] * 2
    odd_off = even_off + 1
    if TAIL:
        mask = groups < num_groups
        xe = tl.load(x_ptr + even_off, mask=mask[:, None], other=0.0).to(
            tl.float32
        )
        xo = tl.load(x_ptr + odd_off, mask=mask[:, None], other=0.0).to(
            tl.float32
        )
    else:
        xe = tl.load(x_ptr + even_off).to(tl.float32)
        xo = tl.load(x_ptr + odd_off).to(tl.float32)
    ae = tl.abs(xe)
    ao = tl.abs(xo)
    amax = tl.reduce(_maximum_nan(ae, ao), 1, _maximum_nan)
    amax = tl.where(amax == amax, amax, float("inf"))
    bits = amax.to(tl.int32, bitcast=True)
    exp = tl.where(
        bits >= 2139095040, 254, tl.maximum((bits + 4194303 >> 23) - 2, 114)
    )
    scale = (exp << 23).to(tl.float32, bitcast=True)
    if (
        x_ptr.dtype.element_ty == tl.bfloat16
        or x_ptr.dtype.element_ty == tl.float16
    ):
        even_code = _e2m1_half_code(tl.div_rn(xe, scale[:, None]))
        odd_code = _e2m1_half_code(tl.div_rn(xo, scale[:, None]))
    else:
        even_code = _e2m1_code(tl.div_rn(xe, scale[:, None]))
        odd_code = _e2m1_code(tl.div_rn(xo, scale[:, None]))
    packed = even_code & 15 | (odd_code & 15) << 4
    out_off = groups[:, None] * 16 + lanes[None, :]
    if TAIL:
        tl.store(x_fp4_ptr + out_off, packed.to(tl.int8), mask=mask[:, None])
        tl.store(x_sf_ptr + groups, exp.to(tl.int8), mask=mask)
    else:
        tl.store(x_fp4_ptr + out_off, packed.to(tl.int8))
        tl.store(x_sf_ptr + groups, exp.to(tl.int8))


def quantize_fp4_indexer(x):
    x = x.contiguous()
    n = x.numel() // 128
    if n > _GRID_CAP * 32:
        x = x.view(-1, 128)
    x_fp4 = x.new_empty((n, 64), dtype=torch.int8)
    x_sf = x.new_empty((n,), dtype=torch.int32)
    rows = (
        (128 if n >= 65536 and x.dtype != torch.float32 else 64)
        if n >= 8192
        else 32
    )
    gpp = 4 * rows
    for start in range(0, n, _GRID_CAP * 32):
        count = min(n - start, _GRID_CAP * 32)
        if start == 0:
            block_x, block_y, block_sf = (x, x_fp4, x_sf)
        else:
            block_x, block_y, block_sf = (
                x[start:],
                x_fp4[start:],
                x_sf[start:],
            )
        _quantize_fp4_indexer_ascend_kernel.run(
            block_x,
            block_y,
            block_sf.view(torch.int8),
            count * 4,
            GPP=gpp,
            TAIL=count % rows != 0,
            num_warps=8,
            num_stages=1,
            grid=((count + rows - 1) // rows,),
            warmup=False,
        )
    return (x_fp4, x_sf)


__all__ = ["quantize_fp4_indexer"]
