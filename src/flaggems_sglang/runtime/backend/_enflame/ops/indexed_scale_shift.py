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


@triton.jit
def _iss_rows(
    X,
    SH,
    SC,
    IDX,
    O,
    H: tl.constexpr,
    SI: tl.constexpr,
    SX: tl.constexpr,
    SV: tl.constexpr,
    RP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    m = k < H
    r0 = pid * RP
    for u in tl.static_range(RP):
        r = r0 + u
        i = tl.load(IDX + r * SI)
        x = tl.load(X + r * SX + k, mask=m, other=0.0)
        sc = tl.load(SC + i * SV + k, mask=m, other=0.0)
        sh = tl.load(SH + i * SV + k, mask=m, other=0.0)
        one = 1.0 + sc
        v = x * one
        tl.store(O + r * H + k, v + sh, mask=m)


@triton.jit
def _iss_rows_pf(
    X,
    SH,
    SC,
    IDX,
    O,
    H: tl.constexpr,
    SI: tl.constexpr,
    SX: tl.constexpr,
    SV: tl.constexpr,
    RP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    m = k < H
    r0 = pid * RP
    i0 = tl.load(IDX + r0 * SI)
    x0 = tl.load(X + r0 * SX + k, mask=m, other=0.0)
    sc0 = tl.load(SC + i0 * SV + k, mask=m, other=0.0)
    sh0 = tl.load(SH + i0 * SV + k, mask=m, other=0.0)
    i1 = tl.load(IDX + (r0 + 1) * SI)
    x1 = tl.load(X + (r0 + 1) * SX + k, mask=m, other=0.0)
    sc1 = tl.load(SC + i1 * SV + k, mask=m, other=0.0)
    sh1 = tl.load(SH + i1 * SV + k, mask=m, other=0.0)
    for u in tl.static_range(RP):
        if u + 2 < RP:
            i2 = tl.load(IDX + (r0 + u + 2) * SI)
            x2 = tl.load(X + (r0 + u + 2) * SX + k, mask=m, other=0.0)
            sc2 = tl.load(SC + i2 * SV + k, mask=m, other=0.0)
            sh2 = tl.load(SH + i2 * SV + k, mask=m, other=0.0)
        one = 1.0 + sc0
        v = x0 * one
        tl.store(O + (r0 + u) * H + k, v + sh0, mask=m)
        i0 = i1
        x0 = x1
        sc0 = sc1
        sh0 = sh1
        if u + 2 < RP:
            i1 = i2
            x1 = x2
            sc1 = sc2
            sh1 = sh2


def indexed_scale_shift(x, shift, scale, indices):
    idx = indices
    si = 1
    if idx.dtype == torch.int64:
        idx = idx.view(torch.int32)
        si = 2
    rows, h = x.shape
    out = torch.empty((rows, h), dtype=x.dtype, device=x.device)
    if x.dtype == torch.bfloat16 and h <= 4096:
        rp = 1
        if rows >= 8192 and rows % 64 == 0:
            rp = 64
        else:
            for cand in (32, 16, 8, 4, 2):
                if rows >= cand * 32 and rows % cand == 0:
                    rp = cand
                    break
        if rp >= 2:
            _iss_rows_pf[(rows // rp,)](
                x,
                shift,
                scale,
                idx,
                out,
                h,
                si,
                x.stride(0),
                shift.stride(0),
                rp,
                4096,
                num_warps=4,
                num_stages=1,
            )
            return out
        _iss_rows[(rows,)](
            x,
            shift,
            scale,
            idx,
            out,
            h,
            si,
            x.stride(0),
            shift.stride(0),
            1,
            4096,
            num_warps=4,
            num_stages=1,
        )
        return out
    block = 256
    if h > 256:
        block = 1024
    if h > 1024:
        block = 2048
    if h > 2048:
        block = 4096
    nh = (h + block - 1) // block
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
