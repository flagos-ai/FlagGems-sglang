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

import triton
import triton.language as tl


@triton.jit
def _norm(
    X,
    W,
    OUT,
    D: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row = tl.program_id(0) // SPLIT
    part = tl.program_id(0) % SPLIT
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    if SPLIT == 1:
        w = tl.load(W + col, col < D, 0).to(tl.float32)
        tl.store(OUT + row * (2 * D) + col, (x * inv) * w, col < D)
    else:
        j = part * (B // SPLIT) + tl.arange(0, B // SPLIT)
        y = tl.load(X + row * D + j, j < D, 0).to(tl.float32)
        w = tl.load(W + j, j < D, 0).to(tl.float32)
        tl.store(OUT + row * (2 * D) + j, (y * inv) * w, j < D)


@triton.jit
def _eh(
    E,
    H,
    EW,
    HW,
    OUT,
    D: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
    SPLIT: tl.constexpr,
):
    if tl.program_id(1) == 0:
        _norm(E, EW, OUT, D, EPS, B, SPLIT)
    else:
        _norm(H, HW, OUT + D, D, EPS, B, SPLIT)


def fused_eh_norm(
    inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps
):
    e = inputs_embeds
    h = previous_hidden
    ew = enorm_weight
    hw = hnorm_weight
    shape = e.shape
    n = shape[0]
    d = shape[1]
    out = e.new_empty(n, d + d)
    if n:
        if not (
            e.is_contiguous()
            and h.is_contiguous()
            and ew.is_contiguous()
            and hw.is_contiguous()
        ):
            e = e.contiguous()
            h = h.contiguous()
            ew = ew.contiguous()
            hw = hw.contiguous()
        b = triton.next_power_of_2(d)
        split = 8 if n <= 32 else 1
        _eh[(n * split, 2)](
            e,
            h,
            ew,
            hw,
            out,
            d,
            eps,
            b,
            split,
            num_warps=4 if n <= 32 else 8,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["fused_eh_norm"]
