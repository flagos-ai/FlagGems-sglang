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
    N: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    col = tl.arange(0, B)
    ptr = tl.make_block_ptr(
        X, (N, D), (D, 1), (tl.program_id(0) * R, 0), (R, B), (1, 0)
    )
    x = tl.load(ptr, boundary_check=(0, 1), padding_option="zero").to(
        tl.float32
    )
    w = tl.load(W + col, col < D, other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / D + EPS)
    value = x * inv[:, None] * w[None, :]
    ptr = tl.make_block_ptr(
        OUT, (N, D), (2 * D, 1), (tl.program_id(0) * R, 0), (R, B), (1, 0)
    )
    tl.store(ptr, value.to(OUT.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _eh(
    E,
    H,
    EW,
    HW,
    OUT,
    N: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    if tl.program_id(1) == 0:
        _norm(E, EW, OUT, N, D, EPS, R, B)
    else:
        _norm(H, HW, OUT + D, N, D, EPS, R, B)


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
        r = min(32, triton.next_power_of_2(n), max(1, 32768 // b))
        _eh[(triton.cdiv(n, r), 2)](
            e,
            h,
            ew,
            hw,
            out,
            n,
            d,
            eps,
            r,
            b,
            num_warps=1,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["fused_eh_norm"]
