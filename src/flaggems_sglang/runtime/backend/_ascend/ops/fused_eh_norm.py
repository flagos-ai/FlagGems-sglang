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
def _eh(
    E, H, EW, HW, OUT, D: tl.constexpr, EPS: tl.constexpr, B: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    e = tl.load(E + row * D + col, col < D, 0).to(tl.float32)
    h = tl.load(H + row * D + col, col < D, 0).to(tl.float32)
    ew = tl.load(EW + col, col < D, 0).to(tl.float32)
    hw = tl.load(HW + col, col < D, 0).to(tl.float32)
    er = tl.rsqrt(tl.sum(e * e, 0) / D + EPS)
    hr = tl.rsqrt(tl.sum(h * h, 0) / D + EPS)
    tl.store(OUT + row * (2 * D) + col, (e * er) * ew, col < D)
    tl.store(OUT + row * (2 * D) + D + col, (h * hr) * hw, col < D)


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
        _eh[(n,)](
            e,
            h,
            ew,
            hw,
            out,
            d,
            eps,
            b,
            num_warps=4 if b <= 2048 else 8,
            enable_fp_fusion=False,
            enable_auto_bind_sub_block=False,
            optimize_dynamic_offset=True,
        )
    return out


__all__ = ["fused_eh_norm"]
