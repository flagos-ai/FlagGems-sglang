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

__all__ = ["fill_padded_rows"]


@triton.jit(do_not_specialize=("X", "N"))
def _fill_padded_rows_kernel(
    X,
    N,
    R: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    GROUP: tl.constexpr,
    NEG_ZERO: tl.constexpr,
):
    n = tl.load(N)
    n = tl.where(n < 0, tl.maximum(n + R, 0), tl.minimum(n, R))
    first = tl.program_id(0) * GROUP
    end = tl.minimum(first + GROUP, R)
    if X.dtype.element_ty == tl.bfloat16:
        value = tl.full((), F, tl.float32).to(tl.bfloat16)
    else:
        value = tl.full((), F, X.dtype.element_ty)
    if NEG_ZERO:
        value = (
            tl.full((), -2147483648, tl.int32)
            .to(tl.float32, bitcast=True)
            .to(X.dtype.element_ty)
        )
    for row in range(tl.maximum(first, n), end):
        base = X + row * S
        v = tl.arange(0, 2048)
        for start in range(0, (H // 2048) * 2048, 2048):
            tl.store(base + start + v, value)
        for bit in tl.static_range(0, 11):
            if H % 2048 & (1 << bit):
                tail = tl.arange(0, 1 << bit)
                tl.store(base + (H - H % (2 << bit)) + tail, value)


def fill_padded_rows(x, num_token_non_padded, fill_value):
    rows, cols = x.shape
    stride = x.stride(0)
    negative_zero = repr(fill_value) == "-0.0"
    aligned = x.data_ptr() % 32 == 0 and stride * x.element_size() % 32 == 0
    group = max(1, triton.cdiv(rows, 65535)) if aligned else max(1, rows)
    grid = (max(1, triton.cdiv(rows, group)), 1, 1)
    _fill_padded_rows_kernel[grid](
        x,
        num_token_non_padded,
        rows,
        cols,
        stride,
        fill_value,
        group,
        negative_zero,
    )
    return x
