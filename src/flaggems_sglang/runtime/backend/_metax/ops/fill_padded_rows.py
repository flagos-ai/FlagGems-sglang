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
def _fill_padded_rows_kernel(
    X, N, R: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    F: tl.constexpr, B: tl.constexpr, NEG_ZERO: tl.constexpr,
):
    n = tl.load(N)
    n = tl.where(n < 0, tl.maximum(n + R, 0), tl.minimum(n, R))
    if X.dtype.element_ty == tl.bfloat16:
        value = tl.full((), F, tl.float32).to(tl.bfloat16)
    else:
        value = tl.full((), F, X.dtype.element_ty)
    if NEG_ZERO:
        value = tl.full((), -2147483648, tl.int32).to(tl.float32, bitcast=True).to(X.dtype.element_ty)
    p = tl.program_id(0)
    v = tl.arange(0, B)
    if S == H:
        start = p * B
        boundary = n * H
        if start + B > boundary:
            offsets = start + v
            tl.store(X + offsets, value,
                     (offsets >= boundary) & (offsets < R * H))
    else:
        parts: tl.constexpr = (H + B - 1) // B if H > 0 else 1
        row = p // parts
        start = (p % parts) * B
        if (row >= n) & (row < R):
            cols = start + v
            tl.store(X + row * S + cols, value, cols < H)
def fill_padded_rows(x, num_token_non_padded, fill_value):
    rows, cols = x.shape
    stride = x.stride(0)
    block = 4096 if rows * cols >= 1048576 else 1024
    if stride == cols:
        programs = triton.cdiv(rows * cols, block)
    else:
        programs = rows * max(1, triton.cdiv(cols, block))
    _fill_padded_rows_kernel[(max(1, programs),)](
        x, num_token_non_padded, rows, cols, stride, fill_value, block,
        repr(fill_value) == "-0.0",
        num_warps=1,
    )
    return x
