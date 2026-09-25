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
def _fill_value(X, F: tl.constexpr, NEG_ZERO: tl.constexpr):
    if X.dtype.element_ty == tl.bfloat16:
        value = tl.full((), F, tl.float32).to(tl.bfloat16)
    else:
        value = tl.full((), F, X.dtype.element_ty)
    if NEG_ZERO:
        value = tl.full((), -2147483648, tl.int32).to(tl.float32, bitcast=True).to(X.dtype.element_ty)
    return value
@triton.jit
def _fill_flat_kernel(
    X, N, R: tl.constexpr, H: tl.constexpr, F: tl.constexpr, NEG_ZERO: tl.constexpr,
    BR: tl.constexpr, BC: tl.constexpr,
):
    n = tl.load(N)
    n = tl.where(n < 0, tl.maximum(n + R, 0), tl.minimum(n, R))
    value = _fill_value(X, F, NEG_ZERO)
    base = X + n * H
    total = (R - n) * H
    nseg = total // BC
    rem = total - nseg * BC
    rr = tl.arange(0, BR)[:, None]
    cc = tl.arange(0, BC)[None, :]
    for blk in tl.range(tl.program_id(0), tl.cdiv(nseg, BR), tl.num_programs(0)):
        seg = blk * BR + rr
        tl.store(base + seg * BC + cc, value, seg < nseg)
    if rem > 0:
        if tl.program_id(0) == 0:
            c1 = tl.arange(0, BC)
            tl.store(base + nseg * BC + c1, value, c1 < rem)
@triton.jit
def _fill_rows_kernel(
    X, N, R: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    F: tl.constexpr, NEG_ZERO: tl.constexpr, BC: tl.constexpr,
):
    n = tl.load(N)
    n = tl.where(n < 0, tl.maximum(n + R, 0), tl.minimum(n, R))
    value = _fill_value(X, F, NEG_ZERO)
    cc = tl.arange(0, BC)
    for row in tl.range(n + tl.program_id(0), R, tl.num_programs(0)):
        base = X + row * S
        for start in range(0, H, BC):
            tl.store(base + start + cc, value, cc < H - start)
def fill_padded_rows(x, num_token_non_padded, fill_value):
    rows, cols = x.shape
    neg_zero = repr(fill_value) == "-0.0"
    grid = (min(max(1, rows), 12),)
    if x.stride(0) == cols:
        nelem = max(1024, 65536 // x.element_size())
        bc = min(2048, nelem)
        br = max(1, nelem // bc)
        _fill_flat_kernel[grid](
            x, num_token_non_padded, rows, cols, fill_value, neg_zero,
            br, bc, num_warps=2, num_stages=1,
        )
        return x
    _fill_rows_kernel[grid](
        x, num_token_non_padded, rows, cols, x.stride(0), fill_value, neg_zero,
        max(16, min(2048, triton.next_power_of_2(max(1, cols)))), num_warps=2, num_stages=1,
    )
    return x
