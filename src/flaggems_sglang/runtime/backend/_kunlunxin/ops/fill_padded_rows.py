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
    F: tl.constexpr, B: tl.constexpr, NEG_ZERO: tl.constexpr, FLAT: tl.constexpr,
):
    if N.dtype.element_ty == tl.int64:
        words = N.to(tl.pointer_type(tl.uint32))
        low = tl.load(words)
        high = tl.load(words + 1).to(tl.int32)
        positive = tl.minimum(low, R).to(tl.int32)
        neg_low = low.to(tl.int32)
        negative = tl.where((high == -1) & (neg_low < 0),
                            tl.maximum(neg_low + R, 0), 0)
        n = tl.where(high < 0, negative,
                     tl.where(high == 0, positive, R))
    else:
        n = tl.load(N).to(tl.int32)
        n = tl.where(n < 0, tl.maximum(n + R, 0), tl.minimum(n, R))
    if X.dtype.element_ty == tl.bfloat16:
        value = tl.full((), F, tl.float32).to(tl.bfloat16)
    else:
        value = tl.full((), F, X.dtype.element_ty)
    if NEG_ZERO:
        value = tl.full((), -2147483648, tl.int32).to(tl.float32, bitcast=True).to(X.dtype.element_ty)
    v = tl.arange(0, B)
    if FLAT:
        start = n * H
        count = R * H - start
        full = (count // B) * B
        for off in range(start, start + full, B):
            tl.store(X + off + v, value)
        rem = count - full
        base = X + start + full
        for bit in tl.static_range(0, 12):
            if rem & (1 << bit):
                tl.store(base + (rem - rem % (2 << bit)) + tl.arange(0, 1 << bit), value)
    else:
        for row in range(n, R):
            base = X + row * S
            for off in range(0, (H // B) * B, B):
                tl.store(base + off + v, value)
            for bit in tl.static_range(0, 12):
                if H % B & (1 << bit):
                    tl.store(base + (H - H % (2 << bit)) + tl.arange(0, 1 << bit), value)
def fill_padded_rows(x, num_token_non_padded, fill_value):
    rows, cols = x.shape
    _fill_padded_rows_kernel[(1,)](
        x, num_token_non_padded, rows, cols, x.stride(0), fill_value, 4096,
        repr(fill_value) == "-0.0", x.stride(0) == cols,
        num_warps=4, num_stages=1,
    )
    return x
