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
from triton.compiler import CompiledKernel
from triton.runtime import driver
_HOOKS = triton.knobs.runtime if hasattr(triton, "knobs") else CompiledKernel
@triton.jit(do_not_specialize=("X", "N"))
def _fill_padded_rows_kernel(
    X, N, R: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    F: tl.constexpr, GROUP: tl.constexpr, NEG_ZERO: tl.constexpr,
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
        value = tl.full((), -2147483648, tl.int32).to(tl.float32, bitcast=True).to(X.dtype.element_ty)
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
    device = x.device
    fill_repr = repr(fill_value)
    negative_zero = fill_repr == "-0.0"
    aligned = x.data_ptr() % 32 == 0 and stride * x.element_size() % 32 == 0
    key = (rows, cols, stride, x.dtype, num_token_non_padded.dtype,
           device, aligned, type(fill_value), fill_repr)
    cached = getattr(_fill_padded_rows_kernel, "_s2_handle", None)
    if cached is not None and cached[0] == key:
        if _HOOKS.launch_enter_hook is not None or _HOOKS.launch_exit_hook is not None:
            if cached[4]:
                cached[9](x.data_ptr(), num_token_non_padded.data_ptr(), rows, cols, stride,
                          fill_value, cached[3], negative_zero)
            else:
                cached[9](x.data_ptr(), num_token_non_padded.data_ptr())
        else:
            stream = cached[8](device.index)
            if cached[4]:
                cached[5](cached[2][0], 1, 1, stream, cached[6], cached[7],
                          None, None, None, x, num_token_non_padded,
                          rows, cols, stride, fill_value, cached[3], negative_zero)
            else:
                cached[5](cached[2][0], 1, 1, stream, cached[6], cached[7],
                          None, None, None, x, num_token_non_padded)
    else:
        group = max(1, triton.cdiv(rows, 65535)) if aligned else max(1, rows)
        grid = (max(1, triton.cdiv(rows, group)), 1, 1)
        handle = _fill_padded_rows_kernel[grid](
            x, num_token_non_padded, rows, cols, stride, fill_value, group, negative_zero)
        abi = "constexpr" in handle.src.signature.values()
        _fill_padded_rows_kernel._s2_handle = (
            key, handle, grid, group, abi, handle.run, handle.function,
            handle.packed_metadata, driver.active.get_current_stream, handle[grid])
    return x
