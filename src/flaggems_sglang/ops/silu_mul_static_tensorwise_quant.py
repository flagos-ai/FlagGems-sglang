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
def _encode(v):
    raw = v.to(tl.int32, bitcast=True)
    mag = raw & 2147483647
    a = mag.to(tl.float32, bitcast=True)
    bias = tl.maximum(mag & 2139095040, 1015021568) + 167772160
    biased = a + bias.to(tl.float32, bitcast=True)
    code = biased.to(tl.int32, bitcast=True) - bias + (bias >> 20) - 1128
    code = tl.where(mag > 1139277824, 127, code)
    return (code | ((raw >> 24) & 128)).to(tl.uint8)


@triton.jit
def _work(
    X,
    Q,
    S,
    n,
    row,
    chunk,
    D: tl.constexpr,
    B: tl.constexpr,
    XS: tl.constexpr,
    XI: tl.constexpr,
    QS: tl.constexpr,
    QI: tl.constexpr,
):
    col = chunk * B + tl.arange(0, B)
    src = X + row * XS + col * XI
    g = tl.load(src, col < D, 0.0, cache_modifier=".cg").to(tl.float32)
    u = tl.load(src + D * XI, col < D, 0.0, cache_modifier=".cg").to(
        tl.float32
    )
    v = g / (1.0 + tl.exp(-g)) * u / tl.load(S)
    tl.store(
        Q + row * QS + col * QI,
        _encode(v),
        (col < D) & (row < n),
        cache_modifier=".cs",
    )


@triton.jit
def _kernel(
    X,
    Q,
    S,
    NT,
    D: tl.constexpr,
    B: tl.constexpr,
    XS: tl.constexpr,
    XI: tl.constexpr,
    QS: tl.constexpr,
    QI: tl.constexpr,
    R: tl.constexpr,
    FULL: tl.constexpr,
):
    n = tl.load(NT)
    row = tl.program_id(1)
    chunk = tl.program_id(0)
    if FULL:
        _work(X, Q, S, n, row, chunk, D, B, XS, XI, QS, QI)
    else:
        for i in range(row, n, R):
            _work(X, Q, S, n, i, chunk, D, B, XS, XI, QS, QI)


def silu_mul_static_tensorwise_quant(
    input,
    output,
    scale,
    num_tokens_tensor,
    expected_num_tokens,
    intermediate_size,
):
    m = input.shape[0]
    if m == 0:
        return output
    b = min(1024, triton.next_power_of_2(intermediate_size))
    rows = min(max(1, expected_num_tokens), m)
    _kernel[(triton.cdiv(intermediate_size, b), rows)](
        input,
        triton.reinterpret(output, tl.uint8),
        scale,
        num_tokens_tensor,
        intermediate_size,
        b,
        *input.stride(),
        *output.stride(),
        rows,
        rows == m,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["silu_mul_static_tensorwise_quant"]
