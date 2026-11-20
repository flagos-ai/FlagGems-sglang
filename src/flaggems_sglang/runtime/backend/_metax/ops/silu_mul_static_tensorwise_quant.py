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


@tl.core.extern
def _native_mul(x, y, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [x, y],
        {(tl.float32, tl.float32): ("llvm.mxc.fmul.f32.f32.f32", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@tl.core.extern
def _native_rcp(x, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        [x],
        {(tl.float32,): ("llvm.mxc.rcp.f32", tl.float32)},
        is_pure=True,
        _semantic=_semantic,
    )


@triton.jit
def _encode(v):
    raw = v.to(tl.int32, bitcast=True)
    mag = raw & 2147483647
    a = mag.to(tl.float32, bitcast=True)
    exponent = tl.maximum(mag & 2139095040, 1015021568)
    bias = exponent + 167772160
    biased = (a + bias.to(tl.float32, bitcast=True)).to(tl.int32, bitcast=True)
    code = biased + (bias >> 20) - 1128
    code = tl.where(mag > 1139277824, 127, code)
    return (code | raw >> 24 & 128).to(tl.uint8)


@triton.jit
def _work(
    X,
    Q,
    S,
    n,
    tile,
    D: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    XS: tl.constexpr,
    XI: tl.constexpr,
    QS: tl.constexpr,
    QI: tl.constexpr,
):
    row = tile // C
    col = tile % C * B + tl.arange(0, B)
    src = X + row * XS + col * XI
    g = tl.load(src, col < D, 0, cache_modifier=".cg").to(tl.float32)
    u = tl.load(src + D * XI, col < D, 0, cache_modifier=".cg").to(tl.float32)
    inv = _native_rcp(1.0 + tl.exp2(g * -1.4426950408889634))
    v = tl.fma(tl.fma(g, inv, 0.0), u, 0.0)
    s = tl.load(S)
    raw = s.to(tl.int32, bitcast=True)
    mag = raw & 2147483647
    if (mag & 8388607 == 0) & (mag >= 528482304) & (mag <= 1602224128):
        inv_scale = (2130706432 - mag | raw & -2147483648).to(
            tl.float32, bitcast=True
        )
        v = _native_mul(v, inv_scale)
    else:
        v = tl.div_rn(v, s)
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
    C: tl.constexpr,
    XS: tl.constexpr,
    XI: tl.constexpr,
    QS: tl.constexpr,
    QI: tl.constexpr,
    P: tl.constexpr,
    FULL: tl.constexpr,
):
    Q = Q.to(tl.pointer_type(tl.uint8))
    n = tl.load(NT)
    pid = tl.program_id(0)
    if FULL:
        _work(X, Q, S, n, pid, D, B, C, XS, XI, QS, QI)
    else:
        for tile in range(pid, n * C, P):
            _work(X, Q, S, n, tile, D, B, C, XS, XI, QS, QI)


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
    elements = m * intermediate_size
    if elements <= 65536:
        if elements <= 4096:
            cap, warps = (128, 2)
        else:
            cap, warps = (1024, 4)
    elif intermediate_size >= 4096:
        cap, warps = (4096, 4)
    elif elements >= 4194304:
        cap, warps = (2048, 2)
    else:
        cap, warps = (2048, 4)
    b = min(cap, triton.next_power_of_2(intermediate_size))
    c = triton.cdiv(intermediate_size, b)
    rows = min(max(1, expected_num_tokens), m)
    p = rows * c
    _kernel[p,](
        input,
        triton.reinterpret(output, tl.uint8),
        scale,
        num_tokens_tensor,
        intermediate_size,
        b,
        c,
        *input.stride(),
        *output.stride(),
        p,
        rows == m,
        num_warps=warps,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["silu_mul_static_tensorwise_quant"]
