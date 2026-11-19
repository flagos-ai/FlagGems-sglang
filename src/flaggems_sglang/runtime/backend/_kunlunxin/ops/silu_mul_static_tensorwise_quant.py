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

import torch
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
    tile,
    D: tl.constexpr,
    B: tl.constexpr,
    C: tl.constexpr,
    XS: tl.constexpr,
    XI: tl.constexpr,
    QS: tl.constexpr,
    QI: tl.constexpr,
    CONT: tl.constexpr,
):
    row = tile // C
    if row < n:
        col = (tile % C) * B + tl.arange(0, B)
        src = X + row * XS + col * XI
        scale = tl.load(S)
        if CONT:
            g = tl.load(src, offset_state_policy="continuous").to(tl.float32)
            u = tl.load(src + D * XI, offset_state_policy="continuous").to(
                tl.float32
            )
            v = g / (1.0 + tl.exp(-g)) * u / scale
            tl.store(
                Q + row * QS + col * QI,
                _encode(v),
                offset_state_policy="continuous",
            )
        else:
            edge = col < D
            g = tl.load(src, edge, 0.0).to(tl.float32)
            u = tl.load(src + D * XI, edge, 0.0).to(tl.float32)
            v = g / (1.0 + tl.exp(-g)) * u / scale
            tl.store(Q + row * QS + col * QI, _encode(v), edge)


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
    CONT: tl.constexpr,
):
    n = tl.load(NT)
    pid = tl.program_id(0)
    if FULL:
        _work(X, Q, S, n, pid, D, B, C, XS, XI, QS, QI, CONT)
    else:
        for tile in range(pid, n * C, P):
            _work(X, Q, S, n, tile, D, B, C, XS, XI, QS, QI, CONT)


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
    block = min(2048, triton.next_power_of_2(intermediate_size))
    chunks = triton.cdiv(intermediate_size, block)
    rows = min(max(1, expected_num_tokens), m)
    programs = rows * chunks
    xs, xi = input.stride()
    qs, qi = output.stride()
    _kernel[(programs,)](
        input,
        output.view(torch.uint8),
        scale,
        num_tokens_tensor,
        intermediate_size,
        block,
        chunks,
        xs,
        xi,
        qs,
        qi,
        programs,
        rows == m,
        intermediate_size % block == 0 and xi == 1 and qi == 1,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["silu_mul_static_tensorwise_quant"]
