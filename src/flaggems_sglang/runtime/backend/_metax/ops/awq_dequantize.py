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

import struct

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _flat(
    Q: tl.float64,
    S: tl.float64,
    Z: tl.float64,
    Output: tl.float64,
    META: tl.constexpr,
):
    K: tl.constexpr = META[0]
    W: tl.constexpr = META[1]
    G: tl.constexpr = META[2]
    B: tl.constexpr = META[3]
    Q0: tl.constexpr = META[4]
    Q1: tl.constexpr = META[5]
    S0: tl.constexpr = META[6]
    S1: tl.constexpr = META[7]
    Z0: tl.constexpr = META[8]
    Z1: tl.constexpr = META[9]
    ALIGN: tl.constexpr = META[10]
    Q = Q.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    S = S.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.float16))
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    Z = Z.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    Output = Output.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.float16))
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
    i = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 + B * 8 >= 2147483648
        or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
        or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
        or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
        else tl.program_id(0)
    ) * B + tl.arange(0, B)
    r = i // (W * 8)
    c = i % (W * 8)
    sh = (c & 1) * 16 + (c & 6) * 2
    q = tl.load(Q + r * Q0 + c // 8 * Q1, i < K * W * 8, 0)
    z = tl.load(Z + r // G * Z0 + c // 8 * Z1, i < K * W * 8, 0)
    s = tl.load(S + r // G * S0 + c * S1, i < K * W * 8, 0)
    v = ((q >> sh & 15) - (z >> sh & 15)).to(tl.float16) * s
    tl.store(Output + i, v, i < K * W * 8)


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _words(
    Q: tl.float64,
    S: tl.float64,
    Z: tl.float64,
    Output: tl.float64,
    META: tl.constexpr,
):
    K: tl.constexpr = META[0]
    W: tl.constexpr = META[1]
    G: tl.constexpr = META[2]
    B: tl.constexpr = META[3]
    Q0: tl.constexpr = META[4]
    Q1: tl.constexpr = META[5]
    S0: tl.constexpr = META[6]
    S1: tl.constexpr = META[7]
    Z0: tl.constexpr = META[8]
    Z1: tl.constexpr = META[9]
    ALIGN: tl.constexpr = META[10]
    Q = Q.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    S = S.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.float16))
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    Z = Z.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.int32))
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    Output = Output.to(tl.uint64, bitcast=True).to(tl.pointer_type(tl.float16))
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
    i = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 + B * 8 >= 2147483648
        or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
        or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
        or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
        else tl.program_id(0)
    ) * B + tl.arange(0, B)
    r = i // W
    c = i % W
    j = tl.arange(0, 8)
    sh = (j & 1) * 16 + (j & 6) * 2
    q = tl.load(Q + r * Q0 + c * Q1, i < K * W, 0)
    z = tl.load(Z + r // G * Z0 + c * Z1, i < K * W, 0)
    s = tl.load(
        S + (r // G * S0)[:, None] + (c[:, None] * 8 + j[None, :]) * S1,
        i[:, None] < K * W,
        0,
    )
    v = (
        (q[:, None] >> sh[None, :] & 15) - (z[:, None] >> sh[None, :] & 15)
    ).to(tl.float16) * s
    tl.store(Output + i[:, None] * 8 + j[None, :], v, i[:, None] < K * W)


def awq_dequantize(qweight, scales, zeros):
    k, w = qweight.shape
    g = k // scales.shape[0]
    out = torch.empty((k, w * 8), dtype=scales.dtype, device=qweight.device)
    q_ptr = qweight.data_ptr()
    s_ptr = scales.data_ptr()
    z_ptr = zeros.data_ptr()
    o_ptr = out.data_ptr()
    align = (
        q_ptr % 16 == 0,
        s_ptr % 16 == 0,
        z_ptr % 16 == 0,
        o_ptr % 16 == 0,
    )
    q_arg, s_arg, z_arg, o_arg = struct.unpack(
        "=4d", struct.pack("=4Q", q_ptr, s_ptr, z_ptr, o_ptr)
    )
    strides = (*qweight.stride(), *scales.stride(), *zeros.stride())
    _words.run(
        q_arg,
        s_arg,
        z_arg,
        o_arg,
        (k, w, g, 256, *strides, align),
        grid=((k * w + 256 - 1) // 256,),
        warmup=False,
    )
    return out


__all__ = ["awq_dequantize"]
