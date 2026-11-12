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


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _flat(Q, S, Z, Output, META: tl.constexpr, ALIGN: tl.constexpr):
    Q = Q.to(tl.uint64).to(tl.pointer_type(tl.int32))
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    S = S.to(tl.uint64).to(tl.pointer_type(tl.float16))
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    Z = Z.to(tl.uint64).to(tl.pointer_type(tl.int32))
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    Output = Output.to(tl.uint64).to(tl.pointer_type(tl.float16))
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
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
def _words(Q, S, Z, Output, META: tl.constexpr, ALIGN: tl.constexpr):
    Q = Q.to(tl.uint64).to(tl.pointer_type(tl.int32))
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    S = S.to(tl.uint64).to(tl.pointer_type(tl.float16))
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    Z = Z.to(tl.uint64).to(tl.pointer_type(tl.int32))
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    Output = Output.to(tl.uint64).to(tl.pointer_type(tl.float16))
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
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


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _words_compact4(Q, S, Z, Output, META: tl.constexpr, ALIGN: tl.constexpr):
    Q = Q.to(tl.uint64).to(tl.pointer_type(tl.int32))
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    S = S.to(tl.uint64).to(tl.pointer_type(tl.float16))
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    Z = Z.to(tl.uint64).to(tl.pointer_type(tl.int32))
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    Output = Output.to(tl.uint64).to(tl.pointer_type(tl.float16))
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
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
    i = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 + B * 8 >= 2147483648
        or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
        or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
        or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
        else tl.program_id(0)
    ) * B + tl.arange(0, B)
    valid = i < K // 4 * W
    r = tl.where(valid, i // W * 4, 0)
    c = i % W
    j = tl.arange(0, 8)
    sh = (j & 1) * 16 + (j & 6) * 2
    z = tl.load(Z + r // G * Z0 + c * Z1, valid, 0)
    z = z[:, None] >> sh[None, :] & 15
    s = tl.load(
        S + (r // G * S0)[:, None] + (c[:, None] * 8 + j[None, :]) * S1,
        valid[:, None],
        0,
    )
    for step in range(4):
        q = tl.load(Q + (r + step) * Q0 + c * Q1, valid, 0)
        v = ((q[:, None] >> sh[None, :] & 15) - z).to(tl.float16) * s
        tl.store(
            Output + ((r + step) * W + c)[:, None] * 8 + j[None, :],
            v,
            valid[:, None],
        )


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _row_address64(Q, S, Z, Output, META: tl.constexpr, ALIGN: tl.constexpr):
    Q = Q.to(tl.uint64).to(tl.pointer_type(tl.int32))
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    S = S.to(tl.uint64).to(tl.pointer_type(tl.float16))
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    Z = Z.to(tl.uint64).to(tl.pointer_type(tl.int32))
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    Output = Output.to(tl.uint64).to(tl.pointer_type(tl.float16))
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)

    W: tl.constexpr = META[1]
    G: tl.constexpr = META[2]
    B: tl.constexpr = META[3]
    Q0: tl.constexpr = META[4]
    Q1: tl.constexpr = META[5]
    S0: tl.constexpr = META[6]
    S1: tl.constexpr = META[7]
    Z0: tl.constexpr = META[8]
    Z1: tl.constexpr = META[9]
    pid = tl.program_id(0).to(tl.int64)
    tiles: tl.constexpr = tl.cdiv(W * 8, B)
    row = pid // tiles
    col = pid % tiles * B + tl.arange(0, B).to(tl.int64)
    group = row // G
    q = tl.load(Q + row * Q0 + col // 8 * Q1, col < W * 8, 0)
    z = tl.load(Z + group * Z0 + col // 8 * Z1, col < W * 8, 0)
    s = tl.load(S + group * S0 + col * S1, col < W * 8, 0)
    shift = (col & 1) * 16 + (col & 6) * 2
    value = ((q >> shift & 15) - (z >> shift & 15)).to(tl.float16) * s
    tl.store(Output + row * (W * 8) + col, value, col < W * 8)


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _row4_reuse(Q, S, Z, Output, META: tl.constexpr, ALIGN: tl.constexpr):
    Q = Q.to(tl.uint64).to(tl.pointer_type(tl.int32))
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    S = S.to(tl.uint64).to(tl.pointer_type(tl.float16))
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    Z = Z.to(tl.uint64).to(tl.pointer_type(tl.int32))
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    Output = Output.to(tl.uint64).to(tl.pointer_type(tl.float16))
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
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
    pid = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 + B * 8 >= 2147483648
        or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
        or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
        or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
        else tl.program_id(0)
    )
    tiles: tl.constexpr = tl.cdiv(W * 8, B)
    row = pid // tiles * 4
    col = pid % tiles * B + tl.arange(0, B)
    shift = (col & 1) * 16 + (col & 6) * 2
    zero = tl.load(Z + row // G * Z0 + col // 8 * Z1, col < W * 8, 0)
    zero = zero >> shift & 15
    scale = tl.load(S + row // G * S0 + col * S1, col < W * 8, 0)
    for step in range(4):
        q = tl.load(Q + (row + step) * Q0 + col // 8 * Q1, col < W * 8, 0)
        value = ((q >> shift & 15) - zero).to(tl.float16) * scale
        tl.store(Output + (row + step) * (W * 8) + col, value, col < W * 8)


def awq_dequantize(qweight, scales, zeros):
    k, w = qweight.shape
    g = k // scales.shape[0]
    out = scales.new_empty((k, w * 8))
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
    strides = (*qweight.stride(), *scales.stride(), *zeros.stride())
    if (
        k * w * 8 + 8192 >= 2147483648
        or (k - 1) * strides[0] + (w + 1023) * strides[1] >= 2147483648
        or (k // g - 1) * strides[2] + (w * 8 + 8191) * strides[3]
        >= 2147483648
        or ((k // g - 1) * strides[4] + (w + 1023) * strides[5] >= 2147483648)
    ):
        block = min(1024, 0 if w * 8 == 0 else 1 << (w * 8 - 1).bit_length())
        _row_address64.run(
            q_ptr,
            s_ptr,
            z_ptr,
            o_ptr,
            (k, w, g, block, *strides),
            align,
            grid=(k * ((w * 8 + block - 1) // block),),
            num_warps=4,
            warmup=False,
        )
        return out
    if k * w * 8 > 1048576 and w * 8 >= 8192 and (g % 4 == 0):
        _row4_reuse.run(
            q_ptr,
            s_ptr,
            z_ptr,
            o_ptr,
            (k, w, g, 1024, *strides),
            align,
            grid=(k // 4 * ((w * 8 + 1024 - 1) // 1024),),
            num_warps=2,
            warmup=False,
        )
        return out
    if k * w * 8 <= 1048576 or w * 8 >= 8192:
        _flat.run(
            q_ptr,
            s_ptr,
            z_ptr,
            o_ptr,
            (k, w, g, 1024, *strides),
            align,
            grid=((k * w * 8 + 1024 - 1) // 1024,),
            warmup=False,
        )
    elif g % 4 == 0 and k % 4 == 0:
        _words_compact4.run(
            q_ptr,
            s_ptr,
            z_ptr,
            o_ptr,
            (k, w, g, 256, *strides),
            align,
            grid=((k // 4 * w + 255) // 256,),
            warmup=False,
        )
    else:
        _words.run(
            q_ptr,
            s_ptr,
            z_ptr,
            o_ptr,
            (k, w, g, 256, *strides),
            align,
            grid=((k * w + 256 - 1) // 256,),
            warmup=False,
        )
    return out


__all__ = ["awq_dequantize"]
