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
from triton.language import float16 as fp16
from triton.language import int32 as i32
from triton.language.extra.cann.extension import extract_slice, insert_slice


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _ordered_float(
    Q: "*i32",
    S: "*fp16",
    Z: "*i32",
    Output: "*fp16",
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    FULL: tl.constexpr,
    R: tl.constexpr,
    ALIGN: tl.constexpr,
):
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
    Q0: tl.constexpr = W
    Q1: tl.constexpr = 1
    S0: tl.constexpr = W * 8
    S1: tl.constexpr = 1
    Z0: tl.constexpr = W
    Z1: tl.constexpr = 1
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if K * W * 8 + B * 8 >= 2147483648
            or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
            or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
            or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
            else tl.program_id(0)
        ),
        K // R * tl.cdiv(WIDTH, B),
        tl.num_programs(0),
    ):
        base_row = tile // tl.cdiv(WIDTH, B) * R
        r = base_row + tl.arange(0, R)
        col_tile = tile % tl.cdiv(WIDTH, B)
        c = START + col_tile * B + tl.arange(0, B)
        if FULL:
            q = tl.load(Q + r[:, None] * Q0 + c[None, :] * Q1)
            z = tl.load(Z + base_row // G * Z0 + c * Z1)[None, :]
        else:
            q = tl.load(
                Q + r[:, None] * Q0 + c[None, :] * Q1, c[None, :] < W, 0
            )
            z = tl.load(Z + base_row // G * Z0 + c * Z1, c < W, 0)[None, :]
        d0 = ((q & 15) - (z & 15)).to(tl.float32).reshape((R * B,))
        d1 = ((q >> 4 & 15) - (z >> 4 & 15)).to(tl.float32).reshape((R * B,))
        d2 = ((q >> 8 & 15) - (z >> 8 & 15)).to(tl.float32).reshape((R * B,))
        d3 = ((q >> 12 & 15) - (z >> 12 & 15)).to(tl.float32).reshape((R * B,))
        d4 = ((q >> 16 & 15) - (z >> 16 & 15)).to(tl.float32).reshape((R * B,))
        d5 = ((q >> 20 & 15) - (z >> 20 & 15)).to(tl.float32).reshape((R * B,))
        d6 = ((q >> 24 & 15) - (z >> 24 & 15)).to(tl.float32).reshape((R * B,))
        d7 = ((q >> 28 & 15) - (z >> 28 & 15)).to(tl.float32).reshape((R * B,))
        planes = tl.full((8, R * B), 0, tl.float32)
        planes = insert_slice(
            planes, d0.reshape((1, R * B)), (0, 0), (1, R * B), (1, 1)
        )
        planes = insert_slice(
            planes, d4.reshape((1, R * B)), (1, 0), (1, R * B), (1, 1)
        )
        planes = insert_slice(
            planes, d1.reshape((1, R * B)), (2, 0), (1, R * B), (1, 1)
        )
        planes = insert_slice(
            planes, d5.reshape((1, R * B)), (3, 0), (1, R * B), (1, 1)
        )
        planes = insert_slice(
            planes, d2.reshape((1, R * B)), (4, 0), (1, R * B), (1, 1)
        )
        planes = insert_slice(
            planes, d6.reshape((1, R * B)), (5, 0), (1, R * B), (1, 1)
        )
        planes = insert_slice(
            planes, d3.reshape((1, R * B)), (6, 0), (1, R * B), (1, 1)
        )
        planes = insert_slice(
            planes, d7.reshape((1, R * B)), (7, 0), (1, R * B), (1, 1)
        )
        value = tl.trans(planes.reshape((8, R * B))).reshape((R, B * 8))
        j = (START + col_tile * B) * 8 + tl.arange(0, B * 8)
        if FULL:
            s = tl.load(S + base_row // G * S0 + j * S1)[None, :]
            tl.store(Output + r[:, None] * W * 8 + j[None, :], value * s)
        else:
            s = tl.load(S + base_row // G * S0 + j * S1, j < W * 8, 0)[None, :]
            tl.store(
                Output + r[:, None] * W * 8 + j[None, :],
                value * s,
                j[None, :] < W * 8,
            )


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _ordered_group_swar_hoist(
    Q: "*i32",
    S: "*fp16",
    Z: "*i32",
    Output: "*fp16",
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    FULL: tl.constexpr,
    R: tl.constexpr,
    ALIGN: tl.constexpr,
):
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
    Q0: tl.constexpr = W
    Q1: tl.constexpr = 1
    S0: tl.constexpr = W * 8
    S1: tl.constexpr = 1
    Z0: tl.constexpr = W
    Z1: tl.constexpr = 1
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if K * W * 8 + B * 8 >= 2147483648
            or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
            or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
            or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
            else tl.program_id(0)
        ),
        K // G * tl.cdiv(WIDTH, B),
        tl.num_programs(0),
    ):
        group = tile // tl.cdiv(WIDTH, B)
        col_tile = tile % tl.cdiv(WIDTH, B)
        c = START + col_tile * B + tl.arange(0, B)
        j = (START + col_tile * B) * 8 + tl.arange(0, B * 8)
        if FULL:
            z = tl.load(Z + group * Z0 + c * Z1)[None, :]
            s = tl.load(S + group * S0 + j * S1)[None, :].to(tl.float16)
        else:
            z = tl.load(Z + group * Z0 + c * Z1, c < W, 0)[None, :]
            s = tl.load(S + group * S0 + j * S1, j < W * 8, 0)[None, :].to(
                tl.float16
            )
        zlo = 269488144 - (z & 252645135)
        zhi = 269488144 - (z >> 4 & 252645135)
        for row_block in range(G // R):
            base_row = group * G + row_block * R
            r = base_row + tl.arange(0, R)
            if FULL:
                q = tl.load(Q + r[:, None] * Q0 + c[None, :] * Q1)
            else:
                q = tl.load(
                    Q + r[:, None] * Q0 + c[None, :] * Q1, c[None, :] < W, 0
                )
            lo = (q & 252645135) + zlo
            hi = (q >> 4 & 252645135) + zhi
            lo0 = extract_slice(lo, (0, 0), (R, B // 2), (1, 2))
            hi0 = extract_slice(hi, (0, 0), (R, B // 2), (1, 2))
            lo1 = extract_slice(lo, (0, 1), (R, B // 2), (1, 2))
            hi1 = extract_slice(hi, (0, 1), (R, B // 2), (1, 2))
            lo0_low = lo0.to(tl.int16)
            lo0_high = (lo0 >> 16).to(tl.int16)
            hi0_low = hi0.to(tl.int16)
            hi0_high = (hi0 >> 16).to(tl.int16)
            lo1_low = lo1.to(tl.int16)
            lo1_high = (lo1 >> 16).to(tl.int16)
            hi1_low = hi1.to(tl.int16)
            hi1_high = (hi1 >> 16).to(tl.int16)
            planes = tl.full((16, R * B // 2), 0, tl.int16)
            d0 = (lo0_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d0, (0, 0), (1, R * B // 2), (1, 1))
            d1 = (lo0_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d1, (1, 0), (1, R * B // 2), (1, 1))
            d2 = (hi0_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d2, (2, 0), (1, R * B // 2), (1, 1))
            d3 = (hi0_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d3, (3, 0), (1, R * B // 2), (1, 1))
            d4 = (lo0_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d4, (4, 0), (1, R * B // 2), (1, 1))
            d5 = (lo0_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d5, (5, 0), (1, R * B // 2), (1, 1))
            d6 = (hi0_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d6, (6, 0), (1, R * B // 2), (1, 1))
            d7 = (hi0_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d7, (7, 0), (1, R * B // 2), (1, 1))
            d8 = (lo1_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d8, (8, 0), (1, R * B // 2), (1, 1))
            d9 = (lo1_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d9, (9, 0), (1, R * B // 2), (1, 1))
            d10 = (hi1_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d10, (10, 0), (1, R * B // 2), (1, 1)
            )
            d11 = (hi1_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d11, (11, 0), (1, R * B // 2), (1, 1)
            )
            d12 = (lo1_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d12, (12, 0), (1, R * B // 2), (1, 1)
            )
            d13 = (lo1_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d13, (13, 0), (1, R * B // 2), (1, 1)
            )
            d14 = (hi1_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d14, (14, 0), (1, R * B // 2), (1, 1)
            )
            d15 = (hi1_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d15, (15, 0), (1, R * B // 2), (1, 1)
            )
            value = (
                tl.trans(planes).reshape((R, B * 8)).to(tl.float16) - 16.0
            ).to(tl.float16)
            if FULL:
                tl.store(Output + r[:, None] * W * 8 + j[None, :], value * s)
            else:
                tl.store(
                    Output + r[:, None] * W * 8 + j[None, :],
                    value * s,
                    j[None, :] < W * 8,
                )


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _ordered_group_swar_hoist_chunks(
    Q: "*i32",
    S: "*fp16",
    Z: "*i32",
    Output: "*fp16",
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    FULL: tl.constexpr,
    R: tl.constexpr,
    ALIGN: tl.constexpr,
):
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
    Q0: tl.constexpr = W
    Q1: tl.constexpr = 1
    S0: tl.constexpr = W * 8
    S1: tl.constexpr = 1
    Z0: tl.constexpr = W
    Z1: tl.constexpr = 1
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if K * W * 8 + B * 8 >= 2147483648
            or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
            or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
            or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
            else tl.program_id(0)
        ),
        K // G * tl.cdiv(WIDTH, B) * 2,
        tl.num_programs(0),
    ):
        group = tile // 2 // tl.cdiv(WIDTH, B)
        col_tile = tile // 2 % tl.cdiv(WIDTH, B)
        row_chunk = tile % 2
        c = START + col_tile * B + tl.arange(0, B)
        j = (START + col_tile * B) * 8 + tl.arange(0, B * 8)
        if FULL:
            z = tl.load(Z + group * Z0 + c * Z1)[None, :]
            s = tl.load(S + group * S0 + j * S1)[None, :].to(tl.float16)
        else:
            z = tl.load(Z + group * Z0 + c * Z1, c < W, 0)[None, :]
            s = tl.load(S + group * S0 + j * S1, j < W * 8, 0)[None, :].to(
                tl.float16
            )
        zlo = 269488144 - (z & 252645135)
        zhi = 269488144 - (z >> 4 & 252645135)
        for row_block in range(
            row_chunk * (G // R // 2), (row_chunk + 1) * (G // R // 2)
        ):
            base_row = group * G + row_block * R
            r = base_row + tl.arange(0, R)
            if FULL:
                q = tl.load(Q + r[:, None] * Q0 + c[None, :] * Q1)
            else:
                q = tl.load(
                    Q + r[:, None] * Q0 + c[None, :] * Q1, c[None, :] < W, 0
                )
            lo = (q & 252645135) + zlo
            hi = (q >> 4 & 252645135) + zhi
            lo0 = extract_slice(lo, (0, 0), (R, B // 2), (1, 2))
            hi0 = extract_slice(hi, (0, 0), (R, B // 2), (1, 2))
            lo1 = extract_slice(lo, (0, 1), (R, B // 2), (1, 2))
            hi1 = extract_slice(hi, (0, 1), (R, B // 2), (1, 2))
            lo0_low = lo0.to(tl.int16)
            lo0_high = (lo0 >> 16).to(tl.int16)
            hi0_low = hi0.to(tl.int16)
            hi0_high = (hi0 >> 16).to(tl.int16)
            lo1_low = lo1.to(tl.int16)
            lo1_high = (lo1 >> 16).to(tl.int16)
            hi1_low = hi1.to(tl.int16)
            hi1_high = (hi1 >> 16).to(tl.int16)
            planes = tl.full((16, R * B // 2), 0, tl.int16)
            d0 = (lo0_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d0, (0, 0), (1, R * B // 2), (1, 1))
            d1 = (lo0_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d1, (1, 0), (1, R * B // 2), (1, 1))
            d2 = (hi0_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d2, (2, 0), (1, R * B // 2), (1, 1))
            d3 = (hi0_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d3, (3, 0), (1, R * B // 2), (1, 1))
            d4 = (lo0_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d4, (4, 0), (1, R * B // 2), (1, 1))
            d5 = (lo0_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d5, (5, 0), (1, R * B // 2), (1, 1))
            d6 = (hi0_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d6, (6, 0), (1, R * B // 2), (1, 1))
            d7 = (hi0_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d7, (7, 0), (1, R * B // 2), (1, 1))
            d8 = (lo1_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d8, (8, 0), (1, R * B // 2), (1, 1))
            d9 = (lo1_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d9, (9, 0), (1, R * B // 2), (1, 1))
            d10 = (hi1_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d10, (10, 0), (1, R * B // 2), (1, 1)
            )
            d11 = (hi1_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d11, (11, 0), (1, R * B // 2), (1, 1)
            )
            d12 = (lo1_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d12, (12, 0), (1, R * B // 2), (1, 1)
            )
            d13 = (lo1_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d13, (13, 0), (1, R * B // 2), (1, 1)
            )
            d14 = (hi1_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d14, (14, 0), (1, R * B // 2), (1, 1)
            )
            d15 = (hi1_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d15, (15, 0), (1, R * B // 2), (1, 1)
            )
            value = (
                tl.trans(planes).reshape((R, B * 8)).to(tl.float16) - 16.0
            ).to(tl.float16)
            if FULL:
                tl.store(Output + r[:, None] * W * 8 + j[None, :], value * s)
            else:
                tl.store(
                    Output + r[:, None] * W * 8 + j[None, :],
                    value * s,
                    j[None, :] < W * 8,
                )


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _ordered_group_swar_hoist_balanced(
    Q: "*i32",
    S: "*fp16",
    Z: "*i32",
    Output: "*fp16",
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    FULL: tl.constexpr,
    R: tl.constexpr,
    ALIGN: tl.constexpr,
):
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
    Q0: tl.constexpr = W
    Q1: tl.constexpr = 1
    S0: tl.constexpr = W * 8
    S1: tl.constexpr = 1
    Z0: tl.constexpr = W
    Z1: tl.constexpr = 1
    pid = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 + B * 8 >= 2147483648
        or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
        or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
        or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
        else tl.program_id(0)
    )
    row_blocks: tl.constexpr = G // R
    total_inner: tl.constexpr = K // G * tl.cdiv(WIDTH, B) * row_blocks
    work_per_program = tl.cdiv(total_inner, tl.num_programs(0))
    begin = pid * work_per_program
    end = tl.minimum(begin + work_per_program, total_inner)
    for tile in range(begin // row_blocks, tl.cdiv(end, row_blocks)):
        group = tile // tl.cdiv(WIDTH, B)
        col_tile = tile % tl.cdiv(WIDTH, B)
        c = START + col_tile * B + tl.arange(0, B)
        j = (START + col_tile * B) * 8 + tl.arange(0, B * 8)
        if FULL:
            z = tl.load(Z + group * Z0 + c * Z1)[None, :]
            s = tl.load(S + group * S0 + j * S1)[None, :].to(tl.float16)
        else:
            z = tl.load(Z + group * Z0 + c * Z1, c < W, 0)[None, :]
            s = tl.load(S + group * S0 + j * S1, j < W * 8, 0)[None, :].to(
                tl.float16
            )
        zlo = 269488144 - (z & 252645135)
        zhi = 269488144 - (z >> 4 & 252645135)
        for row_block in range(
            tl.maximum(begin - tile * row_blocks, 0),
            tl.minimum(end - tile * row_blocks, row_blocks),
        ):
            base_row = group * G + row_block * R
            r = base_row + tl.arange(0, R)
            if FULL:
                q = tl.load(Q + r[:, None] * Q0 + c[None, :] * Q1)
            else:
                q = tl.load(
                    Q + r[:, None] * Q0 + c[None, :] * Q1, c[None, :] < W, 0
                )
            lo = (q & 252645135) + zlo
            hi = (q >> 4 & 252645135) + zhi
            lo0 = extract_slice(lo, (0, 0), (R, B // 2), (1, 2))
            hi0 = extract_slice(hi, (0, 0), (R, B // 2), (1, 2))
            lo1 = extract_slice(lo, (0, 1), (R, B // 2), (1, 2))
            hi1 = extract_slice(hi, (0, 1), (R, B // 2), (1, 2))
            lo0_low = lo0.to(tl.int16)
            lo0_high = (lo0 >> 16).to(tl.int16)
            hi0_low = hi0.to(tl.int16)
            hi0_high = (hi0 >> 16).to(tl.int16)
            lo1_low = lo1.to(tl.int16)
            lo1_high = (lo1 >> 16).to(tl.int16)
            hi1_low = hi1.to(tl.int16)
            hi1_high = (hi1 >> 16).to(tl.int16)
            planes = tl.full((16, R * B // 2), 0, tl.int16)
            d0 = (lo0_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d0, (0, 0), (1, R * B // 2), (1, 1))
            d1 = (lo0_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d1, (1, 0), (1, R * B // 2), (1, 1))
            d2 = (hi0_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d2, (2, 0), (1, R * B // 2), (1, 1))
            d3 = (hi0_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d3, (3, 0), (1, R * B // 2), (1, 1))
            d4 = (lo0_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d4, (4, 0), (1, R * B // 2), (1, 1))
            d5 = (lo0_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d5, (5, 0), (1, R * B // 2), (1, 1))
            d6 = (hi0_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d6, (6, 0), (1, R * B // 2), (1, 1))
            d7 = (hi0_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(planes, d7, (7, 0), (1, R * B // 2), (1, 1))
            d8 = (lo1_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d8, (8, 0), (1, R * B // 2), (1, 1))
            d9 = (lo1_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(planes, d9, (9, 0), (1, R * B // 2), (1, 1))
            d10 = (hi1_low >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d10, (10, 0), (1, R * B // 2), (1, 1)
            )
            d11 = (hi1_high >> 0 & 31).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d11, (11, 0), (1, R * B // 2), (1, 1)
            )
            d12 = (lo1_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d12, (12, 0), (1, R * B // 2), (1, 1)
            )
            d13 = (lo1_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d13, (13, 0), (1, R * B // 2), (1, 1)
            )
            d14 = (hi1_low >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d14, (14, 0), (1, R * B // 2), (1, 1)
            )
            d15 = (hi1_high >> 8).reshape((1, R * B // 2))
            planes = insert_slice(
                planes, d15, (15, 0), (1, R * B // 2), (1, 1)
            )
            value = (
                tl.trans(planes).reshape((R, B * 8)).to(tl.float16) - 16.0
            ).to(tl.float16)
            if FULL:
                tl.store(Output + r[:, None] * W * 8 + j[None, :], value * s)
            else:
                tl.store(
                    Output + r[:, None] * W * 8 + j[None, :],
                    value * s,
                    j[None, :] < W * 8,
                )


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _reuse(
    Q: "*i32",
    S: "*fp16",
    Z: "*i32",
    Output: "*fp16",
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    FULL: tl.constexpr,
    R: tl.constexpr,
    ALIGN: tl.constexpr,
):
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
    Q0: tl.constexpr = W
    Q1: tl.constexpr = 1
    S0: tl.constexpr = W * 8
    S1: tl.constexpr = 1
    Z0: tl.constexpr = W
    Z1: tl.constexpr = 1
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if K * W * 8 + B * 8 >= 2147483648
            or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
            or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
            or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
            else tl.program_id(0)
        ),
        K // R * tl.cdiv(WIDTH, B),
        tl.num_programs(0),
    ):
        base_row = tile // tl.cdiv(WIDTH, B) * R
        r = base_row + tl.arange(0, R)
        col_tile = tile % tl.cdiv(WIDTH, B)
        c = START + col_tile * B + tl.arange(0, B)
        if FULL:
            q = tl.load(Q + r[:, None] * Q0 + c[None, :] * Q1)
            z = tl.load(Z + base_row // G * Z0 + c * Z1)[None, :]
        else:
            q = tl.load(
                Q + r[:, None] * Q0 + c[None, :] * Q1, c[None, :] < W, 0
            )
            z = tl.load(Z + base_row // G * Z0 + c * Z1, c < W, 0)[None, :]
        d0 = (q & 15) - (z & 15)
        d1 = (q >> 4 & 15) - (z >> 4 & 15)
        d2 = (q >> 8 & 15) - (z >> 8 & 15)
        d3 = (q >> 12 & 15) - (z >> 12 & 15)
        d4 = (q >> 16 & 15) - (z >> 16 & 15)
        d5 = (q >> 20 & 15) - (z >> 20 & 15)
        d6 = (q >> 24 & 15) - (z >> 24 & 15)
        d7 = (q >> 28 & 15) - (z >> 28 & 15)
        a = tl.interleave(d0, d2)
        b = tl.interleave(d1, d3)
        c0 = tl.interleave(d4, d6)
        d = tl.interleave(d5, d7)
        e = tl.interleave(a, b)
        f = tl.interleave(c0, d)
        value = tl.interleave(e, f).to(tl.float16)
        j = (START + col_tile * B) * 8 + tl.arange(0, B * 8)
        if FULL:
            s = tl.load(S + base_row // G * S0 + j * S1)[None, :]
            tl.store(Output + r[:, None] * W * 8 + j[None, :], value * s)
        else:
            s = tl.load(S + base_row // G * S0 + j * S1, j < W * 8, 0)[None, :]
            tl.store(
                Output + r[:, None] * W * 8 + j[None, :],
                value * s,
                j[None, :] < W * 8,
            )


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _static(
    Q: "*i32",
    S: "*fp16",
    Z: "*i32",
    Output: "*fp16",
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    FULL: tl.constexpr,
    ALIGN: tl.constexpr,
):
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
    Q0: tl.constexpr = W
    Q1: tl.constexpr = 1
    S0: tl.constexpr = W * 8
    S1: tl.constexpr = 1
    Z0: tl.constexpr = W
    Z1: tl.constexpr = 1
    for tile in range(
        (
            tl.program_id(0).to(tl.int64)
            if K * W * 8 + B * 8 >= 2147483648
            or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
            or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
            or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
            else tl.program_id(0)
        ),
        K * tl.cdiv(WIDTH, B),
        tl.num_programs(0),
    ):
        r = tile // tl.cdiv(WIDTH, B)
        col_tile = tile % tl.cdiv(WIDTH, B)
        c = START + col_tile * B + tl.arange(0, B)
        if FULL:
            q = tl.load(Q + r * Q0 + c * Q1)
            z = tl.load(Z + r // G * Z0 + c * Z1)
        else:
            q = tl.load(Q + r * Q0 + c * Q1, c < W, 0)
            z = tl.load(Z + r // G * Z0 + c * Z1, c < W, 0)
        d0 = (q & 15) - (z & 15)
        d1 = (q >> 4 & 15) - (z >> 4 & 15)
        d2 = (q >> 8 & 15) - (z >> 8 & 15)
        d3 = (q >> 12 & 15) - (z >> 12 & 15)
        d4 = (q >> 16 & 15) - (z >> 16 & 15)
        d5 = (q >> 20 & 15) - (z >> 20 & 15)
        d6 = (q >> 24 & 15) - (z >> 24 & 15)
        d7 = (q >> 28 & 15) - (z >> 28 & 15)
        a = tl.interleave(d0, d2)
        b = tl.interleave(d1, d3)
        c0 = tl.interleave(d4, d6)
        d = tl.interleave(d5, d7)
        e = tl.interleave(a, b)
        f = tl.interleave(c0, d)
        value = tl.interleave(e, f).to(tl.float16)
        j = (START + col_tile * B) * 8 + tl.arange(0, B * 8)
        if FULL:
            s = tl.load(S + r // G * S0 + j * S1)
            tl.store(Output + r * W * 8 + j, value * s)
        else:
            s = tl.load(S + r // G * S0 + j * S1, j < W * 8, 0)
            tl.store(Output + r * W * 8 + j, value * s, j < W * 8)


@triton.jit(do_not_specialize=["Q", "S", "Z", "Output"])
def _ordered_float_balanced(
    Q: "*i32",
    S: "*fp16",
    Z: "*i32",
    Output: "*fp16",
    K: tl.constexpr,
    W: tl.constexpr,
    G: tl.constexpr,
    B: tl.constexpr,
    START: tl.constexpr,
    WIDTH: tl.constexpr,
    FULL: tl.constexpr,
    R: tl.constexpr,
    ALIGN: tl.constexpr,
):
    if ALIGN[0]:
        Q = tl.multiple_of(Q, 16)
    if ALIGN[1]:
        S = tl.multiple_of(S, 16)
    if ALIGN[2]:
        Z = tl.multiple_of(Z, 16)
    if ALIGN[3]:
        Output = tl.multiple_of(Output, 16)
    Q0: tl.constexpr = W
    Q1: tl.constexpr = 1
    S0: tl.constexpr = W * 8
    S1: tl.constexpr = 1
    Z0: tl.constexpr = W
    Z1: tl.constexpr = 1
    pid = (
        tl.program_id(0).to(tl.int64)
        if K * W * 8 + B * 8 >= 2147483648
        or (K - 1) * Q0 + (W + B - 1) * Q1 >= 2147483648
        or (K // G - 1) * S0 + (W * 8 + B * 8 - 1) * S1 >= 2147483648
        or ((K // G - 1) * Z0 + (W + B - 1) * Z1 >= 2147483648)
        else tl.program_id(0)
    )
    row_blocks: tl.constexpr = G // R
    total_inner: tl.constexpr = K // G * tl.cdiv(WIDTH, B) * row_blocks
    work_per_program = tl.cdiv(total_inner, tl.num_programs(0))
    begin = pid * work_per_program
    end = tl.minimum(begin + work_per_program, total_inner)
    for tile in range(begin // row_blocks, tl.cdiv(end, row_blocks)):
        group = tile // tl.cdiv(WIDTH, B)
        col_tile = tile % tl.cdiv(WIDTH, B)
        c = START + col_tile * B + tl.arange(0, B)
        j = (START + col_tile * B) * 8 + tl.arange(0, B * 8)
        if FULL:
            z = tl.load(Z + group * Z0 + c * Z1)[None, :]
            s = tl.load(S + group * S0 + j * S1)[None, :]
        else:
            z = tl.load(Z + group * Z0 + c * Z1, c < W, 0)[None, :]
            s = tl.load(S + group * S0 + j * S1, j < W * 8, 0)[None, :]
        zlo = 269488144 - (z & 252645135)
        zhi = 269488144 - (z >> 4 & 252645135)
        for row_block in range(
            tl.maximum(begin - tile * row_blocks, 0),
            tl.minimum(end - tile * row_blocks, row_blocks),
        ):
            base_row = group * G + row_block * R
            r = base_row + tl.arange(0, R)
            if FULL:
                q = tl.load(Q + r[:, None] * Q0 + c[None, :] * Q1)
            else:
                q = tl.load(
                    Q + r[:, None] * Q0 + c[None, :] * Q1, c[None, :] < W, 0
                )
            lo = (q & 252645135) + zlo
            hi = (q >> 4 & 252645135) + zhi
            d0 = (lo >> 0 & 31).to(tl.int32).reshape((R * B,))
            d1 = (hi >> 0 & 31).to(tl.int32).reshape((R * B,))
            d2 = (lo >> 8 & 31).to(tl.int32).reshape((R * B,))
            d3 = (hi >> 8 & 31).to(tl.int32).reshape((R * B,))
            d4 = (lo >> 16 & 31).to(tl.int32).reshape((R * B,))
            d5 = (hi >> 16 & 31).to(tl.int32).reshape((R * B,))
            d6 = (lo >> 24).to(tl.int32).reshape((R * B,))
            d7 = (hi >> 24).to(tl.int32).reshape((R * B,))
            planes = tl.full((8, R * B), 0, tl.int32)
            planes = insert_slice(
                planes, d0.reshape((1, R * B)), (0, 0), (1, R * B), (1, 1)
            )
            planes = insert_slice(
                planes, d4.reshape((1, R * B)), (1, 0), (1, R * B), (1, 1)
            )
            planes = insert_slice(
                planes, d1.reshape((1, R * B)), (2, 0), (1, R * B), (1, 1)
            )
            planes = insert_slice(
                planes, d5.reshape((1, R * B)), (3, 0), (1, R * B), (1, 1)
            )
            planes = insert_slice(
                planes, d2.reshape((1, R * B)), (4, 0), (1, R * B), (1, 1)
            )
            planes = insert_slice(
                planes, d6.reshape((1, R * B)), (5, 0), (1, R * B), (1, 1)
            )
            planes = insert_slice(
                planes, d3.reshape((1, R * B)), (6, 0), (1, R * B), (1, 1)
            )
            planes = insert_slice(
                planes, d7.reshape((1, R * B)), (7, 0), (1, R * B), (1, 1)
            )
            value = (
                tl.trans(planes.reshape((8, R * B)))
                .reshape((R, B * 8))
                .to(tl.float16)
                - 16.0
            ).to(tl.float16)
            if FULL:
                tl.store(Output + r[:, None] * W * 8 + j[None, :], value * s)
            else:
                tl.store(
                    Output + r[:, None] * W * 8 + j[None, :],
                    value * s,
                    j[None, :] < W * 8,
                )


def awq_dequantize(qweight, scales, zeros):
    qweight = qweight.contiguous()
    scales = scales.contiguous()
    zeros = zeros.contiguous()
    k, w = qweight.shape
    g = k // scales.shape[0]
    rows = (
        16
        if k % 16 == 0 and g % 16 == 0
        else 8 if k % 8 == 0 and g % 8 == 0 else 1
    )
    use_ordered = rows == 16
    use_aspect = use_ordered and w % 128 == 0
    if use_aspect:
        rows = 4 if w >= 256 else 8
    block = (
        min(64, max(16, 0 if w == 0 else 1 << (w - 1).bit_length()))
        if rows > 1
        else min(512, max(128, 0 if w == 0 else 1 << (w - 1).bit_length()))
    )
    if use_aspect:
        block = min(
            256 if w >= 256 else 128,
            max(16, 0 if w == 0 else 1 << (w - 1).bit_length()),
        )
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
    kernel = _ordered_float if use_ordered else _reuse
    full = w // block * block
    if full:
        full_kernel = (
            _ordered_group_swar_hoist
            if use_ordered and k // g * (full // block) >= 40
            else kernel
        )
        wide_group = (
            use_ordered
            and rows == 4
            and (full % 512 == 0)
            and (k // rows * (full // 512) >= 40)
        )
        group_rows = (
            4
            if wide_group
            else (
                8
                if full_kernel is _ordered_group_swar_hoist and rows == 4
                else rows
            )
        )
        group_block = 512 if wide_group else block
        if wide_group:
            full_kernel = _ordered_group_swar_hoist_balanced
        elif (
            full_kernel is _ordered_float and k // rows * (full // block) >= 40
        ):
            full_kernel = _ordered_float_balanced
        grid = (min(40, k // rows * (full // block)),)
        if rows > 1:
            full_kernel.run(
                q_ptr,
                s_ptr,
                z_ptr,
                o_ptr,
                k,
                w,
                g,
                group_block,
                0,
                full,
                True,
                group_rows,
                align,
                grid=grid,
                warmup=False,
                num_stages=(
                    2
                    if use_aspect or full_kernel is _ordered_float_balanced
                    else 1 if use_ordered else 2
                ),
            )
        else:
            _static.run(
                q_ptr,
                s_ptr,
                z_ptr,
                o_ptr,
                k,
                w,
                g,
                block,
                0,
                full,
                True,
                align,
                grid=grid,
                warmup=False,
                num_stages=2 if use_aspect else 1 if use_ordered else 2,
            )
    tail = w - full
    if tail:
        tail_block = max(
            16 if rows > 1 else 128,
            0 if tail == 0 else 1 << (tail - 1).bit_length(),
        )
        grid = (min(40, k // rows),)
        if rows > 1:
            kernel.run(
                q_ptr,
                s_ptr,
                z_ptr,
                o_ptr,
                k,
                w,
                g,
                tail_block,
                full,
                tail,
                tail == tail_block,
                rows,
                align,
                grid=grid,
                warmup=False,
                num_stages=2 if use_aspect else 1 if use_ordered else 2,
            )
        else:
            _static.run(
                q_ptr,
                s_ptr,
                z_ptr,
                o_ptr,
                k,
                w,
                g,
                tail_block,
                full,
                tail,
                tail == tail_block,
                align,
                grid=grid,
                warmup=False,
                num_stages=2 if use_aspect else 1 if use_ordered else 2,
            )
    return out


__all__ = ["awq_dequantize"]
