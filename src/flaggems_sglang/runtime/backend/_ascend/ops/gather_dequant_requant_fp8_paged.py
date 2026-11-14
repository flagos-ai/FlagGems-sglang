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
from triton.language.extra.cann.extension import insert_slice


@triton.jit
def _decode(raw):
    raw = raw.to(tl.int32)
    mag = raw & 127
    normal = (mag + 960 << 20).to(tl.float32, bitcast=True)
    sub = tl.fma(normal, 2.0, -0.015625)
    value = tl.minimum(
        sub.to(tl.uint32, bitcast=True), normal.to(tl.uint32, bitcast=True)
    ).to(tl.float32, bitcast=True)
    bits = tl.where(mag == 127, 2143289344, value.to(tl.int32, bitcast=True))
    bits = bits | (raw & 128) << 24
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _encode_word_lane(value):
    bits = value.to(tl.int32, bitcast=True)
    mag = tl.minimum(bits & 2147483647, 1139802112)
    bias = (tl.maximum(mag, 1015021568) & 2139095040) + 167772160
    rounded = mag.to(tl.float32, bitcast=True) + bias.to(
        tl.float32, bitcast=True
    )
    code = (
        rounded.to(tl.int32, bitcast=True).to(tl.int16)
        + (bias >> 20).to(tl.int16)
        - 1128
    )
    sign = (bits >> 24 & 128).to(tl.int16)
    return code | sign


@triton.jit
def _encode(value):
    bits = value.to(tl.int32, bitcast=True)
    mag = tl.minimum(bits & 2147483647, 1139802112)
    exponent = tl.maximum(mag >> 23, 121)
    bias = exponent + 20 << 23
    rounded = mag.to(tl.float32, bitcast=True) + bias.to(
        tl.float32, bitcast=True
    )
    code = (
        rounded.to(tl.int32, bitcast=True).to(tl.int16)
        + exponent.to(tl.int16) * 8
        + 56
    )
    sign = (bits >> 24 & 128).to(tl.int16)
    return (code | sign).to(tl.uint8)


@triton.jit
def _kernel(Q, S, R, P, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    TOTAL: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    QS: tl.constexpr = META[3]
    SS: tl.constexpr = META[4]
    RS: tl.constexpr = META[5]
    PS: tl.constexpr = META[6]
    OS: tl.constexpr = META[7]
    G: tl.constexpr = META[8]
    col = tl.arange(0, 512)
    tail = tl.arange(0, 64)
    if N >= 4096:
        chunk = tl.cdiv(TOTAL, tl.num_programs(0))
        begin = tl.program_id(0) * chunk
        limit = tl.minimum(begin + chunk, TOTAL)
        step = 1
    else:
        begin = tl.program_id(0)
        limit = TOTAL
        step = tl.num_programs(0)
    for row in range(begin, tl.minimum(limit, N), step):
        src = tl.load(P + row * PS).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        raw = tl.load(Q + src * QS + col)
        scales = tl.load(S + src * SS + 128 + col // G)
        value = _decode(raw) * scales
        tl.store(OUT + row * OS + col, _encode(value))
        rope = tl.load(R + src * RS + (512 + 512 // G * 4) // 2 + tail).to(
            tl.float32
        )
        tl.store(OUT + row * OS + 512 + tail, _encode(rope))
    for row in range(
        begin + tl.cdiv(tl.maximum(N - begin, 0), step) * step, limit, step
    ):
        tl.store(OUT + row * OS + col, tl.full((512,), 0, tl.uint8))
        tl.store(OUT + row * OS + 512 + tail, tl.full((64,), 0, tl.uint8))


def _fast(quant_k_cache, page_table_1_flattened, group_size, extra_rows, out):
    cache = quant_k_cache
    scales = cache.view(torch.float32)
    rope = cache.view(torch.bfloat16)
    n = page_table_1_flattened.numel()
    total = n + extra_rows
    if total:
        _kernel[min(total, 80),](
            cache.view(torch.uint8),
            scales,
            rope,
            page_table_1_flattened,
            out.view(torch.uint8),
            (
                n,
                total,
                cache.shape[0],
                cache.stride(0),
                cache.stride(0) // 4,
                cache.stride(0) // 2,
                page_table_1_flattened.stride(0),
                out.stride(0),
                group_size,
            ),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out


@triton.jit
def _layout_pack(Q, P, T, PAGE_IDS, META: tl.constexpr):
    N: tl.constexpr = META[0]
    M: tl.constexpr = META[1]
    QS: tl.constexpr = META[2]
    QC: tl.constexpr = META[3]
    PS: tl.constexpr = META[4]
    a = tl.arange(0, 512).to(tl.int64)
    b = 512 + tl.arange(0, 128).to(tl.int64)
    c = 640 + tl.arange(0, 16).to(tl.int64)
    for row in range(tl.program_id(0), N, tl.num_programs(0)):
        r = row.to(tl.int64)
        src = tl.load(P + r * PS).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        qa = tl.load(Q + src * QS + a * QC)
        qb = tl.load(Q + src * QS + b * QC)
        qc = tl.load(Q + src * QS + c * QC)
        tl.store(T + r * 656 + a, qa)
        tl.store(T + r * 656 + b, qb)
        tl.store(T + r * 656 + c, qc)
        tl.store(PAGE_IDS + r, row)


@triton.jit
def _layout_scatter(
    T, OUT, TOTAL: tl.constexpr, OS: tl.constexpr, OC: tl.constexpr
):
    a = tl.arange(0, 512).to(tl.int64)
    b = 512 + tl.arange(0, 64).to(tl.int64)
    for row in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        r = row.to(tl.int64)
        qa = tl.load(T + r * 576 + a)
        qb = tl.load(T + r * 576 + b)
        tl.store(OUT + r * OS + a * OC, qa)
        tl.store(OUT + r * OS + b * OC, qb)


def _layout_copy_call(
    quant_k_cache, page_table_1_flattened, group_size, extra_rows, out
):
    n = page_table_1_flattened.numel()
    total = n + extra_rows
    packed = torch.empty((n, 1, 656), dtype=torch.uint8, device=out.device)
    pages = torch.empty((n,), dtype=torch.int32, device=out.device)
    temp = torch.empty((total, 1, 576), dtype=out.dtype, device=out.device)
    if n:
        _layout_pack[min(n, 80),](
            quant_k_cache.view(torch.uint8),
            page_table_1_flattened,
            packed,
            pages,
            (
                n,
                quant_k_cache.shape[0],
                quant_k_cache.stride(0),
                quant_k_cache.stride(-1),
                page_table_1_flattened.stride(0),
            ),
            num_warps=4,
        )
    _fast(
        packed.view(quant_k_cache.dtype), pages, group_size, extra_rows, temp
    )
    if total:
        _layout_scatter[min(total, 80),](
            temp.view(torch.uint8),
            out.view(torch.uint8),
            total,
            out.stride(0),
            out.stride(-1),
            num_warps=4,
        )
    return out


@triton.jit
def _stage(Q, S, R, P, OUT, META: tl.constexpr):
    N: tl.constexpr = META[0]
    TOTAL: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    QS: tl.constexpr = META[3]
    SS: tl.constexpr = META[4]
    RS: tl.constexpr = META[5]
    PS: tl.constexpr = META[6]
    OS: tl.constexpr = META[7]
    G: tl.constexpr = META[8]
    col = tl.arange(0, 512)
    tail = tl.arange(0, 64)
    chunk = tl.cdiv(TOTAL, tl.num_programs(0))
    begin = tl.program_id(0) * chunk
    for first in range(
        begin,
        begin + tl.maximum(tl.minimum(begin + chunk, N) - begin, 0) // 8 * 8,
        8,
    ):
        src0 = tl.load(P + (first + 0) * PS).to(tl.int64)
        src0 = tl.where(src0 < 0, src0 + M, src0)
        raw0 = tl.load(Q + src0 * QS + col)
        scales0 = tl.load(S + src0 * SS + 128 + col // G)
        rope0 = tl.load(R + src0 * RS + (512 + 512 // G * 4) // 2 + tail).to(
            tl.float32
        )
        src1 = tl.load(P + (first + 1) * PS).to(tl.int64)
        src1 = tl.where(src1 < 0, src1 + M, src1)
        raw1 = tl.load(Q + src1 * QS + col)
        scales1 = tl.load(S + src1 * SS + 128 + col // G)
        rope1 = tl.load(R + src1 * RS + (512 + 512 // G * 4) // 2 + tail).to(
            tl.float32
        )
        src2 = tl.load(P + (first + 2) * PS).to(tl.int64)
        src2 = tl.where(src2 < 0, src2 + M, src2)
        raw2 = tl.load(Q + src2 * QS + col)
        scales2 = tl.load(S + src2 * SS + 128 + col // G)
        rope2 = tl.load(R + src2 * RS + (512 + 512 // G * 4) // 2 + tail).to(
            tl.float32
        )
        src3 = tl.load(P + (first + 3) * PS).to(tl.int64)
        src3 = tl.where(src3 < 0, src3 + M, src3)
        raw3 = tl.load(Q + src3 * QS + col)
        scales3 = tl.load(S + src3 * SS + 128 + col // G)
        rope3 = tl.load(R + src3 * RS + (512 + 512 // G * 4) // 2 + tail).to(
            tl.float32
        )
        src4 = tl.load(P + (first + 4) * PS).to(tl.int64)
        src4 = tl.where(src4 < 0, src4 + M, src4)
        raw4 = tl.load(Q + src4 * QS + col)
        scales4 = tl.load(S + src4 * SS + 128 + col // G)
        rope4 = tl.load(R + src4 * RS + (512 + 512 // G * 4) // 2 + tail).to(
            tl.float32
        )
        src5 = tl.load(P + (first + 5) * PS).to(tl.int64)
        src5 = tl.where(src5 < 0, src5 + M, src5)
        raw5 = tl.load(Q + src5 * QS + col)
        scales5 = tl.load(S + src5 * SS + 128 + col // G)
        rope5 = tl.load(R + src5 * RS + (512 + 512 // G * 4) // 2 + tail).to(
            tl.float32
        )
        src6 = tl.load(P + (first + 6) * PS).to(tl.int64)
        src6 = tl.where(src6 < 0, src6 + M, src6)
        raw6 = tl.load(Q + src6 * QS + col)
        scales6 = tl.load(S + src6 * SS + 128 + col // G)
        rope6 = tl.load(R + src6 * RS + (512 + 512 // G * 4) // 2 + tail).to(
            tl.float32
        )
        src7 = tl.load(P + (first + 7) * PS).to(tl.int64)
        src7 = tl.where(src7 < 0, src7 + M, src7)
        raw7 = tl.load(Q + src7 * QS + col)
        scales7 = tl.load(S + src7 * SS + 128 + col // G)
        rope7 = tl.load(R + src7 * RS + (512 + 512 // G * 4) // 2 + tail).to(
            tl.float32
        )
        value0 = _decode(raw0) * scales0
        tl.store(OUT + (first + 0) * OS + col, value0)
        tl.store(OUT + (first + 0) * OS + 512 + tail, rope0)
        value1 = _decode(raw1) * scales1
        tl.store(OUT + (first + 1) * OS + col, value1)
        tl.store(OUT + (first + 1) * OS + 512 + tail, rope1)
        value2 = _decode(raw2) * scales2
        tl.store(OUT + (first + 2) * OS + col, value2)
        tl.store(OUT + (first + 2) * OS + 512 + tail, rope2)
        value3 = _decode(raw3) * scales3
        tl.store(OUT + (first + 3) * OS + col, value3)
        tl.store(OUT + (first + 3) * OS + 512 + tail, rope3)
        value4 = _decode(raw4) * scales4
        tl.store(OUT + (first + 4) * OS + col, value4)
        tl.store(OUT + (first + 4) * OS + 512 + tail, rope4)
        value5 = _decode(raw5) * scales5
        tl.store(OUT + (first + 5) * OS + col, value5)
        tl.store(OUT + (first + 5) * OS + 512 + tail, rope5)
        value6 = _decode(raw6) * scales6
        tl.store(OUT + (first + 6) * OS + col, value6)
        tl.store(OUT + (first + 6) * OS + 512 + tail, rope6)
        value7 = _decode(raw7) * scales7
        tl.store(OUT + (first + 7) * OS + col, value7)
        tl.store(OUT + (first + 7) * OS + 512 + tail, rope7)
    for row in range(
        begin + tl.maximum(tl.minimum(begin + chunk, N) - begin, 0) // 8 * 8,
        tl.minimum(begin + chunk, N),
    ):
        src = tl.load(P + row * PS).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        raw = tl.load(Q + src * QS + col)
        scales = tl.load(S + src * SS + 128 + col // G)
        value = _decode(raw) * scales
        tl.store(OUT + row * OS + col, value)
        rope = tl.load(R + src * RS + (512 + 512 // G * 4) // 2 + tail).to(
            tl.float32
        )
        tl.store(OUT + row * OS + 512 + tail, rope)
    for row in range(tl.maximum(begin, N), tl.minimum(begin + chunk, TOTAL)):
        tl.store(OUT + row * OS + col, tl.full((512,), 0, tl.float32))
        tl.store(OUT + row * OS + 512 + tail, tl.full((64,), 0, tl.float32))


@triton.jit
def _encode_flat(T, OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.arange(0, BLOCK)
    for start in range(
        tl.program_id(0) * BLOCK, SIZE, tl.num_programs(0) * BLOCK
    ):
        index = start + col
        value = tl.load(T + index, index < SIZE, 0)
        tl.store(OUT + index, _encode(value), index < SIZE)


@triton.jit
def _encode_wide(value):
    bits = value.to(tl.int32, bitcast=True)
    mag = tl.minimum(bits & 2147483647, 1139802112)
    exponent = tl.maximum(mag >> 23, 121)
    bias = exponent + 20 << 23
    rounded = mag.to(tl.float32, bitcast=True) + bias.to(
        tl.float32, bitcast=True
    )
    code = rounded.to(tl.int32, bitcast=True) - exponent * 8388600 - 167773128
    return code | bits >> 24 & 128


@triton.jit
def _contiguous_fused(Q, S, R, P, OW, META: tl.constexpr):
    N: tl.constexpr = META[0]
    TOTAL: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    QS: tl.constexpr = META[3]
    PS: tl.constexpr = META[4]
    OS: tl.constexpr = META[5]
    col = tl.arange(0, 512)
    tail = tl.arange(0, 64)
    chunk = tl.cdiv(TOTAL, tl.num_programs(0))
    begin = tl.program_id(0) * chunk
    limit = tl.minimum(begin + chunk, TOTAL)
    for first in range(
        begin, begin + tl.maximum(tl.minimum(limit, N) - begin, 0) // 7 * 7, 7
    ):
        if N >= 8192:
            pass
        else:
            packed_raw = tl.full((4096,), 0, tl.int32)
        src0 = tl.load(P + (first + 0) * PS).to(tl.int64)
        src0 = tl.where(src0 < 0, src0 + M, src0)
        raw0 = tl.load(Q + src0 * QS + col)
        if N >= 8192:
            packed_raw = tl.broadcast_to(
                raw0.to(tl.int32)[None, :], (8, 512)
            ).reshape((4096,))
        else:
            packed_raw = insert_slice(
                packed_raw, raw0.to(tl.int32), (0,), (512,), (1,)
            )
        src1 = tl.load(P + (first + 1) * PS).to(tl.int64)
        src1 = tl.where(src1 < 0, src1 + M, src1)
        raw1 = tl.load(Q + src1 * QS + col)
        packed_raw = insert_slice(
            packed_raw, raw1.to(tl.int32), (576,), (512,), (1,)
        )
        src2 = tl.load(P + (first + 2) * PS).to(tl.int64)
        src2 = tl.where(src2 < 0, src2 + M, src2)
        raw2 = tl.load(Q + src2 * QS + col)
        packed_raw = insert_slice(
            packed_raw, raw2.to(tl.int32), (1152,), (512,), (1,)
        )
        src3 = tl.load(P + (first + 3) * PS).to(tl.int64)
        src3 = tl.where(src3 < 0, src3 + M, src3)
        raw3 = tl.load(Q + src3 * QS + col)
        packed_raw = insert_slice(
            packed_raw, raw3.to(tl.int32), (1728,), (512,), (1,)
        )
        src4 = tl.load(P + (first + 4) * PS).to(tl.int64)
        src4 = tl.where(src4 < 0, src4 + M, src4)
        raw4 = tl.load(Q + src4 * QS + col)
        packed_raw = insert_slice(
            packed_raw, raw4.to(tl.int32), (2304,), (512,), (1,)
        )
        src5 = tl.load(P + (first + 5) * PS).to(tl.int64)
        src5 = tl.where(src5 < 0, src5 + M, src5)
        raw5 = tl.load(Q + src5 * QS + col)
        packed_raw = insert_slice(
            packed_raw, raw5.to(tl.int32), (2880,), (512,), (1,)
        )
        src6 = tl.load(P + (first + 6) * PS).to(tl.int64)
        src6 = tl.where(src6 < 0, src6 + M, src6)
        raw6 = tl.load(Q + src6 * QS + col)
        packed_raw = insert_slice(
            packed_raw, raw6.to(tl.int32), (3456,), (512,), (1,)
        )
        decoded = _decode(packed_raw)
        if N >= 8192:
            packed_scales = packed_raw.to(tl.float32, bitcast=True)
        else:
            packed_scales = tl.full((4096,), 0, tl.float32)
        scales0 = tl.load(S + src0 * (QS // 4) + 128 + col // 128)
        packed_scales = insert_slice(
            packed_scales, scales0, (0,), (512,), (1,)
        )
        scales1 = tl.load(S + src1 * (QS // 4) + 128 + col // 128)
        packed_scales = insert_slice(
            packed_scales, scales1, (576,), (512,), (1,)
        )
        scales2 = tl.load(S + src2 * (QS // 4) + 128 + col // 128)
        packed_scales = insert_slice(
            packed_scales, scales2, (1152,), (512,), (1,)
        )
        scales3 = tl.load(S + src3 * (QS // 4) + 128 + col // 128)
        packed_scales = insert_slice(
            packed_scales, scales3, (1728,), (512,), (1,)
        )
        scales4 = tl.load(S + src4 * (QS // 4) + 128 + col // 128)
        packed_scales = insert_slice(
            packed_scales, scales4, (2304,), (512,), (1,)
        )
        scales5 = tl.load(S + src5 * (QS // 4) + 128 + col // 128)
        packed_scales = insert_slice(
            packed_scales, scales5, (2880,), (512,), (1,)
        )
        scales6 = tl.load(S + src6 * (QS // 4) + 128 + col // 128)
        packed_scales = insert_slice(
            packed_scales, scales6, (3456,), (512,), (1,)
        )
        combined = decoded * packed_scales
        rope0 = tl.load(R + src0 * (QS // 2) + 264 + tail).to(tl.float32)
        combined = insert_slice(combined, rope0, (512,), (64,), (1,))
        rope1 = tl.load(R + src1 * (QS // 2) + 264 + tail).to(tl.float32)
        combined = insert_slice(combined, rope1, (1088,), (64,), (1,))
        rope2 = tl.load(R + src2 * (QS // 2) + 264 + tail).to(tl.float32)
        combined = insert_slice(combined, rope2, (1664,), (64,), (1,))
        rope3 = tl.load(R + src3 * (QS // 2) + 264 + tail).to(tl.float32)
        combined = insert_slice(combined, rope3, (2240,), (64,), (1,))
        rope4 = tl.load(R + src4 * (QS // 2) + 264 + tail).to(tl.float32)
        combined = insert_slice(combined, rope4, (2816,), (64,), (1,))
        rope5 = tl.load(R + src5 * (QS // 2) + 264 + tail).to(tl.float32)
        combined = insert_slice(combined, rope5, (3392,), (64,), (1,))
        rope6 = tl.load(R + src6 * (QS // 2) + 264 + tail).to(tl.float32)
        combined = insert_slice(combined, rope6, (3968,), (64,), (1,))
        lanes = _encode_word_lane(combined).reshape((2048, 2))
        low, high = tl.split(lanes)
        words = low | high << 8
        linear = tl.arange(0, 2048)
        target = OW + first * (OS // 2)
        tl.store(target + linear, words.to(tl.uint16), linear < 2016)
    for row in range(
        begin + tl.maximum(tl.minimum(limit, N) - begin, 0) // 7 * 7,
        tl.minimum(limit, N),
    ):
        src = tl.load(P + row * PS).to(tl.int64)
        src = tl.where(src < 0, src + M, src)
        raw = tl.load(Q + src * QS + col)
        scales = tl.load(S + src * (QS // 4) + 128 + col // 128)
        rope = tl.load(R + src * (QS // 2) + 264 + tail).to(tl.float32)
        nope_codes = _encode_wide(_decode(raw) * scales)
        rope_codes = _encode_wide(rope)
        combined = tl.full((1024,), 0, tl.int32)
        combined = insert_slice(combined, nope_codes, (0,), (512,), (1,))
        combined = insert_slice(combined, rope_codes, (512,), (64,), (1,))
        lanes = combined.to(tl.uint16).reshape((512, 2))
        low, high = tl.split(lanes)
        packed = low | high << 8
        linear_tail = tl.arange(0, 512)
        tl.store(OW + row * (OS // 2) + linear_tail, packed, linear_tail < 288)
    for row in range(tl.maximum(begin, N), limit):
        linear_tail = tl.arange(0, 512)
        tl.store(
            OW + row * (OS // 2) + linear_tail,
            tl.full((512,), 0, tl.uint16),
            linear_tail < 288,
        )


def gather_dequant_requant_fp8_paged(
    quant_k_cache, page_table_1_flattened, group_size, extra_rows, out
):
    n = page_table_1_flattened.numel()
    total = n + extra_rows
    if total == 0:
        return out
    qs = quant_k_cache.stride()
    os = out.stride()
    ps = page_table_1_flattened.stride(0)
    m = quant_k_cache.shape[0]
    qp = quant_k_cache.data_ptr()
    start = out.data_ptr()
    if (
        n >= 1024
        and group_size == 128
        and quant_k_cache.is_contiguous()
        and out.is_contiguous()
        and ((qp | start) & 3 == 0)
        and (m * qs[0] < 2147483648)
        and (total * 576 < 2147483648)
        and (n * ps < 2147483648)
    ):
        pp = page_table_1_flattened.data_ptr()
        end = start + total * 576
        if not (
            start < qp + m * qs[0]
            and qp < end
            or (start < pp + (n - 1) * ps * 4 + 4 and pp < end)
        ):
            _contiguous_fused[40,](
                quant_k_cache.view(torch.uint8),
                quant_k_cache.view(torch.float32),
                quant_k_cache.view(torch.bfloat16),
                page_table_1_flattened,
                out.view(torch.uint16),
                (n, total, m, qs[0], ps, 576),
                num_warps=4,
                enable_fp_fusion=False,
                multibuffer=n >= 8192,
            )
            return out
    if (
        n >= 8192
        and quant_k_cache.is_contiguous()
        and out.is_contiguous()
        and ((qp | start) & 3 == 0)
        and (m * qs[0] < 2147483648)
        and (total * 576 < 2147483648)
        and (n * ps < 2147483648)
    ):
        temp = torch.empty(
            (total, 1, 576), dtype=torch.float32, device=out.device
        )
        _stage[min(total, 80),](
            quant_k_cache.view(torch.uint8),
            quant_k_cache.view(torch.float32),
            quant_k_cache.view(torch.bfloat16),
            page_table_1_flattened,
            temp,
            (n, total, m, qs[0], qs[0] // 4, qs[0] // 2, ps, 576, group_size),
            num_warps=4,
            enable_fp_fusion=False,
            multibuffer=False,
        )
        _encode_flat[min(triton.cdiv(total * 576, 4096), 80),](
            temp,
            out.view(torch.uint8),
            total * 576,
            4096,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return out
    aligned = (
        qs[-1] == 1
        and os[-1] == 1
        and ((qp | start | qs[0] | qs[1] | os[0] | os[1]) & 3 == 0)
    )
    compatible = (
        aligned
        and m * qs[0] < 2147483648
        and (out.shape[0] * os[0] < 2147483648)
        and (n * ps < 2147483648)
    )
    if not compatible:
        return _layout_copy_call(
            quant_k_cache, page_table_1_flattened, group_size, extra_rows, out
        )
    target = out
    if n:
        pp = page_table_1_flattened.data_ptr()
        end = start + (total - 1) * os[0] + 575 * os[-1] + 1
        qe = qp + (m - 1) * qs[0] + 655 * qs[-1] + 1
        pe = pp + (n - 1) * ps * 4 + 4
        if start < qe and qp < end or (start < pe and pp < end):
            target = torch.empty(
                (total, 1, 576), dtype=out.dtype, device=out.device
            )
    _kernel[min(total, 80),](
        quant_k_cache.view(torch.uint8),
        quant_k_cache.view(torch.float32),
        quant_k_cache.view(torch.bfloat16),
        page_table_1_flattened,
        target.view(torch.uint8),
        (
            n,
            total,
            m,
            qs[0],
            qs[0] // 4,
            qs[0] // 2,
            ps,
            target.stride(0),
            group_size,
        ),
        num_warps=4,
        enable_fp_fusion=False,
    )
    if target is not out:
        _layout_scatter[min(total, 80),](
            target.view(torch.uint8),
            out.view(torch.uint8),
            total,
            os[0],
            os[-1],
            num_warps=4,
        )
    return out


__all__ = ["gather_dequant_requant_fp8_paged"]
