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
from triton.language.extra.cann import extension as al


@triton.jit
def _encode(value):
    raw = value.to(tl.int32, bitcast=True)
    magnitude = tl.minimum(
        tl.abs(value).to(tl.int32, bitcast=True), 1139802112
    )
    exponent = tl.maximum(magnitude >> 23, 121)
    magic_bits = exponent * 8388616 + 167771192
    magic = magic_bits.to(tl.float32, bitcast=True)
    rounded = (
        (magnitude.to(tl.float32, bitcast=True) + magic)
        .to(tl.int32, bitcast=True)
        .to(tl.int16)
    )
    code = rounded
    sign = (raw >> 24 & 128).to(tl.int16)
    return al.cast(code | sign, tl.uint8, overflow_mode="saturate")


@triton.jit
def _maximum_propagate(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _quantize(X, Q, S, MASKED, EPS, META: tl.constexpr):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    GROUP: tl.constexpr = META[2]
    ALIGNED_M: tl.constexpr = META[3]
    STRIDE_B: tl.constexpr = META[4]
    STRIDE_M: tl.constexpr = META[5]
    STRIDE_K: tl.constexpr = META[6]
    ROWS: tl.constexpr = META[7]
    BLOCK: tl.constexpr = META[8]
    B: tl.constexpr = META[9]
    WIDE: tl.constexpr = META[10]
    GROUPS: tl.constexpr = K // GROUP
    ROW_TILES: tl.constexpr = triton.cdiv(M, ROWS)
    start = tl.program_id(0)
    if WIDE:
        start = start.to(tl.int64)
    for tile in range(start, B * GROUPS * ROW_TILES, tl.num_programs(0)):
        row_tile = tile % ROW_TILES
        group = tile // ROW_TILES % GROUPS
        batch = tile // (ROW_TILES * GROUPS)
        tokens = row_tile * ROWS + tl.arange(0, ROWS)
        lanes = tl.arange(0, BLOCK)
        columns = group * GROUP + lanes
        value = tl.load(
            X
            + batch * STRIDE_B
            + tokens[:, None] * STRIDE_M
            + columns[None, :] * STRIDE_K,
            (tokens[:, None] < M) & (lanes[None, :] < GROUP),
            other=0.0,
        ).to(tl.float32)
        maximum = tl.maximum(
            tl.reduce(tl.abs(value), axis=1, combine_fn=_maximum_propagate),
            EPS,
            propagate_nan=tl.PropagateNan.ALL,
        )
        scale = tl.div_rn(maximum, 448.0)
        divided = tl.div_rn(value, scale[:, None])
        quantized = tl.minimum(
            tl.maximum(divided, -448.0, propagate_nan=tl.PropagateNan.ALL),
            448.0,
            propagate_nan=tl.PropagateNan.ALL,
        )
        tl.store(
            Q + batch * ALIGNED_M * K + tokens[:, None] * K + columns[None, :],
            _encode(quantized),
            (tokens[:, None] < M) & (lanes[None, :] < GROUP),
        )
        tl.store(
            S + batch * GROUPS * ALIGNED_M + group * ALIGNED_M + tokens,
            scale,
            tokens < M,
        )
        if row_tile == 0 and group == 0:
            tl.store(MASKED + batch, M)


@triton.jit
def _wide_quantize(X, Q, S, MASKED, EPS, META: tl.constexpr):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    GROUP: tl.constexpr = META[2]
    ALIGNED_M: tl.constexpr = META[3]
    STRIDE_B: tl.constexpr = META[4]
    STRIDE_M: tl.constexpr = META[5]
    STRIDE_K: tl.constexpr = META[6]
    ROWS: tl.constexpr = META[7]
    B: tl.constexpr = META[9]
    WIDE: tl.constexpr = META[10]
    GROUPS: tl.constexpr = K // GROUP
    ROW_TILES: tl.constexpr = triton.cdiv(M, ROWS)
    start = tl.program_id(0)
    if WIDE:
        start = start.to(tl.int64)
    for tile in range(start, B * GROUPS * ROW_TILES, tl.num_programs(0)):
        row_tile = tile % ROW_TILES
        group = tile // ROW_TILES % GROUPS
        batch = tile // (ROW_TILES * GROUPS)
        tokens = row_tile * ROWS + tl.arange(0, ROWS)
        lanes = tl.arange(0, 128)
        maximum = tl.full((ROWS, 128), 0.0, tl.float32)
        for chunk in range(triton.cdiv(GROUP, 128)):
            in_group = chunk * 128 + lanes
            columns = group * GROUP + in_group
            value = tl.load(
                X
                + batch * STRIDE_B
                + tokens[:, None] * STRIDE_M
                + columns[None, :] * STRIDE_K,
                (tokens[:, None] < M) & (in_group[None, :] < GROUP),
                other=0.0,
            ).to(tl.float32)
            maximum = tl.maximum(
                maximum, tl.abs(value), propagate_nan=tl.PropagateNan.ALL
            )
        absolute = tl.maximum(
            tl.reduce(maximum, axis=1, combine_fn=_maximum_propagate),
            EPS,
            propagate_nan=tl.PropagateNan.ALL,
        )
        scale = tl.div_rn(absolute, 448.0)
        tl.store(
            S + batch * GROUPS * ALIGNED_M + group * ALIGNED_M + tokens,
            scale,
            tokens < M,
        )
        for chunk in range(triton.cdiv(GROUP, 128)):
            in_group = chunk * 128 + lanes
            columns = group * GROUP + in_group
            value = tl.load(
                X
                + batch * STRIDE_B
                + tokens[:, None] * STRIDE_M
                + columns[None, :] * STRIDE_K,
                (tokens[:, None] < M) & (in_group[None, :] < GROUP),
                other=0.0,
            ).to(tl.float32)
            divided = tl.div_rn(value, scale[:, None])
            quantized = tl.minimum(
                tl.maximum(divided, -448.0, propagate_nan=tl.PropagateNan.ALL),
                448.0,
                propagate_nan=tl.PropagateNan.ALL,
            )
            tl.store(
                Q
                + batch * ALIGNED_M * K
                + tokens[:, None] * K
                + columns[None, :],
                _encode(quantized),
                (tokens[:, None] < M) & (in_group[None, :] < GROUP),
            )
        if row_tile == 0 and group == 0:
            tl.store(MASKED + batch, M)


@triton.jit
def _fill_masked(
    MASKED, M: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    if B > 2147483648:
        pid = pid.to(tl.int64)
    indices = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(MASKED + indices, M, indices < B)


def per_token_group_quant_mla_masked_fp8(x, group_size, eps):
    x = x.contiguous()
    b, m, k = x.shape
    aligned_m = (m + 255) // 256 * 256
    groups = k // group_size
    q = x.new_empty((b, aligned_m, k), dtype=torch.float8_e4m3fn)
    scale_stride = aligned_m or 1
    s = x.new_empty_strided(
        (b, aligned_m, groups),
        ((groups or 1) * scale_stride, 1, scale_stride),
        dtype=torch.float32,
    )
    masked = x.new_empty((b,), dtype=torch.int32)
    if b and m and groups:
        output = q.data_ptr()
        scales = s.data_ptr()
        counts = masked.data_ptr()
        assert (output | scales | counts) % 16 == 0
        block = 1 << (group_size - 1).bit_length()
        rows = 8 if block > 256 else min(32, max(1, 4096 // block))
        tiles = b * groups * ((m + rows - 1) // rows)
        strides = x.stride()
        if max(strides) > 2147483647:
            strides = tuple(
                0 if size == 1 else stride
                for size, stride in zip(x.shape, strides)
            )
        wide = b * aligned_m * k > 2147483648
        meta = (m, k, group_size, aligned_m, *strides, rows, block, b, wide)
        if block > 256:
            _address_wide_quantize.run(
                x,
                output,
                scales,
                counts,
                eps,
                meta,
                grid=(min(tiles, 40),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _address_quantize.run(
                x,
                output,
                scales,
                counts,
                eps,
                meta,
                grid=(min(tiles, 40),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    elif b:
        _fill_masked.run(
            masked, m, b, 128, grid=((b + 127) // 128,), warmup=False
        )
    return q, s, masked, m, aligned_m


@triton.jit(do_not_specialize=["Q", "S", "MASKED"])
def _address_quantize(X, Q, S, MASKED, EPS, META: tl.constexpr):
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    counts = tl.multiple_of(
        MASKED.to(tl.uint64).to(tl.pointer_type(tl.int32)), 16
    )
    _quantize(X, output, scales, counts, EPS, META)


@triton.jit(do_not_specialize=["Q", "S", "MASKED"])
def _address_wide_quantize(X, Q, S, MASKED, EPS, META: tl.constexpr):
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    counts = tl.multiple_of(
        MASKED.to(tl.uint64).to(tl.pointer_type(tl.int32)), 16
    )
    _wide_quantize(X, output, scales, counts, EPS, META)


__all__ = ["per_token_group_quant_mla_masked_fp8"]
