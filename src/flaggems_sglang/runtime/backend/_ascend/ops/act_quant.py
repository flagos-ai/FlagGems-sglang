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
def _floor_amax(maximum):
    return tl.where(maximum < 0.0001, 0.0001, maximum)


@triton.jit
def _maximum_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _encode(v, SATURATE: tl.constexpr, KEEP_I16: tl.constexpr = False):
    raw = v.to(tl.int32, bitcast=True)
    if SATURATE:
        mag = raw & 2147483647
    else:
        mag = tl.minimum(tl.abs(v).to(tl.int32, bitcast=True), 1139802112)
    exponent = tl.maximum(
        mag & 2139095040, 1015021568, propagate_nan=tl.PropagateNan.ALL
    )
    magic_bits = exponent + 167772160
    magic = magic_bits.to(tl.float32, bitcast=True)
    absolute = mag.to(tl.float32, bitcast=True)
    if SATURATE:
        rounded = (absolute + magic).to(tl.int32, bitcast=True) & 31
        code = rounded + (exponent >> 20) - 968
        code = tl.minimum(code, 126, propagate_nan=tl.PropagateNan.ALL)
        code = tl.where(mag > 2139095040, 127, code)
        return (code | raw >> 24 & 128).to(tl.uint8)
    else:
        biased_exponent = tl.maximum(
            mag >> 23, 121, propagate_nan=tl.PropagateNan.ALL
        )
        folded_magic_bits = biased_exponent * 8388616 + 167771192
        if KEEP_I16:
            sign = raw >> 24 & 128
            signed_magic_bits = folded_magic_bits + sign
            rounded = (
                (absolute + signed_magic_bits.to(tl.float32, bitcast=True))
                .to(tl.int32, bitcast=True)
                .to(tl.int16)
            )
            return rounded
        else:
            rounded = (
                (absolute + folded_magic_bits.to(tl.float32, bitcast=True))
                .to(tl.int32, bitcast=True)
                .to(tl.int16)
            )
            code = rounded
            sign = (raw >> 24).to(tl.int16) & 128
            return (code | sign).to(tl.uint8)


@triton.jit
def _kernel(X, Q, S, META: tl.constexpr):
    GROUP: tl.constexpr = META[0]
    TILES: tl.constexpr = META[1]
    BLOCK: tl.constexpr = META[2]
    ROWS: tl.constexpr = META[3]
    ROUND: tl.constexpr = META[4]
    start = tl.program_id(0)
    if TILES + 39 >= 2147483648:
        start = start.to(tl.int64)
        stop = tl.full((), TILES, tl.int64)
    else:
        stop = TILES
    for tile in range(start, stop, tl.num_programs(0)):
        if TILES * ROWS * GROUP >= 2147483648:
            groups = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        else:
            groups = tile * ROWS + tl.arange(0, ROWS)
        lanes = tl.arange(0, BLOCK)
        if BLOCK == GROUP:
            value = tl.load(X + groups[:, None] * GROUP + lanes[None, :]).to(
                tl.float32
            )
        else:
            value = tl.load(
                X + groups[:, None] * GROUP + lanes[None, :],
                lanes[None, :] < GROUP,
                0.0,
            ).to(tl.float32)
        amax = _floor_amax(tl.reduce(tl.abs(value), 1, _maximum_nan))
        scale = amax / 448.0
        if ROUND:
            exponent = tl.ceil(tl.log(scale) * 1.4426950408889634).to(tl.int32)
            bits = tl.minimum(tl.maximum(exponent + 127, 1), 254) << 23
            bits = tl.where(amax == float("inf"), 8388608, bits)
            bits = tl.where(amax != amax, 1065353216, bits)
            scale = bits.to(tl.float32, bitcast=True)
        if ROUND:
            quantized = value * tl.div_rn(2.0, scale[:, None]) * 0.5
        else:
            quantized = tl.div_rn(value, scale[:, None])
        if not ROUND and BLOCK == GROUP and (BLOCK >= 2):
            code = _encode(quantized, ROUND, True)
            pairs = tl.arange(0, BLOCK // 2)
            low = al.extract_slice(code, (0, 0), (ROWS, BLOCK // 2), (1, 2))
            high = al.extract_slice(code, (0, 1), (ROWS, BLOCK // 2), (1, 2))
            packed = low.to(tl.uint16) | high.to(tl.uint16) << 8
            tl.store(
                Q + groups[:, None] * (GROUP // 2) + pairs[None, :], packed
            )
        else:
            code = _encode(quantized, ROUND)
            if BLOCK == GROUP:
                tl.store(Q + groups[:, None] * GROUP + lanes[None, :], code)
            else:
                tl.store(
                    Q + groups[:, None] * GROUP + lanes[None, :],
                    code,
                    lanes[None, :] < GROUP,
                )
        tl.store(S + groups, scale)


@triton.jit
def _wide_fused(
    X,
    Q,
    S,
    GROUP: tl.constexpr,
    TILES: tl.constexpr,
    ROWS: tl.constexpr,
    ROUND: tl.constexpr,
):
    lanes = tl.arange(0, 128)
    start = tl.program_id(0)
    if TILES + 39 >= 2147483648:
        start = start.to(tl.int64)
        stop = tl.full((), TILES, tl.int64)
    else:
        stop = TILES
    for tile in range(start, stop, tl.num_programs(0)):
        if TILES * ROWS * GROUP >= 2147483648:
            groups = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        else:
            groups = tile * ROWS + tl.arange(0, ROWS)
        maximum = tl.full((ROWS, 128), 0.0, tl.float32)
        for chunk in range(tl.cdiv(GROUP, 128)):
            columns = chunk * 128 + lanes
            value = tl.load(
                X + groups[:, None] * GROUP + columns[None, :],
                columns[None, :] < GROUP,
                0,
            ).to(tl.float32)
            maximum = tl.maximum(
                maximum, tl.abs(value), propagate_nan=tl.PropagateNan.ALL
            )
        amax = _floor_amax(tl.reduce(maximum, 1, _maximum_nan))
        scale = amax / 448.0
        if ROUND:
            exponent = tl.ceil(tl.log(scale) * 1.4426950408889634).to(tl.int32)
            bits = tl.minimum(tl.maximum(exponent + 127, 1), 254) << 23
            bits = tl.where(amax == float("inf"), 8388608, bits)
            bits = tl.where(amax != amax, 1065353216, bits)
            scale = bits.to(tl.float32, bitcast=True)
        tl.store(S + groups, scale)
        for chunk in range(tl.cdiv(GROUP, 128)):
            columns = chunk * 128 + lanes
            value = tl.load(
                X + groups[:, None] * GROUP + columns[None, :],
                columns[None, :] < GROUP,
                0,
            ).to(tl.float32)
            if ROUND:
                quantized = value * tl.div_rn(2.0, scale[:, None]) * 0.5
            else:
                quantized = tl.div_rn(value, scale[:, None])
            tl.store(
                Q + groups[:, None] * GROUP + columns[None, :],
                _encode(quantized, ROUND),
                columns[None, :] < GROUP,
            )


def act_quant(x, block_size=128, scale_fmt=None):
    if (
        x.ndim == 2
        and (not x.is_contiguous())
        and (x.stride(1) <= 16)
        and (
            (x.shape[0] - 1) * x.stride(0) + (x.shape[1] - 1) * x.stride(1)
            <= 2147483647
        )
    ):
        return _strided_act_quant(x, block_size, scale_fmt)
    x = x.contiguous()
    shape = x.shape
    q = x.new_empty(shape, dtype=torch.float8_e4m3fn)
    s = x.new_empty(
        shape[:-1] + (shape[-1] // block_size,), dtype=torch.float32
    )
    groups = x.numel() // block_size
    if groups:
        output = q.data_ptr()
        scales = s.data_ptr()
        assert (output | scales) % 16 == 0
        if block_size > 256:
            rows = min(8, groups & -groups)
            tiles = groups // rows
            _address_wide_fused.run(
                x,
                output,
                scales,
                block_size,
                tiles,
                rows,
                scale_fmt is not None,
                grid=(min(tiles, 40),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            element_size = x.element_size()
            if block_size <= 128 and element_size <= 2:
                rows = min(32, groups & -groups)
            else:
                block = 1 << (block_size - 1).bit_length()
                tile_elements = min(
                    8192 // element_size, 2048 if block > 128 else 4096
                )
                rows = min(
                    32, groups & -groups, max(1, tile_elements // block)
                )
            tiles = groups // rows
            meta = (block_size, tiles, rows, scale_fmt is not None)
            _address_kernel.run(
                x,
                output,
                scales,
                meta,
                grid=(min(tiles, 40),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (q, s)


@triton.jit
def _strided_load(
    X,
    groups,
    START,
    GROUP: tl.constexpr,
    COLUMNS: tl.constexpr,
    ROWS: tl.constexpr,
    PER_ROW: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
):
    lanes = tl.arange(0, COLUMNS)
    origin = (
        groups // PER_ROW * ROW_STRIDE + groups % PER_ROW * GROUP * COL_STRIDE
    )
    if COL_STRIDE > 1:
        span = tl.arange(
            0, triton.next_power_of_2((COLUMNS - 1) * COL_STRIDE + 1)
        )
        physical = tl.load(
            X + origin[:, None] + START * COL_STRIDE + span[None, :],
            span[None, :] < (GROUP - START - 1) * COL_STRIDE + 1,
            0,
        )
        index = tl.broadcast_to((lanes * COL_STRIDE)[None, :], (ROWS, COLUMNS))
        value = tl.gather(physical, index, axis=1).to(tl.float32)
    else:
        value = tl.load(
            X + origin[:, None] + (START + lanes[None, :]) * COL_STRIDE,
            START + lanes[None, :] < GROUP,
            0,
        ).to(tl.float32)
    return tl.where(START + lanes[None, :] < GROUP, value, 0.0)


@triton.jit
def _strided_kernel(
    X,
    Q,
    S,
    META: tl.constexpr,
    PER_ROW: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
):
    GROUP: tl.constexpr = META[0]
    TILES: tl.constexpr = META[1]
    BLOCK: tl.constexpr = META[2]
    ROWS: tl.constexpr = META[3]
    ROUND: tl.constexpr = META[4]
    start = tl.program_id(0)
    if TILES + 39 >= 2147483648:
        start = start.to(tl.int64)
        stop = tl.full((), TILES, tl.int64)
    else:
        stop = TILES
    for tile in range(start, stop, tl.num_programs(0)):
        if TILES * ROWS * GROUP >= 2147483648:
            groups = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        else:
            groups = tile * ROWS + tl.arange(0, ROWS)
        lanes = tl.arange(0, BLOCK)
        value = _strided_load(
            X, groups, 0, GROUP, BLOCK, ROWS, PER_ROW, ROW_STRIDE, COL_STRIDE
        )
        amax = _floor_amax(tl.reduce(tl.abs(value), 1, _maximum_nan))
        scale = amax / 448.0
        if ROUND:
            exponent = tl.ceil(tl.log(scale) * 1.4426950408889634).to(tl.int32)
            bits = tl.minimum(tl.maximum(exponent + 127, 1), 254) << 23
            bits = tl.where(amax == float("inf"), 8388608, bits)
            bits = tl.where(amax != amax, 1065353216, bits)
            scale = bits.to(tl.float32, bitcast=True)
        if ROUND:
            quantized = value * tl.div_rn(2.0, scale[:, None]) * 0.5
        else:
            quantized = tl.div_rn(value, scale[:, None])
        code = _encode(quantized, ROUND)
        if BLOCK == GROUP:
            tl.store(Q + groups[:, None] * GROUP + lanes[None, :], code)
        else:
            tl.store(
                Q + groups[:, None] * GROUP + lanes[None, :],
                code,
                lanes[None, :] < GROUP,
            )
        tl.store(S + groups, scale)


@triton.jit
def _strided_wide_fused(
    X,
    Q,
    S,
    GROUP: tl.constexpr,
    TILES: tl.constexpr,
    ROWS: tl.constexpr,
    ROUND: tl.constexpr,
    PER_ROW: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
):
    lanes = tl.arange(0, 128)
    start = tl.program_id(0)
    if TILES + 39 >= 2147483648:
        start = start.to(tl.int64)
        stop = tl.full((), TILES, tl.int64)
    else:
        stop = TILES
    for tile in range(start, stop, tl.num_programs(0)):
        if TILES * ROWS * GROUP >= 2147483648:
            groups = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        else:
            groups = tile * ROWS + tl.arange(0, ROWS)
        maximum = tl.full((ROWS, 128), 0.0, tl.float32)
        for chunk in range(tl.cdiv(GROUP, 128)):
            columns = chunk * 128 + lanes
            value = _strided_load(
                X,
                groups,
                chunk * 128,
                GROUP,
                128,
                ROWS,
                PER_ROW,
                ROW_STRIDE,
                COL_STRIDE,
            )
            maximum = tl.maximum(
                maximum, tl.abs(value), propagate_nan=tl.PropagateNan.ALL
            )
        amax = _floor_amax(tl.reduce(maximum, 1, _maximum_nan))
        scale = amax / 448.0
        if ROUND:
            exponent = tl.ceil(tl.log(scale) * 1.4426950408889634).to(tl.int32)
            bits = tl.minimum(tl.maximum(exponent + 127, 1), 254) << 23
            bits = tl.where(amax == float("inf"), 8388608, bits)
            bits = tl.where(amax != amax, 1065353216, bits)
            scale = bits.to(tl.float32, bitcast=True)
        tl.store(S + groups, scale)
        for chunk in range(tl.cdiv(GROUP, 128)):
            columns = chunk * 128 + lanes
            value = _strided_load(
                X,
                groups,
                chunk * 128,
                GROUP,
                128,
                ROWS,
                PER_ROW,
                ROW_STRIDE,
                COL_STRIDE,
            )
            if ROUND:
                quantized = value * tl.div_rn(2.0, scale[:, None]) * 0.5
            else:
                quantized = tl.div_rn(value, scale[:, None])
            tl.store(
                Q + groups[:, None] * GROUP + columns[None, :],
                _encode(quantized, ROUND),
                columns[None, :] < GROUP,
            )


def _strided_act_quant(x, block_size, scale_fmt):
    shape = x.shape
    q = x.new_empty(shape, dtype=torch.float8_e4m3fn)
    s = x.new_empty(
        shape[:-1] + (shape[-1] // block_size,), dtype=torch.float32
    )
    groups = x.numel() // block_size
    if groups:
        output = q.data_ptr()
        scales = s.data_ptr()
        assert (output | scales) % 16 == 0
        block = 1 << (block_size - 1).bit_length()
        per_row = shape[1] // block_size
        row_stride, col_stride = x.stride()
        columns = min(block, 128) if block > 256 else block
        span = 1 << ((columns - 1) * col_stride + 1 - 1).bit_length()
        tile_elements = min(
            8192 // x.element_size(), 2048 if 128 < block <= 256 else 4096
        )
        row_budget = max(1, tile_elements // max(columns, span))
        rows = min(
            8 if block > 256 else 32,
            groups & -groups,
            1 << row_budget.bit_length() - 1,
        )
        tiles = groups // rows
        if block > 256:
            _address_strided_wide_fused.run(
                x,
                output,
                scales,
                block_size,
                tiles,
                rows,
                scale_fmt is not None,
                per_row,
                row_stride,
                col_stride,
                grid=(min(tiles, 40),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            meta = (block_size, tiles, block, rows, scale_fmt is not None)
            _address_strided_kernel.run(
                x,
                output,
                scales,
                meta,
                per_row,
                row_stride,
                col_stride,
                grid=(min(tiles, 40),),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (q, s)


@triton.jit(do_not_specialize=["Q", "S"])
def _address_kernel(X, Q, S, META: tl.constexpr):
    FULL_META: tl.constexpr = (
        META[0],
        META[1],
        triton.next_power_of_2(META[0]),
        META[2],
        META[3],
    )
    if not META[3] and META[0] >= 2 and (META[0] & META[0] - 1 == 0):
        output = tl.multiple_of(
            Q.to(tl.uint64).to(tl.pointer_type(tl.uint16)), 16
        )
    else:
        output = tl.multiple_of(
            Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16
        )
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _kernel(X, output, scales, FULL_META)


@triton.jit(do_not_specialize=["Q", "S"])
def _address_wide_fused(
    X,
    Q,
    S,
    GROUP: tl.constexpr,
    TILES: tl.constexpr,
    ROWS: tl.constexpr,
    ROUND: tl.constexpr,
):
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _wide_fused(X, output, scales, GROUP, TILES, ROWS, ROUND)


@triton.jit(do_not_specialize=["Q", "S"])
def _address_strided_kernel(
    X,
    Q,
    S,
    META: tl.constexpr,
    PER_ROW: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
):
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _strided_kernel(X, output, scales, META, PER_ROW, ROW_STRIDE, COL_STRIDE)


@triton.jit(do_not_specialize=["Q", "S"])
def _address_strided_wide_fused(
    X,
    Q,
    S,
    GROUP: tl.constexpr,
    TILES: tl.constexpr,
    ROWS: tl.constexpr,
    ROUND: tl.constexpr,
    PER_ROW: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
):
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _strided_wide_fused(
        X,
        output,
        scales,
        GROUP,
        TILES,
        ROWS,
        ROUND,
        PER_ROW,
        ROW_STRIDE,
        COL_STRIDE,
    )


__all__ = ["act_quant"]
