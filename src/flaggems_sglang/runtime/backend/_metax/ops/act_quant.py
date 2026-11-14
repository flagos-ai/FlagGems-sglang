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
from triton.language.extra import libdevice


@triton.jit
def _maximum_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _divide_bf16(
    value, scale, BF16: tl.constexpr, SHARED_NEGATE: tl.constexpr = False
):
    if BF16:
        reciprocal = 1.0 / scale
        coarse = value * reciprocal
        safe_scale = tl.where(scale < float("inf"), scale, 1.0)
        if SHARED_NEGATE:
            residual = tl.fma(coarse, -safe_scale, value)
        else:
            residual = tl.fma(-coarse, safe_scale, value)
        result = tl.fma(residual, reciprocal, coarse)
        return result
    return value / scale


@triton.jit
def _encode(v, SATURATE: tl.constexpr, original):
    if SATURATE:
        raw = v.to(tl.uint32, bitcast=True)
        sign = original.to(tl.uint32, bitcast=True) >> 24 & 128
        mag = raw & 2147483647
        normal = mag + 3288858623 + (mag >> 20 & 1) >> 20
        if SATURATE:
            normal = tl.minimum(normal, 126, propagate_nan=tl.PropagateNan.ALL)
        sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) & 15
        code = tl.where(tl.abs(v) >= 0.015625, normal, sub)
        code = tl.where(v != v, 127, code)
        return (code | sign).to(tl.uint8)
    else:
        raw = v.to(tl.uint32, bitcast=True)
        sign = original.to(tl.uint32, bitcast=True) >> 24 & 128
        mag = tl.minimum(raw & 2147483647, 1139802112)
        exponent = tl.maximum(mag >> 23, 121)
        magic_bits = exponent * 8388616 + 167771192
        absolute = mag.to(tl.float32, bitcast=True)
        rounded = (absolute + magic_bits.to(tl.float32, bitcast=True)).to(
            tl.uint32, bitcast=True
        )
        return (rounded | sign).to(tl.uint8)


@triton.jit
def _load_strided(
    X,
    groups,
    lanes,
    GROUP: tl.constexpr,
    PER_ROW: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COLUMN_STRIDE: tl.constexpr,
    PAD: tl.constexpr,
):
    rows = groups.to(tl.int64) // PER_ROW
    if PAD:
        safe_lanes = tl.minimum(lanes, GROUP - 1)
    else:
        safe_lanes = lanes
    columns = (groups.to(tl.int64) % PER_ROW)[None, :] * GROUP + safe_lanes[
        :, None
    ]
    offsets = rows[None, :] * ROW_STRIDE + columns * COLUMN_STRIDE
    value = tl.trans(tl.load(X + offsets).to(tl.float32))
    if PAD:
        return tl.where(lanes[None, :] < GROUP, value, 0.0)
    else:
        return value


@triton.jit
def _tile(X, Q, S, META: tl.constexpr, tile):
    GROUP: tl.constexpr = META[0]
    TILES: tl.constexpr = META[1]
    BLOCK: tl.constexpr = META[2]
    ROWS: tl.constexpr = META[3]
    ROUND: tl.constexpr = META[4]
    Q = Q.to(tl.pointer_type(tl.uint8))
    if (
        X.dtype.element_ty == tl.bfloat16
        and (not ROUND)
        and (BLOCK == GROUP)
        and (GROUP >= 2)
        and (TILES * ROWS * GROUP >= 8388608)
    ):
        WIDTH: tl.constexpr = GROUP // 2
        if TILES * ROWS * GROUP >= 2147483648:
            groups = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        else:
            groups = tile * ROWS + tl.arange(0, ROWS)
        lanes = tl.arange(0, WIDTH)
        offsets = groups[:, None] * GROUP + lanes[None, :]
        if len(META) > 5:
            first = _load_strided(
                X, groups, lanes, GROUP, META[5], META[6], META[7], False
            )
        else:
            first = tl.load(X + offsets).to(tl.float32)
        if len(META) > 5:
            second = _load_strided(
                X,
                groups,
                lanes + WIDTH,
                GROUP,
                META[5],
                META[6],
                META[7],
                False,
            )
        else:
            second = tl.load(X + offsets + WIDTH).to(tl.float32)
        first_bits = tl.abs(first).to(tl.uint32, bitcast=True)
        second_bits = tl.abs(second).to(tl.uint32, bitcast=True)
        maximum_bits = tl.maximum(
            tl.max(tl.maximum(first_bits, second_bits), axis=1), 953267991
        )
        amax = maximum_bits.to(tl.float32, bitcast=True)
        scale = amax * 0.0022321429569274187
        q0 = _divide_bf16(first, scale[:, None], True, True)
        q1 = _divide_bf16(second, scale[:, None], True, True)
        tl.store(Q + offsets, _encode(q0, False, first))
        tl.store(Q + offsets + WIDTH, _encode(q1, False, second))
        tl.store(S + groups, scale)
    else:
        if TILES * ROWS * GROUP >= 2147483648:
            groups = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
        else:
            groups = tile * ROWS + tl.arange(0, ROWS)
        lanes = tl.arange(0, BLOCK)
        if len(META) > 5:
            value = _load_strided(
                X,
                groups,
                lanes,
                GROUP,
                META[5],
                META[6],
                META[7],
                BLOCK != GROUP,
            )
        elif BLOCK == GROUP:
            value = tl.load(X + groups[:, None] * GROUP + lanes[None, :]).to(
                tl.float32
            )
        else:
            value = tl.load(
                X
                + groups[:, None] * GROUP
                + tl.minimum(
                    lanes[None, :],
                    GROUP - 1,
                    propagate_nan=tl.PropagateNan.ALL,
                )
            ).to(tl.float32)
            value = tl.where(lanes[None, :] < GROUP, value, 0.0)
        magnitude = tl.abs(value).to(tl.uint32, bitcast=True)
        maximum_bits = tl.maximum(tl.max(magnitude, axis=1), 953267991)
        amax = maximum_bits.to(tl.float32, bitcast=True)
        scale = amax * 0.0022321429569274187
        if ROUND:
            if (
                X.dtype.element_ty == tl.bfloat16
                or X.dtype.element_ty == tl.float16
            ):
                bits = scale.to(tl.uint32, bitcast=True)
                bits = (
                    tl.minimum(tl.maximum(bits + 8388607 >> 23, 1), 254) << 23
                ).to(tl.int32)
            else:
                exponent = tl.ceil(libdevice.log2(scale)).to(tl.int32)
                bits = tl.minimum(tl.maximum(exponent + 127, 1), 254) << 23
            bits = tl.where(amax == float("inf"), 8388608, bits)
            bits = tl.where(amax != amax, 1065353216, bits)
            scale = bits.to(tl.float32, bitcast=True)
        if ROUND:
            reciprocal = tl.maximum(
                2130706432 - bits, 4194304, propagate_nan=tl.PropagateNan.ALL
            ).to(tl.float32, bitcast=True)
            quantized = value * reciprocal[:, None]
        else:
            quantized = _divide_bf16(
                value, scale[:, None], X.dtype.element_ty == tl.bfloat16
            )
        code = _encode(
            quantized,
            ROUND,
            (
                value
                if X.dtype.element_ty == tl.bfloat16 and (not ROUND)
                else quantized
            ),
        )
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
def _kernel(X, Q, S, META: tl.constexpr):
    if META[1] > 2147483647:
        for tile in range(
            tl.program_id(0).to(tl.int64),
            tl.full((), META[1], tl.int64),
            tl.num_programs(0),
        ):
            _tile(X, Q, S, META, tile)
    else:
        _tile(X, Q, S, META, tl.program_id(0))


def act_quant(x, block_size=128, scale_fmt=None):
    shape = x.shape
    strides = ()
    if not x.is_contiguous():
        if len(shape) == 2 and x.stride(1) * x.element_size() <= 64:
            strides = (shape[-1] // block_size,) + x.stride()
        else:
            x = x.contiguous()
    q = x.new_empty(shape, dtype=torch.float8_e4m3fn)
    s = x.new_empty(
        shape[:-1] + (shape[-1] // block_size,), dtype=torch.float32
    )
    groups = x.numel() // block_size
    if groups:
        block = 1 << (block_size - 1).bit_length()
        rows = min(
            32,
            1 << max(0, (groups // 128).bit_length() - 1),
            groups & -groups,
            max(1, 4096 // block),
        )
        warps = (
            1
            if rows * block <= 512
            else (
                2
                if groups * block_size >= 8388608
                and x.dtype == torch.bfloat16
                and (scale_fmt is None)
                and (block == block_size)
                and (block_size >= 2)
                else 4
            )
        )
        tiles = groups // rows
        meta = (block_size, tiles, block, rows, scale_fmt is not None)
        if strides:
            meta += strides
        _kernel.run(
            x,
            q,
            s,
            meta,
            grid=(65535 if tiles > 2147483647 else tiles,),
            warmup=False,
            num_warps=warps,
            enable_fp_fusion=False,
        )
    return (q, s)


__all__ = ["act_quant"]
