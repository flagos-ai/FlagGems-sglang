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
def _quantize_exact(value, scale, ROUND: tl.constexpr):
    result = tl.div_rn(value, scale)
    if ROUND:
        result = tl.minimum(
            tl.maximum(result, -448.0, propagate_nan=tl.PropagateNan.ALL),
            448.0,
            propagate_nan=tl.PropagateNan.ALL,
        )
    return result


@triton.jit
def _encode(v):
    raw = v.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    mag = raw & 2147483647
    normal = tl.minimum((mag + 524287 + (mag >> 20 & 1) >> 20) - 960, 127)
    sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) - 1182793728
    code = tl.where(mag >= 1015021568, normal, sub)
    return (code | sign).to(tl.uint8)


@triton.jit
def _tile(X, Q, S, META: tl.constexpr, tile):
    GROUP: tl.constexpr = META[0]
    TILES: tl.constexpr = META[1]
    BLOCK: tl.constexpr = META[2]
    ROWS: tl.constexpr = META[3]
    ROUND: tl.constexpr = META[4]
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
            X
            + groups[:, None] * GROUP
            + tl.minimum(
                lanes[None, :], GROUP - 1, propagate_nan=tl.PropagateNan.ALL
            )
        ).to(tl.float32)
        value = tl.where(lanes[None, :] < GROUP, value, 0.0)
    magnitude = tl.abs(value).to(tl.uint32, bitcast=True)
    maximum_bits = tl.maximum(tl.max(magnitude, axis=1), 953267991)
    amax = maximum_bits.to(tl.float32, bitcast=True)
    scale = amax * 0.0022321429569274187
    if ROUND:
        exponent = tl.ceil(libdevice.log2(scale)).to(tl.int32)
        bits = tl.minimum(tl.maximum(exponent + 127, 1), 254) << 23
        scale = bits.to(tl.float32, bitcast=True)
    quantized = _quantize_exact(value, scale[:, None], ROUND)
    code = _encode(quantized)
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
    if META[1] > 16777215:
        tile = tl.program_id(0).to(tl.int64)
        limit = tl.full((), META[1], tl.int64)
        step = tl.num_programs(0).to(tl.int64)
        while tile < limit:
            _tile(X, Q, S, META, tile)
            tile += tl.minimum(step, limit - tile)
    else:
        _tile(X, Q, S, META, tl.program_id(0))


@triton.jit
def _split_tile(X, Q, S, META: tl.constexpr, tile):
    GROUP: tl.constexpr = META[0]
    ROWS: tl.constexpr = META[1]
    ROUND: tl.constexpr = META[2]
    SPLIT: tl.constexpr = META[3]
    WIDTH: tl.constexpr = GROUP // SPLIT
    groups = tile * ROWS + tl.arange(0, ROWS)
    lanes = tl.arange(0, WIDTH)
    off = groups[:, None] * GROUP + lanes[None, :]
    v0 = tl.load(X + off).to(tl.float32)
    v1 = tl.load(X + off + WIDTH).to(tl.float32)
    ab = tl.maximum(tl.abs(v0), tl.abs(v1), propagate_nan=tl.PropagateNan.ALL)
    if SPLIT == 4:
        v2 = tl.load(X + off + WIDTH * 2).to(tl.float32)
        v3 = tl.load(X + off + WIDTH * 3).to(tl.float32)
        ab = tl.maximum(
            ab,
            tl.maximum(
                tl.abs(v2), tl.abs(v3), propagate_nan=tl.PropagateNan.ALL
            ),
            propagate_nan=tl.PropagateNan.ALL,
        )
    magnitude = ab.to(tl.uint32, bitcast=True)
    maximum_bits = tl.maximum(tl.max(magnitude, axis=1), 953267991)
    amax = maximum_bits.to(tl.float32, bitcast=True)
    scale = amax * 0.0022321429569274187
    if ROUND:
        exponent = tl.ceil(libdevice.log2(scale)).to(tl.int32)
        bits = tl.minimum(tl.maximum(exponent + 127, 1), 254) << 23
        scale = bits.to(tl.float32, bitcast=True)
    q0 = _encode(_quantize_exact(v0, scale[:, None], ROUND))
    q1 = _encode(_quantize_exact(v1, scale[:, None], ROUND))
    tl.store(Q + off, q0)
    tl.store(Q + off + WIDTH, q1)
    if SPLIT == 4:
        q2 = _encode(_quantize_exact(v2, scale[:, None], ROUND))
        q3 = _encode(_quantize_exact(v3, scale[:, None], ROUND))
        tl.store(Q + off + WIDTH * 2, q2)
        tl.store(Q + off + WIDTH * 3, q3)
    if GROUP <= 256 and ROWS >= 4 and (WIDTH >= 4) and (ROWS * WIDTH <= 1024):
        SCALE_SPT: tl.constexpr = (
            1 if ROWS * WIDTH < 256 else ROWS * WIDTH // 256
        )
        SCALE_COLS: tl.constexpr = WIDTH // SCALE_SPT
        SCALE_LANES: tl.constexpr = 64 if SCALE_COLS > 64 else SCALE_COLS
        if ROWS > 64 // SCALE_LANES:
            if SCALE_SPT == 2:
                scale_halves = (
                    scale.to(tl.uint32, bitcast=True)[:, None]
                    >> (lanes[None, :] & 1) * 16
                ).to(tl.uint16)
                scale_ptrs = (
                    S.to(tl.pointer_type(tl.uint16))
                    + groups[:, None] * 2
                    + lanes[None, :]
                )
                tl.store(scale_ptrs, scale_halves, lanes[None, :] < 2)
            else:
                scale_bytes = (
                    scale.to(tl.uint32, bitcast=True)[:, None]
                    >> (lanes[None, :] & 3) * 8
                ).to(tl.uint8)
                scale_ptrs = (
                    S.to(tl.pointer_type(tl.uint8))
                    + groups[:, None] * 4
                    + lanes[None, :]
                )
                tl.store(scale_ptrs, scale_bytes, lanes[None, :] < 4)
        else:
            tl.store(S + groups, scale)
    else:
        tl.store(S + groups, scale)


@triton.jit
def _split_kernel(X, Q, S, META: tl.constexpr):
    if len(META) > 4:
        tile = tl.program_id(0).to(tl.int64)
        limit = tl.full((), META[4], tl.int64)
        step = tl.num_programs(0).to(tl.int64)
        while tile < limit:
            _split_tile(X, Q, S, META, tile)
            tile += tl.minimum(step, limit - tile)
    else:
        _split_tile(X, Q, S, META, tl.program_id(0))


@triton.jit(do_not_specialize=["Q"])
def _address_kernel(X, Q, S, META: tl.constexpr):
    output = tl.multiple_of(Q.to(tl.pointer_type(tl.uint8)), 16)
    _kernel(X, output, S, META)


@triton.jit(do_not_specialize=["Q"])
def _address_split_kernel(X, Q, S, META: tl.constexpr):
    output = tl.multiple_of(Q.to(tl.pointer_type(tl.uint8)), 16)
    _split_kernel(X, output, S, META)


def act_quant(x, block_size=128, scale_fmt=None):
    x = x.contiguous()
    shape = x.shape
    numel = x.numel()
    q = x.new_empty(shape, dtype=torch.float8_e4m3fn)
    s = x.new_empty(
        shape[:-1] + (shape[-1] // block_size,), dtype=torch.float32
    )
    groups = numel // block_size
    if groups:
        output = q.data_ptr()
        assert output % 16 == 0
        split = (
            block_size >= 2
            and block_size & block_size - 1 == 0
            and (numel < 2147483648)
        )
        tile_rows = (
            8
            if split and 128 <= block_size <= 256
            else (
                16
                if numel * (x.element_size() + 1) + groups * 4 > 16777216
                else 8
            )
        )
        rows = min(tile_rows, groups & -groups)
        tiles = groups // rows
        if split:
            _address_split_kernel.run(
                x,
                output,
                s,
                (
                    (block_size, rows, scale_fmt is not None, 2, tiles)
                    if tiles > 16777215
                    else (block_size, rows, scale_fmt is not None, 2)
                ),
                grid=(65535 if tiles > 16777215 else tiles,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _address_kernel.run(
                x,
                output,
                s,
                (
                    block_size,
                    tiles,
                    triton.next_power_of_2(block_size),
                    rows,
                    scale_fmt is not None,
                ),
                grid=(65535 if tiles > 16777215 else tiles,),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (q, s)


__all__ = ["act_quant"]
