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
def _encode_bounded(v, original):
    raw = v.to(tl.uint32, bitcast=True)
    sign = original.to(tl.uint32, bitcast=True) >> 24 & 128
    mag = raw & 2147483647
    normal = tl.minimum(((mag + 524287 + (mag >> 20 & 1)) >> 20) - 960, 127)
    sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) - 1182793728
    code = tl.where(mag >= 1015021568, normal, sub)
    return (code | sign).to(tl.uint8)


@triton.jit
def _maximum_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _divide_bf16(value, scale, BF16: tl.constexpr):
    if BF16:
        reciprocal = tl.div_rn(1.0, scale)
        coarse = value * reciprocal
        safe_scale = tl.where(scale < float("inf"), scale, 1.0)
        residual = tl.fma(-coarse, safe_scale, value)
        result = tl.fma(residual, reciprocal, coarse)
        return result
    return tl.div_rn(value, scale)


@triton.jit
def _encode(v):
    raw = v.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    mag = raw & 2147483647
    normal = tl.minimum(
        (mag + 524287 + (mag >> 20 & 1) >> 20) - 960,
        126,
        propagate_nan=tl.PropagateNan.ALL,
    )
    sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) - 1182793728
    code = tl.where(tl.abs(v) >= 0.015625, normal, sub)
    code = tl.where(v != v, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _tile(X, Q, S, META: tl.constexpr, tile):
    Q = Q.to(tl.pointer_type(tl.uint8))
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
        bits = tl.where(amax == float("inf"), 8388608, bits)
        bits = tl.where(amax != amax, 1065353216, bits)
        scale = bits.to(tl.float32, bitcast=True)
    quantized = _divide_bf16(
        value,
        scale[:, None],
        X.dtype.element_ty == tl.bfloat16 and (not ROUND),
    )
    if X.dtype.element_ty == tl.bfloat16 and not ROUND:
        code = _encode_bounded(quantized, value)
    else:
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
    if META[1] > 2147483647:
        for tile in range(
            tl.program_id(0).to(tl.int64),
            tl.full((), META[1], tl.int64),
            tl.num_programs(0),
        ):
            _tile(X, Q, S, META, tile)
    else:
        _tile(X, Q, S, META, tl.program_id(0))


@triton.jit(do_not_specialize=["X", "Q", "S"])
def _all_address_kernel(X, Q, S, META: tl.constexpr):
    INPUT_TYPE: tl.constexpr = META[0]
    INPUT_ALIGNMENT: tl.constexpr = META[1]
    GROUP: tl.constexpr = META[2][0]
    KERNEL_META: tl.constexpr = (
        GROUP,
        META[2][1],
        triton.next_power_of_2(GROUP),
        META[2][2],
        META[2][3],
    )
    values = tl.multiple_of(
        X.to(tl.uint64).to(tl.pointer_type(INPUT_TYPE)), INPUT_ALIGNMENT
    )
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _kernel(values, output, scales, KERNEL_META)


def act_quant(x, block_size=128, scale_fmt=None):
    x = x.contiguous()
    shape = x.shape
    q = x.new_empty(shape, dtype=torch.float8_e4m3fn)
    s = x.new_empty(
        shape[:-1] + (shape[-1] // block_size,), dtype=torch.float32
    )
    groups = x.numel() // block_size
    if groups:
        values = x.data_ptr()
        output = q.data_ptr()
        scales = s.data_ptr()
        assert (output | scales) % 16 == 0
        input_type = (
            tl.bfloat16
            if x.dtype == torch.bfloat16
            else tl.float16 if x.dtype == torch.float16 else tl.float32
        )
        input_alignment = min(16, values & -values)
        rows = min(8, groups & -groups)
        if block_size > 128:
            rows = min(
                rows, max(1, 1024 // triton.next_power_of_2(block_size))
            )
        tiles = groups // rows
        meta = (block_size, tiles, rows, scale_fmt is not None)
        _all_address_kernel.run(
            values,
            output,
            scales,
            (input_type, input_alignment, meta),
            grid=(65535 if tiles > 2147483647 else tiles,),
            warmup=False,
            num_warps=1,
            enable_fp_fusion=False,
        )
    return (q, s)


__all__ = ["act_quant"]
