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
def _load_strided(
    X,
    groups,
    lanes,
    GROUP: tl.constexpr,
    PER_ROW: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COLUMN_STRIDE: tl.constexpr,
    PACKED: tl.constexpr,
):
    rows = groups.to(tl.int64) // PER_ROW
    columns = (groups.to(tl.int64) % PER_ROW)[:, None] * GROUP + tl.minimum(
        lanes[None, :], GROUP - 1
    )
    offsets = rows[:, None] * ROW_STRIDE + columns * COLUMN_STRIDE
    value = tl.load(X + offsets, cache_modifier=".cg" if PACKED else "").to(
        tl.float32
    )
    return tl.where(lanes[None, :] < GROUP, value, 0.0)


@triton.jit
def _divide_hip_bf16(value, scale, reciprocal):
    coarse = value * reciprocal
    safe_scale = tl.where(scale < float("inf"), scale, 1.0)
    residual = tl.fma(-coarse, safe_scale, value)
    result = tl.fma(residual, reciprocal, coarse)
    return tl.where((value == 0.0) & (scale == scale), value, result)


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
    normal = (mag + 524287 + (mag >> 20 & 1) >> 20) - 960
    sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) - 1182793728
    code = tl.where(tl.abs(v) >= 0.015625, normal, sub)
    code = tl.where(v != v, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _encode_bf16(v, original):
    sign = original.to(tl.uint32, bitcast=True) >> 24 & 128
    absolute = tl.minimum(tl.abs(v), 480.0, propagate_nan=tl.PropagateNan.NONE)
    mag = absolute.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(mag & 2139095040, 1015021568)
    exponent_sum = tl.inline_asm_elementwise(
        "{ .reg .b32 shifted; shr.u32 shifted, $1, 20; add.u32 $0, $1, shifted; }",
        constraints="=r,r",
        args=[exponent],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    magic_bits = (exponent_sum + 167771192) | sign
    rounded = (absolute + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    return rounded.to(tl.uint8)


@triton.jit
def _encode_fold(v):
    raw = v.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    mag = raw & 2147483647
    normal = mag + 3288858623 + (mag >> 20 & 1) >> 20
    sub = (tl.abs(v) + 16384.0).to(tl.uint32, bitcast=True) & 15
    code = tl.where(tl.abs(v) >= 0.015625, normal, sub)
    code = tl.where(v != v, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _tile(X, Q, S, META: tl.constexpr, tile):
    GROUP: tl.constexpr = META[0]
    TILES: tl.constexpr = META[1]
    BLOCK: tl.constexpr = META[2]
    ROWS: tl.constexpr = META[3]
    ROUND: tl.constexpr = META[4]
    PACKED: tl.constexpr = META[5]
    FOLDED: tl.constexpr = META[6]
    if TILES * ROWS * GROUP >= 2147483648:
        groups = tile.to(tl.int64) * ROWS + tl.arange(0, ROWS)
    else:
        groups = tile * ROWS + tl.arange(0, ROWS)
    lanes = tl.arange(0, BLOCK)
    if len(META) > 7:
        PER_ROW: tl.constexpr = META[7]
        ROW_STRIDE: tl.constexpr = META[8]
        COLUMN_STRIDE: tl.constexpr = META[9]
        value = _load_strided(
            X, groups, lanes, GROUP, PER_ROW, ROW_STRIDE, COLUMN_STRIDE, PACKED
        )
    elif BLOCK == GROUP:
        value = tl.load(
            X + groups[:, None] * GROUP + lanes[None, :],
            cache_modifier=".cg" if PACKED else "",
        ).to(tl.float32)
    else:
        value = tl.load(
            X
            + groups[:, None] * GROUP
            + tl.minimum(
                lanes[None, :], GROUP - 1, propagate_nan=tl.PropagateNan.ALL
            ),
            cache_modifier=".cg" if PACKED else "",
        ).to(tl.float32)
        value = tl.where(lanes[None, :] < GROUP, value, 0.0)
    if FOLDED:
        magnitude = tl.abs(value).to(tl.uint32, bitcast=True)
        maximum_bits = tl.maximum(tl.max(magnitude, axis=1), 953267991)
        amax = maximum_bits.to(tl.float32, bitcast=True)
    else:
        amax = tl.maximum(
            tl.reduce(tl.abs(value), 1, _maximum_nan),
            0.0001,
            propagate_nan=tl.PropagateNan.ALL,
        )
    scale = amax * 0.0022321429569274187
    if ROUND:
        if not FOLDED and (
            X.dtype.element_ty == tl.bfloat16
            or X.dtype.element_ty == tl.float16
        ):
            bits = scale.to(tl.uint32, bitcast=True)
            bits = tl.minimum(tl.maximum(bits + 8388607 >> 23, 1), 254) << 23
        else:
            exponent = tl.ceil(libdevice.log2(scale)).to(tl.int32)
            bits = tl.minimum(tl.maximum(exponent + 127, 1), 254) << 23
        bits = tl.where(amax == float("inf"), 8388608, bits)
        bits = tl.where(amax != amax, 1065353216, bits)
        scale = bits.to(tl.float32, bitcast=True)
    if ROUND:
        quantized = tl.minimum(
            tl.maximum(
                value * tl.div_rn(1.0, scale[:, None]),
                -448.0,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            448.0,
            propagate_nan=tl.PropagateNan.ALL,
        )
    elif FOLDED and X.dtype.element_ty == tl.bfloat16:
        reciprocal = libdevice.fast_dividef(1.0, scale)
        quantized = tl.minimum(
            tl.maximum(
                _divide_hip_bf16(value, scale[:, None], reciprocal[:, None]),
                -448.0,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            448.0,
            propagate_nan=tl.PropagateNan.ALL,
        )
    elif X.dtype.element_ty == tl.bfloat16 and (not FOLDED):
        quantized = _divide_bf16(value, scale[:, None], True)
    else:
        quantized = tl.minimum(
            tl.maximum(
                _divide_bf16(value, scale[:, None], False),
                -448.0,
                propagate_nan=tl.PropagateNan.ALL,
            ),
            448.0,
            propagate_nan=tl.PropagateNan.ALL,
        )
    if FOLDED:
        code = _encode_fold(quantized)
    elif X.dtype.element_ty == tl.bfloat16 and (not ROUND):
        code = _encode_bf16(quantized, value)
    else:
        code = _encode(quantized)
    if PACKED:
        shifts = tl.arange(0, 4) * 8
        packed = tl.sum(
            code.to(tl.uint32).reshape((ROWS, BLOCK // 4, 4))
            << shifts[None, None, :],
            axis=2,
        )
        words = tl.arange(0, BLOCK // 4)
        if BLOCK == GROUP:
            tl.store(
                Q.to(tl.pointer_type(tl.uint32))
                + groups[:, None] * (GROUP // 4)
                + words[None, :],
                packed,
                cache_modifier=".cs",
            )
        else:
            tl.store(
                Q.to(tl.pointer_type(tl.uint32))
                + groups[:, None] * (GROUP // 4)
                + words[None, :],
                packed,
                words[None, :] < GROUP // 4,
                cache_modifier=".cs",
            )
    elif BLOCK == GROUP:
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


@triton.jit
def _split_divide_bf16(value, scale, BF16: tl.constexpr):
    if BF16:
        reciprocal = libdevice.rcp_rn(scale)
        coarse = value * reciprocal
        safe_scale = tl.where(scale < float("inf"), scale, 1.0)
        residual = tl.fma(-coarse, safe_scale, value)
        result = tl.fma(residual, reciprocal, coarse)
        return result
    return tl.div_rn(value, scale)


@triton.jit
def _split_kernel(X, Q, S, META: tl.constexpr):
    GROUP: tl.constexpr = META[0]
    ROWS: tl.constexpr = META[1]
    ROUND: tl.constexpr = META[2]
    SPLIT: tl.constexpr = META[3]
    if len(META) > 4:
        PER_ROW: tl.constexpr = META[4]
        ROW_STRIDE: tl.constexpr = META[5]
        COLUMN_STRIDE: tl.constexpr = META[6]
    WIDTH: tl.constexpr = GROUP // SPLIT
    groups = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    lanes = tl.arange(0, WIDTH)
    off = groups[:, None] * GROUP + lanes[None, :]
    if len(META) > 4:
        v0 = _load_strided(
            X,
            groups,
            lanes + WIDTH * 0,
            GROUP,
            PER_ROW,
            ROW_STRIDE,
            COLUMN_STRIDE,
            False,
        )
    else:
        v0 = tl.load(X + off).to(tl.float32)
    if len(META) > 4:
        v1 = _load_strided(
            X,
            groups,
            lanes + WIDTH * 1,
            GROUP,
            PER_ROW,
            ROW_STRIDE,
            COLUMN_STRIDE,
            False,
        )
    else:
        v1 = tl.load(X + off + WIDTH).to(tl.float32)
    ab = tl.maximum(tl.abs(v0), tl.abs(v1), propagate_nan=tl.PropagateNan.ALL)
    if SPLIT == 4:
        if len(META) > 4:
            v2 = _load_strided(
                X,
                groups,
                lanes + WIDTH * 2,
                GROUP,
                PER_ROW,
                ROW_STRIDE,
                COLUMN_STRIDE,
                False,
            )
        else:
            v2 = tl.load(X + off + WIDTH * 2).to(tl.float32)
        if len(META) > 4:
            v3 = _load_strided(
                X,
                groups,
                lanes + WIDTH * 3,
                GROUP,
                PER_ROW,
                ROW_STRIDE,
                COLUMN_STRIDE,
                False,
            )
        else:
            v3 = tl.load(X + off + WIDTH * 3).to(tl.float32)
        ab = tl.maximum(
            ab,
            tl.maximum(
                tl.abs(v2), tl.abs(v3), propagate_nan=tl.PropagateNan.ALL
            ),
            propagate_nan=tl.PropagateNan.ALL,
        )
    amax = tl.maximum(
        tl.reduce(ab, 1, _maximum_nan),
        0.0001,
        propagate_nan=tl.PropagateNan.ALL,
    )
    scale = amax * 0.0022321429569274187
    if ROUND:
        exponent = tl.ceil(libdevice.log2(scale)).to(tl.int32)
        bits = tl.minimum(tl.maximum(exponent + 127, 1), 254) << 23
        bits = tl.where(amax == float("inf"), 8388608, bits)
        bits = tl.where(amax != amax, 1065353216, bits)
        scale = bits.to(tl.float32, bitcast=True)
    if X.dtype.element_ty == tl.bfloat16:
        q0 = _encode_bf16(_split_divide_bf16(v0, scale[:, None], True), v0)
    else:
        q0 = _encode(
            tl.minimum(
                tl.maximum(
                    _split_divide_bf16(v0, scale[:, None], False),
                    -448.0,
                    propagate_nan=tl.PropagateNan.ALL,
                ),
                448.0,
                propagate_nan=tl.PropagateNan.ALL,
            )
        )
    if X.dtype.element_ty == tl.bfloat16:
        q1 = _encode_bf16(_split_divide_bf16(v1, scale[:, None], True), v1)
    else:
        q1 = _encode(
            tl.minimum(
                tl.maximum(
                    _split_divide_bf16(v1, scale[:, None], False),
                    -448.0,
                    propagate_nan=tl.PropagateNan.ALL,
                ),
                448.0,
                propagate_nan=tl.PropagateNan.ALL,
            )
        )
    tl.store(Q + off, q0)
    tl.store(Q + off + WIDTH, q1)
    if SPLIT == 4:
        if X.dtype.element_ty == tl.bfloat16:
            q2 = _encode_bf16(_split_divide_bf16(v2, scale[:, None], True), v2)
        else:
            q2 = _encode(
                tl.minimum(
                    tl.maximum(
                        _split_divide_bf16(v2, scale[:, None], False),
                        -448.0,
                        propagate_nan=tl.PropagateNan.ALL,
                    ),
                    448.0,
                    propagate_nan=tl.PropagateNan.ALL,
                )
            )
        if X.dtype.element_ty == tl.bfloat16:
            q3 = _encode_bf16(_split_divide_bf16(v3, scale[:, None], True), v3)
        else:
            q3 = _encode(
                tl.minimum(
                    tl.maximum(
                        _split_divide_bf16(v3, scale[:, None], False),
                        -448.0,
                        propagate_nan=tl.PropagateNan.ALL,
                    ),
                    448.0,
                    propagate_nan=tl.PropagateNan.ALL,
                )
            )
        tl.store(Q + off + WIDTH * 2, q2)
        tl.store(Q + off + WIDTH * 3, q3)
    tl.store(S + groups, scale)


@triton.jit(do_not_specialize=["Q", "S"])
def _address_kernel(X, Q, S, META: tl.constexpr):
    output = tl.multiple_of(Q.to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _kernel(X, output, scales, META)


@triton.jit(do_not_specialize=["Q", "S"])
def _address_split_kernel(X, Q, S, META: tl.constexpr):
    output = tl.multiple_of(Q.to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _split_kernel(X, output, scales, META)


@triton.jit(do_not_specialize=["X", "Q", "S"])
def _all_address_kernel(X, Q, S, META: tl.constexpr, POINTERS: tl.constexpr):
    INPUT_TYPE: tl.constexpr = POINTERS[0]
    INPUT_ALIGNMENT: tl.constexpr = POINTERS[1]
    values = tl.multiple_of(
        X.to(tl.uint64).to(tl.pointer_type(INPUT_TYPE)), INPUT_ALIGNMENT
    )
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _kernel(values, output, scales, META)


@triton.jit(do_not_specialize=["X", "Q", "S"])
def _all_address_split_kernel(
    X, Q, S, META: tl.constexpr, POINTERS: tl.constexpr
):
    INPUT_TYPE: tl.constexpr = POINTERS[0]
    INPUT_ALIGNMENT: tl.constexpr = POINTERS[1]
    values = tl.multiple_of(
        X.to(tl.uint64).to(tl.pointer_type(INPUT_TYPE)), INPUT_ALIGNMENT
    )
    output = tl.multiple_of(Q.to(tl.uint64).to(tl.pointer_type(tl.uint8)), 16)
    scales = tl.multiple_of(
        S.to(tl.uint64).to(tl.pointer_type(tl.float32)), 16
    )
    _split_kernel(values, output, scales, META)


def act_quant(x, block_size=128, scale_fmt=None):
    shape = x.shape
    hip = torch.version.hip is not None
    strides = ()
    if not x.is_contiguous():
        if (
            not hip
            and len(shape) == 2
            and (x.stride(1) * x.element_size() <= 64)
        ):
            strides = (shape[-1] // block_size,) + x.stride()
        else:
            x = x.contiguous()
    q = x.new_empty(shape, dtype=torch.float8_e4m3fn)
    s = x.new_empty(
        shape[:-1] + (shape[-1] // block_size,), dtype=torch.float32
    )
    groups = x.numel() // block_size
    if groups:
        output = q.data_ptr()
        assert output % 16 == 0
        if hip:
            values = x.data_ptr()
            scales = s.data_ptr()
            assert scales % 16 == 0
            input_type = (
                tl.bfloat16
                if x.dtype == torch.bfloat16
                else tl.float16 if x.dtype == torch.float16 else tl.float32
            )
            pointers = (input_type, min(16, values & -values))
        else:
            scales = s.data_ptr()
            assert scales % 16 == 0
        rows = min(16 if hip else 4, groups & -groups)
        if not hip and block_size > 128:
            block = 1 << (block_size - 1).bit_length()
            rows = min(rows, max(1, 512 // block))
        tiles = groups // rows
        warps = 8 if hip or block_size > 2048 else 1
        if (
            not hip
            and block_size >= 2
            and (block_size & block_size - 1 == 0)
            and (scale_fmt is None)
            and (groups * block_size >= 8388608)
        ) and groups * block_size < 2147483648:
            meta = (block_size, rows, False, 2)
            if strides:
                meta += strides
            if hip:
                _all_address_split_kernel.run(
                    values,
                    output,
                    scales,
                    meta,
                    pointers,
                    grid=(65535 if tiles > 2147483647 else tiles,),
                    warmup=False,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )
            else:
                _address_split_kernel.run(
                    x,
                    output,
                    scales,
                    meta,
                    grid=(65535 if tiles > 2147483647 else tiles,),
                    warmup=False,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )
        else:
            meta = (
                block_size,
                tiles,
                1 << (block_size - 1).bit_length(),
                rows,
                scale_fmt is not None,
                hip and block_size % 4 == 0,
                hip,
            )
            if strides:
                meta += strides
            if hip:
                _all_address_kernel.run(
                    values,
                    output,
                    scales,
                    meta,
                    pointers,
                    grid=(65535 if tiles > 2147483647 else tiles,),
                    warmup=False,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )
            else:
                _address_kernel.run(
                    x,
                    output,
                    scales,
                    meta,
                    grid=(65535 if tiles > 2147483647 else tiles,),
                    warmup=False,
                    num_warps=warps,
                    enable_fp_fusion=False,
                )
    return (q, s)


__all__ = ["act_quant"]
