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
def _encode(value):
    raw = value.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = raw & 2147483647
    normal = ((magnitude + 524287 + (magnitude >> 20 & 1)) >> 20) - 960
    subnormal = (tl.abs(value) + 16384.0).to(
        tl.uint32, bitcast=True
    ) - 1182793728
    code = tl.where(tl.abs(value) >= 0.015625, normal, subnormal)
    code = tl.where(value != value, 127, code)
    return (code | sign).to(tl.uint8)


@triton.jit
def _encode_bounded(value):
    raw = value.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = tl.minimum(
        tl.abs(value), 480.0, propagate_nan=tl.PropagateNan.NONE
    ).to(tl.uint32, bitcast=True)
    exponent = tl.maximum(magnitude >> 23, 121)
    magic_bits = (exponent * 8388616 + 167771192) | sign
    rounded = (
        magnitude.to(tl.float32, bitcast=True)
        + magic_bits.to(tl.float32, bitcast=True)
    ).to(tl.uint32, bitcast=True)
    return rounded.to(tl.uint8)


@triton.jit(do_not_specialize=["S", "MASKED"])
def _quantize(X, Q, S, MASKED, EPS, META: tl.constexpr):
    S = S.to(tl.uint64).to(tl.pointer_type(tl.float32))
    MASKED = MASKED.to(tl.uint64).to(tl.pointer_type(tl.int32))
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    GROUP: tl.constexpr = META[2]
    ALIGNED_M: tl.constexpr = META[3]
    STRIDE_B: tl.constexpr = META[4]
    STRIDE_M: tl.constexpr = META[5]
    STRIDE_K: tl.constexpr = META[6]
    ROWS: tl.constexpr = META[7]
    BLOCK: tl.constexpr = META[8]
    EXACT_SCALE: tl.constexpr = META[9]
    RECIPROCAL: tl.constexpr = META[10]
    WIDE: tl.constexpr = META[11]
    GROUPS: tl.constexpr = K // GROUP
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    if WIDE:
        pid = pid.to(tl.int64)
        batch = batch.to(tl.int64)
    indices = pid * ROWS + tl.arange(0, ROWS)
    tokens = indices // GROUPS
    groups = indices % GROUPS
    lanes = tl.arange(0, BLOCK)
    if WIDE:
        lanes = lanes.to(tl.int64)
    columns = groups[:, None] * GROUP + lanes[None, :]
    value = tl.load(
        X + batch * STRIDE_B + tokens[:, None] * STRIDE_M + columns * STRIDE_K,
        lanes[None, :] < GROUP,
        other=0.0,
    ).to(tl.float32)
    maximum_bits = tl.max(
        value.to(tl.uint32, bitcast=True) & 2147483647, axis=1
    )
    maximum = maximum_bits.to(tl.float32, bitcast=True)
    absolute = tl.maximum(maximum, EPS)
    absolute = tl.where(maximum_bits > 2139095040, maximum, absolute)
    if EXACT_SCALE:
        scale = tl.div_rn(absolute, 448.0)
    else:
        scale = absolute * 0.0022321429569274187
    if RECIPROCAL:
        inverse = libdevice.fast_dividef(1.0, scale[:, None])
        coarse = value * inverse
        safe_scale = tl.where(
            scale[:, None] < float("inf"), scale[:, None], 1.0
        )
        residual = tl.fma(coarse, -safe_scale, value)
        divided = tl.fma(residual, inverse, coarse)
    else:
        divided = tl.div_rn(value, scale[:, None])
    if RECIPROCAL:
        code = _encode_bounded(divided)
    else:
        quantized = tl.minimum(tl.maximum(divided, -448.0), 448.0)
        code = tl.where(divided != divided, 127, _encode(quantized)).to(
            tl.uint8
        )
    tl.store(
        Q + batch * ALIGNED_M * K + tokens[:, None] * K + columns,
        code,
        lanes[None, :] < GROUP,
    )
    tl.store(
        S + batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens, scale
    )
    if pid == 0:
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
    b, m, k = x.shape
    aligned_m = (m + 255) // 256 * 256
    groups = k // group_size
    q = x.new_empty((b, aligned_m, k), dtype=torch.uint8)
    scale_stride = aligned_m or 1
    s = x.new_empty_strided(
        (b, aligned_m, groups),
        ((groups or 1) * scale_stride, 1, scale_stride),
        dtype=torch.float32,
    )
    masked = x.new_empty((b,), dtype=torch.int32)
    tiles = m * groups
    if b and tiles:
        rows = min(16, tiles & -tiles)
        block = 1 << (group_size - 1).bit_length()
        strides = x.stride()
        if max(strides) > 2147483647:
            strides = tuple(
                0 if size == 1 else stride
                for size, stride in zip(x.shape, strides)
            )
        wide = b * aligned_m * k > 2147483648
        if not wide and not x.is_contiguous():
            wide = (
                sum(
                    (size - 1) * stride
                    for size, stride in zip(x.shape, strides)
                )
                >= 2147483648
            )
        meta = (
            m,
            k,
            group_size,
            aligned_m,
            *strides,
            rows,
            block,
            False,
            1e-30 <= eps < float("inf"),
            wide,
        )
        _quantize.run(
            x,
            q,
            s.data_ptr(),
            masked.data_ptr(),
            eps,
            meta,
            grid=(tiles // rows, b),
            warmup=False,
            num_warps=4,
            enable_fp_fusion=False,
        )
    elif b:
        _fill_masked.run(
            masked, m, b, 128, grid=((b + 127) // 128,), warmup=False
        )
    return q.view(torch.float8_e4m3fn), s, masked, m, aligned_m


__all__ = ["per_token_group_quant_mla_masked_fp8"]
