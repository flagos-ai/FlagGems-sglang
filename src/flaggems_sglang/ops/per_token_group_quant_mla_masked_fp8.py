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
def _encode_folded(value):
    raw = value.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    absolute = tl.minimum(
        tl.abs(value), 448.0, propagate_nan=tl.PropagateNan.NONE
    )
    magnitude = absolute.to(tl.uint32, bitcast=True)
    exponent = tl.maximum(magnitude >> 23, 121)
    magic_bits = (exponent * 8388616 + 167771192) | sign
    rounded = (absolute + magic_bits.to(tl.float32, bitcast=True)).to(
        tl.uint32, bitcast=True
    )
    return rounded.to(tl.uint8)


@triton.jit
def _maximum_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _quantize(X, Q, S, MASKED, EPS, META: tl.constexpr):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    GROUP: tl.constexpr = META[2]
    ALIGNED_M: tl.constexpr = (M + 255) // 256 * 256
    if META[3] is None:
        STRIDE_B: tl.constexpr = M * K
        STRIDE_M: tl.constexpr = K
        STRIDE_K: tl.constexpr = 1
    else:
        STRIDE_B: tl.constexpr = META[3]
        STRIDE_M: tl.constexpr = META[4]
        STRIDE_K: tl.constexpr = META[5]
    ROWS: tl.constexpr = (
        4
        if (M * (K // GROUP)) % 4 == 0
        else 2 if (M * (K // GROUP)) % 2 == 0 else 1
    )
    BLOCK: tl.constexpr = triton.next_power_of_2(GROUP)
    EXACT_SCALE: tl.constexpr = False
    RECIPROCAL: tl.constexpr = META[6]
    WIDE: tl.constexpr = META[7]
    STREAM: tl.constexpr = META[8]
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
        cache_modifier=".cg" if STREAM else "",
    ).to(tl.float32)
    if STREAM or WIDE:
        maximum_bits = tl.max(
            value.to(tl.uint32, bitcast=True) & 2147483647, axis=1
        )
        maximum = maximum_bits.to(tl.float32, bitcast=True)
        absolute = tl.maximum(maximum, EPS)
        absolute = tl.where(maximum_bits > 2139095040, maximum, absolute)
    else:
        maximum = tl.reduce(tl.abs(value), 1, _maximum_nan)
        absolute = tl.maximum(maximum, EPS)
        absolute = tl.where(maximum != maximum, maximum, absolute)
    if EXACT_SCALE:
        scale = tl.div_rn(absolute, 448.0)
    else:
        scale = absolute * 0.0022321429569274187
    if RECIPROCAL:
        inverse = tl.div_rn(1.0, scale[:, None])
        coarse = value * inverse
        safe_scale = tl.where(
            scale[:, None] < float("inf"), scale[:, None], 1.0
        )
        residual = tl.fma(-coarse, safe_scale, value)
        divided = tl.fma(residual, inverse, coarse)
    else:
        divided = tl.div_rn(value, scale[:, None])
    code = tl.where(divided != divided, 127, _encode_folded(divided)).to(
        tl.uint8
    )
    tl.store(
        Q + batch * ALIGNED_M * K + tokens[:, None] * K + columns,
        code,
        lanes[None, :] < GROUP,
        cache_modifier=".cs" if STREAM else "",
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
    s = x.new_empty((b, groups, aligned_m), dtype=torch.float32)
    masked = x.new_empty((b,), dtype=torch.int32)
    tiles = m * groups
    if b and tiles:
        rows = min(4, tiles & -tiles)
        stream = torch.version.hip is not None
        contiguous = x.is_contiguous() if not stream else False
        if not stream and b * aligned_m * k < 2147483648 and contiguous:
            strides = (None, None, None)
            wide = False
        else:
            strides = x.stride()
            if max(strides) > 2147483647:
                strides = tuple(
                    0 if size == 1 else stride
                    for size, stride in zip(x.shape, strides)
                )
            wide = b * aligned_m * k > 2147483648
            if not wide and not (x.is_contiguous() if stream else contiguous):
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
            *strides,
            1e-30 <= eps < float("inf"),
            wide,
            stream,
        )
        if not stream and (tiles // rows > 2147483647 or b > 65535):
            _quantize_bounded.run(
                x,
                q,
                s,
                masked,
                eps,
                meta,
                b,
                grid=(min(tiles // rows, 65535), min(b, 65535)),
                warmup=False,
                num_warps=4 if group_size > 2048 else 1,
                enable_fp_fusion=False,
            )
        else:
            _quantize.run(
                x,
                q,
                s,
                masked,
                eps,
                meta,
                grid=(tiles // rows, b),
                warmup=False,
                num_warps=4 if group_size > 2048 else 1,
                enable_fp_fusion=False,
            )
    elif b:
        _fill_masked.run(
            masked, m, b, 128, grid=((b + 127) // 128,), warmup=False
        )
    return q.view(torch.float8_e4m3fn), s.transpose(1, 2), masked, m, aligned_m


@triton.jit
def _quantize_tile(X, Q, S, MASKED, EPS, META: tl.constexpr, PID, BATCH):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    GROUP: tl.constexpr = META[2]
    ALIGNED_M: tl.constexpr = (M + 255) // 256 * 256
    if META[3] is None:
        STRIDE_B: tl.constexpr = M * K
        STRIDE_M: tl.constexpr = K
        STRIDE_K: tl.constexpr = 1
    else:
        STRIDE_B: tl.constexpr = META[3]
        STRIDE_M: tl.constexpr = META[4]
        STRIDE_K: tl.constexpr = META[5]
    ROWS: tl.constexpr = (
        4
        if M * (K // GROUP) % 4 == 0
        else 2 if M * (K // GROUP) % 2 == 0 else 1
    )
    BLOCK: tl.constexpr = triton.next_power_of_2(GROUP)
    EXACT_SCALE: tl.constexpr = False
    RECIPROCAL: tl.constexpr = META[6]
    WIDE: tl.constexpr = META[7]
    STREAM: tl.constexpr = META[8]
    GROUPS: tl.constexpr = K // GROUP
    pid = PID
    batch = BATCH
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
        cache_modifier=".cg" if STREAM else "",
    ).to(tl.float32)
    if STREAM or WIDE:
        maximum_bits = tl.max(
            value.to(tl.uint32, bitcast=True) & 2147483647, axis=1
        )
        maximum = maximum_bits.to(tl.float32, bitcast=True)
        absolute = tl.maximum(maximum, EPS)
        absolute = tl.where(maximum_bits > 2139095040, maximum, absolute)
    else:
        maximum = tl.reduce(tl.abs(value), 1, _maximum_nan)
        absolute = tl.maximum(maximum, EPS)
        absolute = tl.where(maximum != maximum, maximum, absolute)
    if EXACT_SCALE:
        scale = tl.div_rn(absolute, 448.0)
    else:
        scale = absolute * 0.0022321429569274187
    if RECIPROCAL:
        inverse = tl.div_rn(1.0, scale[:, None])
        coarse = value * inverse
        safe_scale = tl.where(
            scale[:, None] < float("inf"), scale[:, None], 1.0
        )
        residual = tl.fma(-coarse, safe_scale, value)
        divided = tl.fma(residual, inverse, coarse)
    else:
        divided = tl.div_rn(value, scale[:, None])
    code = tl.where(divided != divided, 127, _encode_folded(divided)).to(
        tl.uint8
    )
    tl.store(
        Q + batch * ALIGNED_M * K + tokens[:, None] * K + columns,
        code,
        lanes[None, :] < GROUP,
        cache_modifier=".cs" if STREAM else "",
    )
    tl.store(
        S + batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens, scale
    )
    if pid == 0:
        tl.store(MASKED + batch, M)


@triton.jit
def _quantize_bounded(
    X, Q, S, MASKED, EPS, META: tl.constexpr, B: tl.constexpr
):
    M: tl.constexpr = META[0]
    K: tl.constexpr = META[1]
    GROUP: tl.constexpr = META[2]
    ROWS: tl.constexpr = (
        4
        if (M * (K // GROUP)) % 4 == 0
        else 2 if (M * (K // GROUP)) % 2 == 0 else 1
    )
    TILES: tl.constexpr = M * (K // GROUP) // ROWS
    width = tl.num_programs(0).to(tl.int64)
    linear = (
        tl.program_id(0).to(tl.int64) + tl.program_id(1).to(tl.int64) * width
    )
    step = width * tl.num_programs(1).to(tl.int64)
    limit = tl.full((), B * TILES, tl.int64)
    while linear < limit:
        batch = linear // TILES
        tile = linear % TILES
        _quantize_tile(X, Q, S, MASKED, EPS, META, tile, batch)
        linear += tl.minimum(step, limit - linear)


__all__ = ["per_token_group_quant_mla_masked_fp8"]
