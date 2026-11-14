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
def _encode_bounded(value):
    raw = value.to(tl.uint32, bitcast=True)
    sign = raw >> 24 & 128
    magnitude = tl.minimum(raw & 2147483647, 1139802112)
    exponent = tl.maximum(magnitude >> 23, 121)
    magic_bits = (exponent * 8388616 + 167771192) | sign
    rounded = (
        magnitude.to(tl.float32, bitcast=True)
        + magic_bits.to(tl.float32, bitcast=True)
    ).to(tl.uint32, bitcast=True)
    return rounded.to(tl.uint8)


@triton.jit
def _quantize_original(X, Q, S, MASKED, EPS, META: tl.constexpr):
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
    ROWS: tl.constexpr = min(8, (M * (K // GROUP)) & -(M * (K // GROUP)))
    BLOCK: tl.constexpr = triton.next_power_of_2(GROUP)
    EXACT_SCALE: tl.constexpr = False
    WIDE: tl.constexpr = META[6]
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
    divided = tl.div_rn(value, scale[:, None])
    if META[7]:
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
    if STRIDE_K == 1 and GROUP % 4 == 0 and ROWS >= 4 and ROWS * BLOCK <= 1024:
        scale_offsets = (
            batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens
        ).to(tl.int64)
        scale_bits = scale.to(tl.uint32, bitcast=True)
        scale_bytes = (scale_bits[:, None] >> ((lanes[None, :] & 3) * 8)).to(
            tl.uint8
        )
        scale_ptrs = (
            S.to(tl.pointer_type(tl.uint8))
            + scale_offsets[:, None] * 4
            + lanes[None, :]
        )
        tl.store(scale_ptrs, scale_bytes, lanes[None, :] < 4)
    else:
        tl.store(
            S + batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens, scale
        )
    if pid == 0:
        tl.store(MASKED + batch, M)


@triton.jit
def _quantize_halves(X, Q, S, MASKED, EPS, META: tl.constexpr):
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
    ROWS: tl.constexpr = min(8, (M * (K // GROUP)) & -(M * (K // GROUP)))
    BLOCK: tl.constexpr = triton.next_power_of_2(GROUP)
    EXACT_SCALE: tl.constexpr = False
    WIDE: tl.constexpr = META[6]
    WIDTH: tl.constexpr = BLOCK // 2
    GROUPS: tl.constexpr = K // GROUP
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    if WIDE:
        pid = pid.to(tl.int64)
        batch = batch.to(tl.int64)
    indices = pid * ROWS + tl.arange(0, ROWS)
    tokens = indices // GROUPS
    groups = indices % GROUPS
    lanes = tl.arange(0, WIDTH)
    if WIDE:
        lanes = lanes.to(tl.int64)
    columns = groups[:, None] * GROUP + lanes[None, :]
    value0 = tl.load(
        X + batch * STRIDE_B + tokens[:, None] * STRIDE_M + columns * STRIDE_K,
        lanes[None, :] < GROUP,
        other=0.0,
    ).to(tl.float32)
    value1 = tl.load(
        X
        + batch * STRIDE_B
        + tokens[:, None] * STRIDE_M
        + (columns + WIDTH) * STRIDE_K,
        lanes[None, :] + WIDTH < GROUP,
        other=0.0,
    ).to(tl.float32)
    magnitude0 = value0.to(tl.uint32, bitcast=True) & 2147483647
    magnitude1 = value1.to(tl.uint32, bitcast=True) & 2147483647
    maximum_bits = tl.max(tl.maximum(magnitude0, magnitude1), axis=1)
    maximum = maximum_bits.to(tl.float32, bitcast=True)
    absolute = tl.maximum(maximum, EPS)
    absolute = tl.where(maximum_bits > 2139095040, maximum, absolute)
    if EXACT_SCALE:
        scale = tl.div_rn(absolute, 448.0)
    else:
        scale = absolute * 0.0022321429569274187
    divided0 = tl.div_rn(value0, scale[:, None])
    divided1 = tl.div_rn(value1, scale[:, None])
    if META[7]:
        code0 = _encode_bounded(divided0)
        code1 = _encode_bounded(divided1)
    else:
        quantized0 = tl.minimum(tl.maximum(divided0, -448.0), 448.0)
        quantized1 = tl.minimum(tl.maximum(divided1, -448.0), 448.0)
        code0 = tl.where(divided0 != divided0, 127, _encode(quantized0)).to(
            tl.uint8
        )
        code1 = tl.where(divided1 != divided1, 127, _encode(quantized1)).to(
            tl.uint8
        )
    tl.store(
        Q + batch * ALIGNED_M * K + tokens[:, None] * K + columns,
        code0,
        lanes[None, :] < GROUP,
    )
    tl.store(
        Q + batch * ALIGNED_M * K + tokens[:, None] * K + columns + WIDTH,
        code1,
        lanes[None, :] + WIDTH < GROUP,
    )
    scale_offsets = (
        batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens
    ).to(tl.int64)
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_halves = (scale_bits[:, None] >> ((lanes[None, :] & 1) * 16)).to(
        tl.uint16
    )
    scale_ptrs = (
        S.to(tl.pointer_type(tl.uint16))
        + scale_offsets[:, None] * 2
        + lanes[None, :]
    )
    tl.store(scale_ptrs, scale_halves, lanes[None, :] < 2)
    if pid == 0:
        tl.store(MASKED + batch, M)


@triton.jit
def _quantize(X, Q, S, MASKED, EPS, META: tl.constexpr):
    ROWS: tl.constexpr = min(
        8, (META[0] * (META[1] // META[2])) & -(META[0] * (META[1] // META[2]))
    )
    BLOCK: tl.constexpr = triton.next_power_of_2(META[2])
    STRIDE_K: tl.constexpr = 1 if META[3] is None else META[5]
    if (
        STRIDE_K == 1
        and META[2] % 4 == 0
        and ROWS >= 4
        and ROWS * BLOCK >= 512
        and ROWS * BLOCK <= 1024
    ):
        _quantize_halves(X, Q, S, MASKED, EPS, META)
    else:
        _quantize_original(X, Q, S, MASKED, EPS, META)


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
    q = torch.empty((b, aligned_m, k), device=x.device, dtype=torch.uint8)
    s = torch.empty(
        (b, groups, aligned_m), device=x.device, dtype=torch.float32
    )
    masked = torch.empty((b,), device=x.device, dtype=torch.int32)
    tiles = m * groups
    if b and tiles:
        rows = min(8, tiles & -tiles)
        wide = b * aligned_m * k > 2147483648
        contiguous = x.is_contiguous() if not wide else False
        if contiguous and b * aligned_m * k < 2147483648:
            strides = (None, None, None)
        else:
            strides = x.stride()
            if max(strides) > 2147483647:
                strides = tuple(
                    0 if size == 1 else stride
                    for size, stride in zip(x.shape, strides)
                )
            if not wide:
                wide = (
                    sum(
                        (size - 1) * stride
                        for size, stride in zip(x.shape, strides)
                    )
                    >= 2147483648
                )
        meta = (m, k, group_size, *strides, wide, 1e-30 <= eps < float("inf"))
        if tiles // rows <= 16777215 and b <= 65535:
            _quantize.run(
                x,
                q,
                s,
                masked,
                eps,
                meta,
                grid=(tiles // rows, b),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
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
                num_warps=4,
                enable_fp_fusion=False,
            )
    elif b:
        if (b + 127) // 128 <= 16777215:
            _fill_masked.run(
                masked, m, b, 128, grid=((b + 127) // 128,), warmup=False
            )
        else:
            _fill_masked_bounded.run(
                masked, m, b, 128, grid=(65535,), warmup=False, num_warps=4
            )
    return q.view(torch.float8_e4m3fn), s.transpose(1, 2), masked, m, aligned_m


@triton.jit
def _quantize_original_tile(
    X, Q, S, MASKED, EPS, META: tl.constexpr, pid, batch
):
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
    ROWS: tl.constexpr = min(8, M * (K // GROUP) & -(M * (K // GROUP)))
    BLOCK: tl.constexpr = triton.next_power_of_2(GROUP)
    EXACT_SCALE: tl.constexpr = False
    WIDE: tl.constexpr = META[6]
    GROUPS: tl.constexpr = K // GROUP
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
    divided = tl.div_rn(value, scale[:, None])
    if META[7]:
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
    if (
        STRIDE_K == 1
        and GROUP % 4 == 0
        and (ROWS >= 4)
        and (ROWS * BLOCK <= 1024)
    ):
        scale_offsets = (
            batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens
        ).to(tl.int64)
        scale_bits = scale.to(tl.uint32, bitcast=True)
        scale_bytes = (scale_bits[:, None] >> (lanes[None, :] & 3) * 8).to(
            tl.uint8
        )
        scale_ptrs = (
            S.to(tl.pointer_type(tl.uint8))
            + scale_offsets[:, None] * 4
            + lanes[None, :]
        )
        tl.store(scale_ptrs, scale_bytes, lanes[None, :] < 4)
    else:
        tl.store(
            S + batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens, scale
        )
    if pid == 0:
        tl.store(MASKED + batch, M)


@triton.jit
def _quantize_halves_tile(
    X, Q, S, MASKED, EPS, META: tl.constexpr, pid, batch
):
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
    ROWS: tl.constexpr = min(8, M * (K // GROUP) & -(M * (K // GROUP)))
    BLOCK: tl.constexpr = triton.next_power_of_2(GROUP)
    EXACT_SCALE: tl.constexpr = False
    WIDE: tl.constexpr = META[6]
    WIDTH: tl.constexpr = BLOCK // 2
    GROUPS: tl.constexpr = K // GROUP
    if WIDE:
        pid = pid.to(tl.int64)
        batch = batch.to(tl.int64)
    indices = pid * ROWS + tl.arange(0, ROWS)
    tokens = indices // GROUPS
    groups = indices % GROUPS
    lanes = tl.arange(0, WIDTH)
    if WIDE:
        lanes = lanes.to(tl.int64)
    columns = groups[:, None] * GROUP + lanes[None, :]
    value0 = tl.load(
        X + batch * STRIDE_B + tokens[:, None] * STRIDE_M + columns * STRIDE_K,
        lanes[None, :] < GROUP,
        other=0.0,
    ).to(tl.float32)
    value1 = tl.load(
        X
        + batch * STRIDE_B
        + tokens[:, None] * STRIDE_M
        + (columns + WIDTH) * STRIDE_K,
        lanes[None, :] + WIDTH < GROUP,
        other=0.0,
    ).to(tl.float32)
    magnitude0 = value0.to(tl.uint32, bitcast=True) & 2147483647
    magnitude1 = value1.to(tl.uint32, bitcast=True) & 2147483647
    maximum_bits = tl.max(tl.maximum(magnitude0, magnitude1), axis=1)
    maximum = maximum_bits.to(tl.float32, bitcast=True)
    absolute = tl.maximum(maximum, EPS)
    absolute = tl.where(maximum_bits > 2139095040, maximum, absolute)
    if EXACT_SCALE:
        scale = tl.div_rn(absolute, 448.0)
    else:
        scale = absolute * 0.0022321429569274187
    divided0 = tl.div_rn(value0, scale[:, None])
    divided1 = tl.div_rn(value1, scale[:, None])
    if META[7]:
        code0 = _encode_bounded(divided0)
        code1 = _encode_bounded(divided1)
    else:
        quantized0 = tl.minimum(tl.maximum(divided0, -448.0), 448.0)
        quantized1 = tl.minimum(tl.maximum(divided1, -448.0), 448.0)
        code0 = tl.where(divided0 != divided0, 127, _encode(quantized0)).to(
            tl.uint8
        )
        code1 = tl.where(divided1 != divided1, 127, _encode(quantized1)).to(
            tl.uint8
        )
    tl.store(
        Q + batch * ALIGNED_M * K + tokens[:, None] * K + columns,
        code0,
        lanes[None, :] < GROUP,
    )
    tl.store(
        Q + batch * ALIGNED_M * K + tokens[:, None] * K + columns + WIDTH,
        code1,
        lanes[None, :] + WIDTH < GROUP,
    )
    scale_offsets = (
        batch * GROUPS * ALIGNED_M + groups * ALIGNED_M + tokens
    ).to(tl.int64)
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_halves = (scale_bits[:, None] >> (lanes[None, :] & 1) * 16).to(
        tl.uint16
    )
    scale_ptrs = (
        S.to(tl.pointer_type(tl.uint16))
        + scale_offsets[:, None] * 2
        + lanes[None, :]
    )
    tl.store(scale_ptrs, scale_halves, lanes[None, :] < 2)
    if pid == 0:
        tl.store(MASKED + batch, M)


@triton.jit
def _quantize_tile(X, Q, S, MASKED, EPS, META: tl.constexpr, pid, batch):
    ROWS: tl.constexpr = min(
        8, META[0] * (META[1] // META[2]) & -(META[0] * (META[1] // META[2]))
    )
    BLOCK: tl.constexpr = triton.next_power_of_2(META[2])
    STRIDE_K: tl.constexpr = 1 if META[3] is None else META[5]
    if (
        STRIDE_K == 1
        and META[2] % 4 == 0
        and (ROWS >= 4)
        and (ROWS * BLOCK >= 512)
        and (ROWS * BLOCK <= 1024)
    ):
        _quantize_halves_tile(X, Q, S, MASKED, EPS, META, pid, batch)
    else:
        _quantize_original_tile(X, Q, S, MASKED, EPS, META, pid, batch)


@triton.jit
def _quantize_bounded(
    X, Q, S, MASKED, EPS, META: tl.constexpr, BATCHES: tl.constexpr
):
    GROUPS: tl.constexpr = META[1] // META[2]
    ROWS: tl.constexpr = min(8, (META[0] * GROUPS) & -(META[0] * GROUPS))
    TILES: tl.constexpr = META[0] * GROUPS // ROWS
    width = tl.num_programs(0).to(tl.int64)
    flat = (
        tl.program_id(0).to(tl.int64) + tl.program_id(1).to(tl.int64) * width
    )
    step = width * tl.num_programs(1).to(tl.int64)
    limit = tl.full((), TILES * BATCHES, tl.int64)
    while flat < limit:
        _quantize_tile(X, Q, S, MASKED, EPS, META, flat % TILES, flat // TILES)
        flat += tl.minimum(step, limit - flat)


@triton.jit
def _fill_masked_bounded(
    MASKED, M: tl.constexpr, B: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    step = tl.num_programs(0).to(tl.int64)
    limit = tl.full((), (B + BLOCK - 1) // BLOCK, tl.int64)
    while pid < limit:
        indices = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(MASKED + indices, M, indices < B)
        pid += tl.minimum(step, limit - pid)


__all__ = ["per_token_group_quant_mla_masked_fp8"]
