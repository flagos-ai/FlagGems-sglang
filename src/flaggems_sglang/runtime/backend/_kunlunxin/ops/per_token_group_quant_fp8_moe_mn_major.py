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
def _encode(v):
    raw = v.to(tl.int32, bitcast=True)
    mag = raw & 2147483647
    exponent = tl.maximum(mag >> 23, 121)
    bias = exponent * 8388616 + 167771192
    rounded = (
        mag.to(tl.float32, bitcast=True) + bias.to(tl.float32, bitcast=True)
    ).to(tl.int32, bitcast=True)
    code = rounded & 255 | raw >> 24 & 128
    return tl.where(mag > 2139095040, 127, code).to(tl.uint8)


@triton.jit
def _block(
    A,
    Q,
    S,
    start,
    count,
    tile,
    KG: tl.constexpr,
    G: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
    FULL: tl.constexpr,
):
    if FULL and G == B and G % 4 == 0:
        _pair_full(A, Q, S, start, count, tile, KG, G, NG, B)
    else:
        ids = tile * NG + tl.arange(0, NG)
        cols = tl.arange(0, B)
        offsets = (start * KG + ids[:, None]) * G + cols[None, :]
        mask = (
            None
            if FULL and G == B
            else (ids[:, None] < count * KG) & (cols[None, :] < G)
        )
        bits = tl.load(
            A + offsets,
            mask,
            None if FULL and G == B else 0,
            offset_state_policy="continuous" if G == B else "unknown",
        ).to(tl.uint32)
        x = (bits << 16).to(tl.float32, bitcast=True)
        ordered = (((bits & 32767) << 8) | 1065353216).to(
            tl.float32, bitcast=True
        )
        key = (
            tl.max(ordered, 1).to(tl.uint32, bitcast=True) - 1065353216
        ) << 8
        amax = tl.maximum(key, 953267991).to(tl.float32, bitcast=True)
        v = x * _reference_inverse(amax)[:, None]
        tl.store(
            Q + offsets,
            _encode(v),
            mask,
            offset_state_policy="continuous" if G == B else "unknown",
        )
        dst = start * KG + ids % KG * count + ids // KG
        tl.store(
            S + dst,
            tl.fma(amax, 0.002232142724096775, amax * 1.3304608803554885e-10),
            True if FULL else ids < count * KG,
        )


def per_token_group_quant_fp8_moe_mn_major(
    A, expert_offsets, problem_sizes, group_size
):
    if (
        A.numel() >= 2147483648
        or problem_sizes.shape[0] > 65535
        or group_size > 4096
    ):
        return _large_entry(A, expert_offsets, problem_sizes, group_size)
    m, k = A.shape
    kg = k // group_size
    e = problem_sizes.shape[0]
    q = torch.empty((m, k), dtype=torch.uint8, device=A.device)
    s = torch.empty((m, kg), dtype=torch.float32, device=A.device)
    if m * kg and e:
        waves = 4 if m * kg >= 8192 else 1
        if m * kg >= 8192:
            _expert_full[max(3, triton.cdiv(12, e)), e](
                A.view(torch.uint16),
                q,
                s,
                expert_offsets,
                problem_sizes,
                kg,
                group_size,
                expert_offsets.stride(0),
                problem_sizes.stride(0),
                64,
                triton.next_power_of_2(group_size),
                num_warps=4,
                enable_fp_fusion=False,
            )
            _expert_tail[1, e](
                A.view(torch.uint16),
                q,
                s,
                expert_offsets,
                problem_sizes,
                kg,
                group_size,
                expert_offsets.stride(0),
                problem_sizes.stride(0),
                64,
                triton.next_power_of_2(group_size),
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _expert[max(waves, triton.cdiv(12 * waves, e)), e](
                A.view(torch.uint16),
                q,
                s,
                expert_offsets,
                problem_sizes,
                kg,
                group_size,
                expert_offsets.stride(0),
                problem_sizes.stride(0),
                64,
                triton.next_power_of_2(group_size),
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (q.view(torch.float8_e4m3fn), s)


@triton.jit
def _expert(
    A,
    Q,
    S,
    OFF,
    PS,
    KG: tl.constexpr,
    G: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(1)
    start = tl.load(OFF + e * OS)
    count = tl.load(PS + e * PS0)
    for tile in range(
        tl.program_id(0), tl.cdiv(count * KG, NG), tl.num_programs(0)
    ):
        _block(A, Q, S, start, count, tile, KG, G, NG, B, False)


@triton.jit
def _reference_inverse(amax):
    bits = amax.to(tl.int32, bitcast=True)
    exponent = bits >> 23
    index = bits >> 16 & 127
    divisor = index + 128
    normalized = (index << 16 | 1065353216).to(tl.float32, bitcast=True)
    reciprocal = tl.div_rn(1.0, normalized)
    q = reciprocal.to(tl.int32, bitcast=True) - (125 << 23)
    normal_bits = q + (252 - tl.minimum(exponent, 252) << 23)
    inv = normal_bits.to(tl.float32, bitcast=True) * 448.0
    shift = tl.where(exponent >= 254, 2, 1)
    high_q = q >> shift
    product = q.to(tl.uint32) * divisor.to(tl.uint32)
    low = q & (1 << shift) - 1
    half = 1 << shift - 1
    high_q += (
        (low > half)
        | (low == half)
        & (
            (product < 2147483648)
            | (product == 2147483648) & (high_q & 1 != 0)
        )
    ).to(tl.int32)
    high_bits = high_q.to(tl.float32).to(tl.int32, bitcast=True) - (141 << 23)
    high_inv = high_bits.to(tl.float32, bitcast=True) * 1.75
    inv = tl.where(exponent > 252, high_inv, inv)
    inv = tl.where(bits < 953286656, 4480000.0, inv)
    inv = tl.where(bits == 2139095040, 0.0, inv)
    return tl.where(bits > 2139095040, float("nan"), inv)


@triton.jit
def _wide_parts(
    A,
    P,
    OFF,
    PS,
    KG: tl.constexpr,
    G: tl.constexpr,
    NC: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    EXPERT_START: tl.constexpr,
):
    expert = tl.program_id(0) + EXPERT_START
    start = tl.load(OFF + expert * OS).to(tl.int64)
    count = tl.load(PS + expert * PS0).to(tl.int64)
    lane = tl.arange(0, 128)
    for tile in range(
        tl.program_id(1).to(tl.int64),
        tl.cdiv(count * KG * NC, 64),
        tl.num_programs(1),
    ):
        ids = tile * 64 + tl.arange(0, 64)
        group = ids // NC
        col = ids % NC * 128
        indices = (
            start * KG + tl.minimum(group[:, None], count * KG - 1)
        ) * G + tl.minimum(col[:, None] + lane[None, :], G - 1)
        valid = (group[:, None] < count * KG) & (
            col[:, None] + lane[None, :] < G
        )
        x = tl.where(valid, tl.load(A + indices).to(tl.float32), 0.0)
        keys = (x.to(tl.uint32, bitcast=True) >> 16 & 32767).to(tl.float32)
        maximum = tl.max(keys, 1)
        tl.store(P + start * KG * NC + ids, maximum, ids < count * KG * NC)


@triton.jit
def _wide_scales(
    P,
    InvPtr,
    S,
    OFF,
    PS,
    KG: tl.constexpr,
    NC: tl.constexpr,
    BC: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    EXPERT_START: tl.constexpr,
):
    expert = tl.program_id(0) + EXPERT_START
    start = tl.load(OFF + expert * OS).to(tl.int64)
    count = tl.load(PS + expert * PS0).to(tl.int64)
    lanes = tl.arange(0, BC)
    for tile in range(
        tl.program_id(1).to(tl.int64),
        tl.cdiv(count * KG, 8),
        tl.num_programs(1),
    ):
        groups = tile * 8 + tl.arange(0, 8)
        valid = groups < count * KG
        keys = tl.load(
            P
            + (start * KG + tl.minimum(groups[:, None], count * KG - 1)) * NC
            + tl.minimum(lanes[None, :], NC - 1)
        )
        keys = tl.where(valid[:, None] & (lanes[None, :] < NC), keys, 0.0)
        maximum = tl.max(keys, 1).to(tl.uint32) << 16
        amax = tl.maximum(maximum, 953267991).to(tl.float32, bitcast=True)
        inverse = _reference_inverse(amax)
        tl.store(InvPtr + start * KG + groups, inverse, valid)
        target = start * KG + groups % KG * count + groups // KG
        tl.store(
            S + target,
            tl.fma(amax, 0.002232142724096775, amax * 1.3304608803554885e-10),
            valid,
        )


@triton.jit
def _wide_quant(
    A,
    Q,
    InvPtr,
    OFF,
    PS,
    KG: tl.constexpr,
    G: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    EXPERT_START: tl.constexpr,
):
    expert = tl.program_id(0) + EXPERT_START
    start = tl.load(OFF + expert * OS).to(tl.int64)
    count = tl.load(PS + expert * PS0).to(tl.int64)
    lane = tl.arange(0, 1024)
    for tile in range(
        tl.program_id(1).to(tl.int64),
        tl.cdiv(count * KG * G, 1024),
        tl.num_programs(1),
    ):
        ids = tile * 1024 + lane
        valid = ids < count * KG * G
        x = tl.load(
            A + start * KG * G + ids,
            valid,
            0,
            offset_state_policy="continuous",
        ).to(tl.float32)
        inverse = tl.load(InvPtr + start * KG + ids // G, valid, 0)
        tl.store(
            Q + start * KG * G + ids,
            _encode(x * inverse),
            valid,
            offset_state_policy="continuous",
        )


def _large_entry(A, expert_offsets, problem_sizes, group_size):
    m, k = A.shape
    kg = k // group_size
    e = problem_sizes.shape[0]
    q = torch.empty((m, k), dtype=torch.uint8, device=A.device)
    s = torch.empty((m, kg), dtype=torch.float32, device=A.device)
    if m * kg and e:
        nc = triton.cdiv(group_size, 128)
        partials = torch.empty(
            (m * kg * nc,), dtype=torch.float32, device=A.device
        )
        inverse = torch.empty((m * kg,), dtype=torch.float32, device=A.device)
        for expert_start in range(0, e, 65535):
            grid = (min(65535, e - expert_start), max(1, 12 // e))
            _wide_parts[grid](
                A,
                partials,
                expert_offsets,
                problem_sizes,
                kg,
                group_size,
                nc,
                expert_offsets.stride(0),
                problem_sizes.stride(0),
                expert_start,
                enable_fp_fusion=False,
            )
            _wide_scales[grid](
                partials,
                inverse,
                s,
                expert_offsets,
                problem_sizes,
                kg,
                nc,
                triton.next_power_of_2(nc),
                expert_offsets.stride(0),
                problem_sizes.stride(0),
                expert_start,
                enable_fp_fusion=False,
            )
            _wide_quant[grid](
                A,
                q,
                inverse,
                expert_offsets,
                problem_sizes,
                kg,
                group_size,
                expert_offsets.stride(0),
                problem_sizes.stride(0),
                expert_start,
                enable_fp_fusion=False,
            )
    return q.view(torch.float8_e4m3fn), s


@triton.jit
def _expert_full(
    A,
    Q,
    S,
    OFF,
    PS,
    KG: tl.constexpr,
    G: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(1)
    start = tl.load(OFF + e * OS)
    count = tl.load(PS + e * PS0)
    for tile in range(tl.program_id(0), count * KG // NG, tl.num_programs(0)):
        _block(A, Q, S, start, count, tile, KG, G, NG, B, True)


@triton.jit
def _expert_tail(
    A,
    Q,
    S,
    OFF,
    PS,
    KG: tl.constexpr,
    G: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(1)
    start = tl.load(OFF + e * OS)
    count = tl.load(PS + e * PS0)
    for tile in range(count * KG // NG, tl.cdiv(count * KG, NG)):
        _block(A, Q, S, start, count, tile, KG, G, NG, B, False)


@triton.jit
def _pair_full(
    A,
    Q,
    S,
    start,
    count,
    tile,
    KG: tl.constexpr,
    G: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    ids = tile * NG + tl.arange(0, NG)
    cols = tl.arange(0, B // 4)
    offsets = (start * KG + ids[:, None]) * (G // 4) + cols[None, :]
    words = tl.load(
        A.to(tl.pointer_type(tl.uint64)) + offsets,
        offset_state_policy="continuous",
    )
    b0 = (words & 65535).to(tl.uint32)
    b1 = ((words >> 16) & 65535).to(tl.uint32)
    b2 = ((words >> 32) & 65535).to(tl.uint32)
    b3 = (words >> 48).to(tl.uint32)
    keys = tl.maximum(
        tl.maximum(b0 & 32767, b1 & 32767), tl.maximum(b2 & 32767, b3 & 32767)
    )
    ordered = ((keys << 8) | 1065353216).to(tl.float32, bitcast=True)
    key = (tl.max(ordered, 1).to(tl.uint32, bitcast=True) - 1065353216) << 8
    amax = tl.maximum(key, 953267991).to(tl.float32, bitcast=True)
    inv = _reference_inverse(amax)[:, None]
    q0 = _encode((b0 << 16).to(tl.float32, bitcast=True) * inv).to(tl.uint32)
    q1 = _encode((b1 << 16).to(tl.float32, bitcast=True) * inv).to(tl.uint32)
    q2 = _encode((b2 << 16).to(tl.float32, bitcast=True) * inv).to(tl.uint32)
    q3 = _encode((b3 << 16).to(tl.float32, bitcast=True) * inv).to(tl.uint32)
    tl.store(
        Q.to(tl.pointer_type(tl.uint32)) + offsets,
        q0 | (q1 << 8) | (q2 << 16) | (q3 << 24),
        offset_state_policy="continuous",
    )
    dst = start * KG + ids % KG * count + ids // KG
    tl.store(
        S + dst,
        tl.fma(amax, 0.002232142724096775, amax * 1.3304608803554885e-10),
    )


__all__ = ["per_token_group_quant_fp8_moe_mn_major"]
