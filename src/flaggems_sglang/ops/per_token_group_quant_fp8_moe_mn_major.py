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
    code = (rounded & 255) | ((raw >> 24) & 128)
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
    ids = tile * NG + tl.arange(0, NG)
    cols = tl.arange(0, B)
    offsets = (start * KG + ids[:, None]) * G + cols[None, :]
    mask = (
        (cols[None, :] < G)
        if FULL
        else ((ids[:, None] < count * KG) & (cols[None, :] < G))
    )
    x = tl.load(A + offsets, mask, 0, cache_modifier=".cg").to(tl.float32)
    amax = tl.maximum(
        tl.max(x.to(tl.uint32, bitcast=True) & 0x7FFFFFFF, 1), 0x38D1B717
    ).to(tl.float32, bitcast=True)
    v = x * (tl.div_rn(1.0, amax) * 448.0)[:, None]
    tl.store(Q + offsets, _encode(v), mask, cache_modifier=".cs")
    dst = start * KG + (ids % KG) * count + ids // KG
    tl.store(
        S + dst,
        tl.fma(amax, 0.002232142724096775, amax * 1.3304608803554885e-10),
        True if FULL else (ids < count * KG),
    )


@triton.jit
def _expert_multi(
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
    if count * KG == tl.num_programs(0) * NG * 2:
        for wave in tl.static_range(2):
            _block(
                A,
                Q,
                S,
                start,
                count,
                tl.program_id(0) + wave * tl.num_programs(0),
                KG,
                G,
                NG,
                B,
                True,
            )
    else:
        tile = tl.program_id(0)
        if tile * NG < count * KG:
            if (tile + 1) * NG <= count * KG:
                _block(A, Q, S, start, count, tile, KG, G, NG, B, True)
            else:
                _block(A, Q, S, start, count, tile, KG, G, NG, B, False)
        for tile in range(
            tl.program_id(0) + tl.num_programs(0),
            tl.cdiv(count * KG, NG),
            tl.num_programs(0),
        ):
            if (tile + 1) * NG <= count * KG:
                _block(A, Q, S, start, count, tile, KG, G, NG, B, True)
            else:
                _block(A, Q, S, start, count, tile, KG, G, NG, B, False)


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
        kernel = (
            _expert_multi if group_size == 128 and m * kg >= 65536 else _expert
        )
        kernel[
            (
                max(
                    1,
                    triton.cdiv(
                        m * kg,
                        8
                        * e
                        * (2 if group_size == 128 and m * kg >= 65536 else 1),
                    ),
                ),
                e,
            )
        ](
            A,
            q,
            s,
            expert_offsets,
            problem_sizes,
            kg,
            group_size,
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            8,
            triton.next_power_of_2(group_size),
            num_warps=(4 if group_size == 128 else 8),
            enable_fp_fusion=False,
        )
    return q.view(torch.float8_e4m3fn), s


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
    tile = tl.program_id(0)
    if tile * NG < count * KG:
        if (tile + 1) * NG <= count * KG:
            _block(A, Q, S, start, count, tile, KG, G, NG, B, True)
        else:
            _block(A, Q, S, start, count, tile, KG, G, NG, B, False)
    for tile in range(
        tl.program_id(0) + tl.num_programs(0),
        tl.cdiv(count * KG, NG),
        tl.num_programs(0),
    ):
        if (tile + 1) * NG <= count * KG:
            _block(A, Q, S, start, count, tile, KG, G, NG, B, True)
        else:
            _block(A, Q, S, start, count, tile, KG, G, NG, B, False)


@triton.jit
def _large_address(
    A,
    Q,
    S,
    OFF,
    PS,
    E: tl.constexpr,
    KG: tl.constexpr,
    G: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
):
    pass
    lane = tl.arange(0, 128)
    for expert in range(tl.program_id(1).to(tl.int64), E, tl.num_programs(1)):
        start = tl.load(OFF + expert * OS).to(tl.int64)
        count = tl.load(PS + expert * PS0).to(tl.int64)
        for group in range(
            tl.program_id(0).to(tl.int64), count * KG, tl.num_programs(0)
        ):
            base = (start * KG + group) * G
            maximum = tl.full((1, 128), 0, tl.uint32)
            for chunk in range(tl.cdiv(G, 128)):
                col = chunk.to(tl.int64) * 128 + lane
                x = tl.load(A + base + col[None, :], col[None, :] < G, 0).to(
                    tl.float32
                )
                maximum = tl.maximum(
                    maximum, x.to(tl.uint32, bitcast=True) & 2147483647
                )
            amax = tl.maximum(tl.max(maximum, 1), 953267991).to(
                tl.float32, bitcast=True
            )
            inverse = tl.div_rn(1.0, amax) * 448.0
            dst = start * KG + (group % KG) * count + group // KG
            tl.store(
                S + dst + tl.arange(0, 1),
                tl.fma(
                    amax, 0.002232142724096775, amax * 1.3304608803554885e-10
                ),
            )
            for chunk in range(tl.cdiv(G, 128)):
                col = chunk.to(tl.int64) * 128 + lane
                x = tl.load(A + base + col[None, :], col[None, :] < G, 0).to(
                    tl.float32
                )
                tl.store(
                    Q + base + col[None, :],
                    _encode(x * inverse[:, None]),
                    col[None, :] < G,
                )


def _large_entry(A, expert_offsets, problem_sizes, group_size):
    m, k = A.shape
    kg = k // group_size
    e = problem_sizes.shape[0]
    q = torch.empty((m, k), dtype=torch.uint8, device=A.device)
    s = torch.empty((m, kg), dtype=torch.float32, device=A.device)
    if m * kg and e:
        _large_address[(32, min(e, 40))](
            A,
            q,
            s,
            expert_offsets,
            problem_sizes,
            e,
            kg,
            group_size,
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q.view(torch.float8_e4m3fn), s


__all__ = ["per_token_group_quant_fp8_moe_mn_major"]
