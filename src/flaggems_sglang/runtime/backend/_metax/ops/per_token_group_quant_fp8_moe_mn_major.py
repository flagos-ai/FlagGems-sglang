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
def _map(
    OFF,
    PS,
    BASE,
    COUNT,
    KGROUPS: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    M: tl.constexpr,
    E: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(0)
    off = tl.load(OFF + e * OS)
    m = tl.load(PS + e * PS0)
    next_off = tl.load(OFF + (e + 1) * OS, e + 1 < E, M)
    begin = tl.where(e == 0, 0, off)
    lane = tl.arange(0, B)
    row = begin + lane
    member = (row >= off) & (row < off + m)
    base = tl.where(member, off * KGROUPS + row - off, 0)
    count = tl.where(member, m, 0)
    tl.store(
        BASE + row,
        base.to(tl.uint32).to(tl.uint64) | (count.to(tl.uint64) << 32),
        row < next_off,
    )
    for tile in range(1, tl.cdiv(next_off - begin, B)):
        row = begin + tile * B + lane
        member = (row >= off) & (row < off + m)
        base = tl.where(member, off * KGROUPS + row - off, 0)
        count = tl.where(member, m, 0)
        tl.store(
            BASE + row,
            base.to(tl.uint32).to(tl.uint64) | (count.to(tl.uint64) << 32),
            row < next_off,
        )


@triton.jit
def _quant(
    A,
    Q,
    S,
    BASE,
    COUNT,
    KGROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    TOTAL_GROUPS: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    tile = tl.program_id(0)
    ids = tile * NG + tl.arange(0, NG)
    rows = ids // KGROUPS
    packed = tl.load(BASE + rows)
    base = packed.to(tl.int32)
    count = (packed.to(tl.uint64) >> 32).to(tl.int32)
    valid = count > 0
    cols = tl.arange(0, B)
    offsets = ids[:, None] * GROUP + cols[None, :]
    x = tl.load(
        A + offsets, valid[:, None] & (cols[None, :] < GROUP), other=0
    ).to(tl.float32)
    amax = tl.maximum(
        tl.max(x.to(tl.uint32, bitcast=True) & 0x7FFFFFFF, 1), 0x38D1B717
    ).to(tl.float32, bitcast=True)
    inv = tl.div_rn(1.0, amax) * 448.0
    v = x * inv[:, None]
    code = _encode(v)
    tl.store(Q + offsets, code, valid[:, None] & (cols[None, :] < GROUP))
    dst = base + (ids % KGROUPS) * count
    tl.store(
        S + dst,
        tl.fma(amax, 0.002232142724096775, amax * 1.3304608803554885e-10),
        valid,
    )


@triton.jit
def _small(
    A,
    Q,
    S,
    OFF,
    PS,
    E: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    KGROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    TOTAL_GROUPS: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
):
    tile = tl.program_id(0)
    ids = tile * NG + tl.arange(0, NG)
    rows = ids // KGROUPS
    base = tl.full((NG,), 0, tl.int32)
    count = tl.full((NG,), 0, tl.int32)
    for expert in tl.static_range(E):
        first = tl.load(OFF + expert * OS)
        size = tl.load(PS + expert * PS0)
        member = (rows >= first) & (rows < first + size)
        base = tl.where(member, first * KGROUPS + rows - first, base)
        count = tl.where(member, size, count)
    valid = count > 0
    cols = tl.arange(0, B)
    offsets = ids[:, None] * GROUP + cols[None, :]
    x = tl.load(
        A + offsets, valid[:, None] & (cols[None, :] < GROUP), other=0
    ).to(tl.float32)
    amax = tl.maximum(
        tl.max(x.to(tl.uint32, bitcast=True) & 0x7FFFFFFF, 1), 0x38D1B717
    ).to(tl.float32, bitcast=True)
    inv = tl.div_rn(1.0, amax) * 448.0
    v = x * inv[:, None]
    code = _encode(v)
    tl.store(Q + offsets, code, valid[:, None] & (cols[None, :] < GROUP))
    dst = base + (ids % KGROUPS) * count
    tl.store(
        S + dst,
        tl.fma(amax, 0.002232142724096775, amax * 1.3304608803554885e-10),
        valid,
    )


@triton.jit
def _owner(
    A,
    Q,
    S,
    OFF,
    PS,
    KG: tl.constexpr,
    G: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
    STEPS: tl.constexpr,
):
    first = tl.program_id(0) * NG
    row = first // KG
    lo = 0
    hi = (1 << STEPS) - 1
    for step in tl.static_range(STEPS):
        mid = (lo + hi + 1) // 2
        off = tl.load(OFF + mid * OS, mid < E, 2147483647)
        before = (mid < E) & (off <= row)
        lo = tl.where(before, mid, lo)
        hi = tl.where(before, hi, mid - 1)
    expert = lo
    start = tl.load(OFF + expert * OS, expert >= 0, 0)
    count = tl.load(PS + expert * PS0, expert >= 0, 0)
    valid = (expert >= 0) & (row >= start) & (row < start + count)
    ids = first + tl.arange(0, NG)
    cols = tl.arange(0, B)
    offsets = ids[:, None] * G + cols[None, :]
    mask = valid & (cols[None, :] < G)
    x = tl.load(A + offsets, mask, 0).to(tl.float32)
    amax = tl.maximum(
        tl.max(x.to(tl.uint32, bitcast=True) & 0x7FFFFFFF, 1), 0x38D1B717
    ).to(tl.float32, bitcast=True)
    v = x * (tl.div_rn(1.0, amax) * 448.0)[:, None]
    tl.store(Q + offsets, _encode(v), mask)
    dst = start * KG + (ids % KG) * count + row - start
    tl.store(
        S + dst,
        tl.fma(amax, 0.002232142724096775, amax * 1.3304608803554885e-10),
        valid,
    )


def per_token_group_quant_fp8_moe_mn_major(
    A, expert_offsets, problem_sizes, group_size
):
    if (
        A.numel() >= 2147483648
        or problem_sizes.shape[0] > 65535
        or group_size > 4096
        or (
            group_size == 128
            and triton.cdiv(
                A.shape[0], 8 if A.numel() // group_size >= 98304 else 1
            )
            > 65535
        )
    ):
        return _large_entry(A, expert_offsets, problem_sizes, group_size)
    m, k = A.shape
    groups = k // group_size
    q = torch.empty((m, k), dtype=torch.uint8, device=A.device)
    s = torch.empty((m, groups), dtype=torch.float32, device=A.device)
    total = m * groups
    owner_ng = 4 if total <= 8192 else 8
    if (
        total
        and group_size == 128
        and groups % owner_ng == 0
        and 0 < problem_sizes.shape[0] <= 8
        and total < 32768
    ):
        _owner[(total // owner_ng,)](
            A,
            q,
            s,
            expert_offsets,
            problem_sizes,
            groups,
            group_size,
            problem_sizes.shape[0],
            triton.next_power_of_2(problem_sizes.shape[0]),
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            owner_ng,
            128,
            (problem_sizes.shape[0] - 1).bit_length(),
            num_warps=1,
            enable_fp_fusion=False,
        )
    elif total and problem_sizes.shape[0]:
        block = triton.next_power_of_2(group_size)
        fused = problem_sizes.shape[0] <= 4 and total <= (
            16384 if group_size == 128 else 8192
        )
        low_warps = group_size == 128 and (total >= 32768 or fused)
        ng = (
            (8 if total <= 1024 or (not fused and total <= 65536) else 16)
            if low_warps
            else 32
        )
        while total % ng != 0 or ng * block > (
            8192 if group_size == 128 else 4096
        ):
            ng //= 2
            if ng == 1:
                break
        if fused:
            _small[(total // ng,)](
                A,
                q,
                s,
                expert_offsets,
                problem_sizes,
                problem_sizes.shape[0],
                expert_offsets.stride(0),
                problem_sizes.stride(0),
                groups,
                group_size,
                total,
                ng,
                block,
                num_warps=(1 if low_warps else 4),
                enable_fp_fusion=False,
            )
        else:
            if group_size == 128 and 32768 <= total < 98304:
                cluster = 8 if total >= 98304 else 1
                _predicted[
                    (cluster * triton.cdiv(groups, 8), triton.cdiv(m, cluster))
                ](
                    A,
                    q,
                    s,
                    expert_offsets,
                    problem_sizes,
                    problem_sizes.shape[0],
                    expert_offsets.stride(0),
                    problem_sizes.stride(0),
                    (problem_sizes.shape[0] - 1).bit_length(),
                    groups,
                    group_size,
                    total,
                    8,
                    block,
                    cluster,
                    num_warps=1,
                    pipeline="cpasync",
                    scenario="disable_int8_opt",
                    enable_fp_fusion=False,
                )
            else:
                base = torch.empty((m,), dtype=torch.int64, device=A.device)
                count = base
                _map[(problem_sizes.shape[0],)](
                    expert_offsets,
                    problem_sizes,
                    base,
                    count,
                    groups,
                    expert_offsets.stride(0),
                    problem_sizes.stride(0),
                    m,
                    problem_sizes.shape[0],
                    128,
                )
                if group_size == 128:
                    cluster = 8 if total >= 98304 else 1
                    _grid2[
                        (
                            cluster * triton.cdiv(groups, 8),
                            triton.cdiv(m, cluster),
                        )
                    ](
                        A,
                        q,
                        s,
                        base,
                        count,
                        groups,
                        group_size,
                        total,
                        8,
                        block,
                        cluster,
                        num_warps=1,
                        pipeline="cpasync",
                        scenario="disable_int8_opt",
                        enable_fp_fusion=False,
                    )
                else:
                    _quant[(total // ng,)](
                        A,
                        q,
                        s,
                        base,
                        count,
                        groups,
                        group_size,
                        total,
                        ng,
                        block,
                        num_warps=(1 if low_warps else 4),
                        enable_fp_fusion=False,
                    )
    return q.view(torch.float8_e4m3fn), s


@triton.jit
def _grid2(
    A,
    Q,
    S,
    BASE,
    COUNT,
    KGROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    TOTAL_GROUPS: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
    ORDER: tl.constexpr,
):
    rows = tl.program_id(1) * ORDER + tl.program_id(0) % ORDER
    groups = (tl.program_id(0) // ORDER) * NG + tl.arange(0, NG)
    ids = rows * KGROUPS + groups
    packed = tl.load(
        BASE + rows,
        (
            True
            if (TOTAL_GROUPS // KGROUPS) % ORDER == 0
            else rows < TOTAL_GROUPS // KGROUPS
        ),
        0,
    )
    base = packed.to(tl.int32)
    count = (packed.to(tl.uint64) >> 32).to(tl.int32)
    if count > 0:
        valid = True if KGROUPS % NG == 0 else groups < KGROUPS
        cols = tl.arange(0, B)
        offsets = ids[:, None] * GROUP + cols[None, :]
        bits = (
            tl.load(
                A + offsets, valid[:, None] & (cols[None, :] < GROUP), other=0
            )
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
        magnitude = bits & 32767
        x = (magnitude << 16).to(tl.float32, bitcast=True)
        amax = tl.maximum(tl.max(magnitude, 1) << 16, 0x38D1B717).to(
            tl.float32, bitcast=True
        )
        inv = tl.div_rn(1.0, amax) * 448.0
        v = x * inv[:, None]
        code = _encode_finite(v, (bits >> 8) & 128)
        if tl.max(amax.to(tl.uint32, bitcast=True), 0) >= 2139095040:
            code = tl.where(v != v, 127, code).to(tl.uint8)
        tl.store(Q + offsets, code, valid[:, None] & (cols[None, :] < GROUP))
        dst = base + groups * count
        tl.store(
            S + dst,
            tl.fma(amax, 0.002232142724096775, amax * 1.3304608803554885e-10),
            valid,
        )


@triton.jit
def _encode_positive(v, sign):
    raw = v.to(tl.int32, bitcast=True)
    exponent = tl.maximum(raw >> 23, 121)
    bias = exponent * 8388616 + 167771192
    rounded = (v + bias.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    code = (rounded & 255) | sign
    return tl.where(v != v, 127, code).to(tl.uint8)


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


@triton.jit
def _lookup_owner(
    row,
    OFF,
    PS,
    E: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    STEPS: tl.constexpr,
):
    if E <= 128:
        experts = tl.arange(0, triton.next_power_of_2(E))
        offsets = tl.load(OFF + experts * OS, experts < E, 0)
        counts = tl.load(PS + experts * PS0, experts < E, 0)
        member = (experts < E) & (row >= offsets) & (row < offsets + counts)
        packed = (offsets.to(tl.uint32) << 16) | counts.to(tl.uint32)
        selected = tl.max(tl.where(member, packed, 0), 0)
        start = (selected >> 16).to(tl.int32)
        count = (selected & 65535).to(tl.int32)
        return start, count
    else:
        lo = 0
        hi = (1 << STEPS) - 1
        for step in tl.static_range(STEPS):
            mid = (lo + hi + 1) // 2
            offset = tl.load(OFF + mid * OS, mid < E, 2147483647)
            before = (mid < E) & (offset <= row)
            lo = tl.where(before, mid, lo)
            hi = tl.where(before, hi, mid - 1)
        start = tl.load(OFF + lo * OS)
        count = tl.load(PS + lo * PS0)
        count = tl.where((row >= start) & (row < start + count), count, 0)
        return start, count


@triton.jit
def _predicted(
    A,
    Q,
    S,
    OFF,
    PS,
    E: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    STEPS: tl.constexpr,
    KGROUPS: tl.constexpr,
    GROUP: tl.constexpr,
    TOTAL_GROUPS: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
    ORDER: tl.constexpr,
):
    rows = tl.program_id(1) * ORDER + tl.program_id(0) % ORDER
    groups = (tl.program_id(0) // ORDER) * NG + tl.arange(0, NG)
    ids = rows * KGROUPS + groups
    M: tl.constexpr = TOTAL_GROUPS // KGROUPS
    EXPECTED: tl.constexpr = max(1, M // E)
    if rows < M:
        if E <= 8:
            start, count = _lookup_owner(rows, OFF, PS, E, OS, PS0, STEPS)
        else:
            guess = tl.minimum(rows // EXPECTED, E - 1)
            start = tl.load(OFF + guess * OS)
            count = tl.load(PS + guess * PS0)
            if (rows < start) | (rows >= start + count):
                start, count = _lookup_owner(rows, OFF, PS, E, OS, PS0, STEPS)
        base = start * KGROUPS + rows - start
    else:
        base = 0
        count = 0
    if count > 0:
        valid = True if KGROUPS % NG == 0 else groups < KGROUPS
        cols = tl.arange(0, B)
        offsets = ids[:, None] * GROUP + cols[None, :]
        bits = (
            tl.load(
                A + offsets, valid[:, None] & (cols[None, :] < GROUP), other=0
            )
            .to(tl.uint16, bitcast=True)
            .to(tl.uint32)
        )
        magnitude = bits & 32767
        x = (magnitude << 16).to(tl.float32, bitcast=True)
        amax = tl.maximum(tl.max(magnitude, 1) << 16, 0x38D1B717).to(
            tl.float32, bitcast=True
        )
        inv = tl.div_rn(1.0, amax) * 448.0
        v = x * inv[:, None]
        code = _encode_positive(v, (bits >> 8) & 128)
        tl.store(Q + offsets, code, valid[:, None] & (cols[None, :] < GROUP))
        dst = base + groups * count
        tl.store(
            S + dst,
            tl.fma(amax, 0.002232142724096775, amax * 1.3304608803554885e-10),
            valid,
        )


@triton.jit
def _encode_finite(v, sign):
    raw = v.to(tl.int32, bitcast=True)
    exponent = tl.maximum(raw >> 23, 121)
    bias = exponent * 8388616 + 167771192
    rounded = (v + bias.to(tl.float32, bitcast=True)).to(
        tl.int32, bitcast=True
    )
    code = (rounded & 255) | sign
    return code.to(tl.uint8)


__all__ = ["per_token_group_quant_fp8_moe_mn_major"]
