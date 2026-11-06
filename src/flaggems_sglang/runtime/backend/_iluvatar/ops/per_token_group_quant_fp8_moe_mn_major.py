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
    raw = v.to(tl.uint32, bitcast=True)
    sign = (raw >> 24) & 128
    mag = raw & 0x7FFFFFFF
    limit = tl.full((), 448.0, tl.float32).to(tl.uint32, bitcast=True)
    mag = tl.minimum(mag, limit)
    exp = (mag >> 23).to(tl.int32) - 127
    normal = ((mag + ((1 << 19) - 1) + ((mag >> 20) & 1)) >> 20) - 960
    shift = 14 - exp
    safe_shift = tl.minimum(tl.maximum(shift, 1), 24)
    mant = (mag & 0x7FFFFF) | 0x800000
    sub = (
        mant + ((1 << (safe_shift - 1)) - 1) + ((mant >> safe_shift) & 1)
    ) >> safe_shift
    sub = tl.where(shift > 24, 0, sub)
    code = tl.where(exp >= -6, normal, sub)
    return (code | sign).to(tl.uint8)


@triton.jit
def _map(
    OFF,
    PS,
    BASE,
    COUNT,
    KGROUPS: tl.constexpr,
    OS: tl.constexpr,
    PS0: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(0)
    off = tl.load(OFF + e * OS)
    m = tl.load(PS + e * PS0)
    lane = tl.arange(0, B)
    for tile in range(tl.cdiv(m, B)):
        row = tile * B + lane
        tl.store(BASE + off + row, off * KGROUPS + row, row < m)
        tl.store(COUNT + off + row, m, row < m)


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
    cols = tl.arange(0, B)
    offsets = ids[:, None] * GROUP + cols[None, :]
    x = tl.load(A + offsets, cols[None, :] < GROUP, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-4)
    inv = tl.div_rn(448.0, amax)
    v = tl.minimum(tl.maximum(x * inv[:, None], -448.0), 448.0)
    code = _encode(v)
    tl.store(Q + offsets, code, cols[None, :] < GROUP)
    rows = ids // KGROUPS
    base = tl.load(BASE + rows)
    count = tl.load(COUNT + rows)
    dst = base + (ids % KGROUPS) * count
    tl.store(S + dst, tl.div_rn(amax, 448.0))


def per_token_group_quant_fp8_moe_mn_major(
    A, expert_offsets, problem_sizes, group_size
):
    m, k = A.shape
    groups = k // group_size
    q = torch.empty((m, k), dtype=torch.uint8, device=A.device)
    s = torch.empty((m, groups), dtype=torch.float32, device=A.device)
    base = torch.empty((m,), dtype=torch.int32, device=A.device)
    count = torch.empty((m,), dtype=torch.int32, device=A.device)
    total = m * groups
    if total:
        _map[(problem_sizes.shape[0],)](
            expert_offsets,
            problem_sizes,
            base,
            count,
            groups,
            expert_offsets.stride(0),
            problem_sizes.stride(0),
            128,
        )
        block = triton.next_power_of_2(group_size)
        ng = 8
        while total % ng != 0 or ng * block > 2048:
            ng //= 2
            if ng == 1:
                break
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
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q.view(torch.float8_e4m3fn), s


__all__ = ["per_token_group_quant_fp8_moe_mn_major"]
