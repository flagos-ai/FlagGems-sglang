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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Concatenate MLA query components along the last dimension."""

import torch
import triton
import triton.language as tl


@triton.jit
def _c2_dense_two(
    o_ptr,
    a_ptr,
    b_ptr,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    """两段式: (ROWS, A_LAST) + (ROWS, B_LAST) 各一次连续 load/store."""
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    da = tl.arange(0, A_LAST)
    db = tl.arange(0, B_LAST)
    ob = rows[:, None] * D
    a = tl.load(a_ptr + rows[:, None] * A_LAST + da[None, :])
    b = tl.load(b_ptr + rows[:, None] * B_LAST + db[None, :])
    tl.store(o_ptr + ob + da[None, :], a)
    tl.store(o_ptr + ob + A_LAST + db[None, :], b)


@triton.jit
def _c2_dense_one(
    o_ptr,
    a_ptr,
    b_ptr,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    """单段式: 一次 (ROWS, D) 稠密无谓词 store 覆盖整个输出行块."""
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    dc = tl.arange(0, D)
    is_a = (dc < A_LAST)[None, :]
    av = tl.load(
        a_ptr + rows[:, None] * A_LAST + dc[None, :], mask=is_a, other=0
    )
    bv = tl.load(
        b_ptr + rows[:, None] * B_LAST + (dc - A_LAST)[None, :],
        mask=~is_a,
        other=0,
    )
    tl.store(o_ptr + rows[:, None] * D + dc[None, :], tl.where(is_a, av, bv))


@triton.jit(do_not_specialize=["n_rows"])
def _c2_rows(
    o_ptr,
    a_ptr,
    b_ptr,
    n_rows,
    A_LAST: tl.constexpr,
    B_LAST: tl.constexpr,
    D: tl.constexpr,
    BA: tl.constexpr,
    BB: tl.constexpr,
    ROWS: tl.constexpr,
):
    """通用兜底: 任意行数 / 任意行宽 (padded arange + mask)."""
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    da = tl.arange(0, BA)
    db = tl.arange(0, BB)
    am = rmask[:, None] & (da < A_LAST)[None, :]
    bm = rmask[:, None] & (db < B_LAST)[None, :]
    ob = rows[:, None] * D
    a = tl.load(a_ptr + rows[:, None] * A_LAST + da[None, :], mask=am, other=0)
    b = tl.load(b_ptr + rows[:, None] * B_LAST + db[None, :], mask=bm, other=0)
    tl.store(o_ptr + ob + da[None, :], a, mask=am)
    tl.store(o_ptr + ob + A_LAST + db[None, :], b, mask=bm)


_PAYLOAD_ELEMS_BF16 = 24576


def cap(dtype):
    return max(1, (_PAYLOAD_ELEMS_BF16 * 2) // dtype.itemsize)


_DENSE_TIERS = (
    (4096, 16, 4),  # t1 (rows=128, grid=8) / t8 (rows=1024, grid=64)
    (32768, 32, 4),  # t64: rows=8192, grid=256
    (262144, 64, 4),  # t512: rows=65536, grid=1024
    (1 << 62, 128, 8),  # t4096: rows=524288, grid=4096
)

# 稠密单段式 (_c2_dense_one) 是否优于两段式: 由本机 sweep 标定 (t4096 一次
# 事件计时 1342us vs 两段式 1304us, 单段式不优), 保持两段式.
_DENSE_ONE = False

_DENSE_STAGES = 1

# 通用路径 per-program 行数.
_GEN_ROWS = 16


def _tier(rows):
    for lim, r, w in _DENSE_TIERS:
        if rows <= lim:
            return r, w
    return _DENSE_TIERS[-1][1], _DENSE_TIERS[-1][2]


def concat_mla_absorb_q(a, b):
    a_last = a.shape[-1]
    b_last = b.shape[-1]
    d = a_last + b_last
    out = torch.empty(a.shape[:-1] + (d,), dtype=a.dtype, device=a.device)
    rows = (a.numel() // a_last) if a_last else (b.numel() // b_last)
    if rows == 0 or d == 0:
        return out

    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    if (a_last & (a_last - 1)) == 0 and (b_last & (b_last - 1)) == 0:
        r, w = _tier(rows)
        while r > 1 and (rows % r or r * d > cap(a.dtype)):
            r //= 2
        _c2_dense_two[(rows // r,)](
            out,
            a,
            b,
            A_LAST=a_last,
            B_LAST=b_last,
            D=d,
            ROWS=r,
            num_warps=w,
            num_stages=_DENSE_STAGES,
        )
        return out
    bb = max(triton.next_power_of_2(b_last), 1)
    _c2_rows[(-(-rows // _GEN_ROWS),)](
        out,
        a,
        b,
        rows,
        A_LAST=a_last,
        B_LAST=b_last,
        D=d,
        BA=a_last,
        BB=bb,
        ROWS=_GEN_ROWS,
        num_warps=4,
    )
    return out


__all__ = ["concat_mla_absorb_q"]
