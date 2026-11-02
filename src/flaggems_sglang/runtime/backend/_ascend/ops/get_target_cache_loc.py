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

import torch
import triton
import triton.language as tl


@triton.jit
def _get_target_cache_loc_kernel(
    correct_ptr,
    free_n_ptr,
    out_ptr,
    tgt_ptr,
    freed_ptr,
    num_verify_tokens,
    BS: tl.constexpr,
    BLOCK: tl.constexpr,
    KBLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    """一个 program 顺序处理 ROWS 行 (行号 pid*ROWS .. pid*ROWS+ROWS-1):"""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    nvt = num_verify_tokens.to(tl.int32)
    k = tl.arange(0, KBLOCK)
    tail = BS * num_verify_tokens

    for r in tl.static_range(ROWS):
        b = pid * ROWS + r
        if b < BS:
            # ---- 本行的两个起始偏移 (exclusive prefix sums, masked load
            # 避免 tl.where) ----
            pre = offs < b  # 只累加严格在 b 之前的行
            c_pre = tl.load(correct_ptr + offs, mask=pre, other=0).to(tl.int32)
            s_pre = tl.load(free_n_ptr + offs, mask=pre, other=0).to(tl.int32)
            tgt_base = tl.sum(c_pre, axis=0) + b
            s_row = tl.sum(s_pre, axis=0)

            row_base = b * nvt

            # ---- tgt: out_cache_loc[b, :keep] -> tgt[tgt_base :
            # tgt_base + keep] ----
            keep = tl.load(correct_ptr + b).to(tl.int32) + 1
            kt = k < keep
            vt = tl.load(out_ptr + row_base + k, mask=kt, other=0)
            tl.store(tgt_ptr + tgt_base + k, vt, mask=kt)

            # ---- free: out_cache_loc[b, nvt - f : nvt] ->
            # free[free_base : +f] ----
            f = tl.load(free_n_ptr + b).to(tl.int32)
            kf = k < f
            vf = tl.load(out_ptr + row_base + (nvt - f) + k, mask=kf, other=0)
            tl.store(freed_ptr + s_row + k, vf, mask=kf)

            if (pid == 0) and (r == 0):
                row_mask = offs < BS
                c = tl.load(correct_ptr + offs, mask=row_mask, other=0).to(
                    tl.int32
                )
                s = tl.load(free_n_ptr + offs, mask=row_mask, other=0).to(
                    tl.int32
                )
                c_tot = tl.sum(c, axis=0).to(tl.int64)
                s_tot = tl.sum(s, axis=0).to(tl.int64)
                tl.store(freed_ptr + tail + 0, c_tot)
                tl.store(freed_ptr + tail + 1, s_tot)
                tl.store(freed_ptr + s_tot, 0)


def _pick_launch(bs):
    """``(ROWS, num_warps)`` — 直连 launch 下重新标定的最优配置."""
    if bs >= 1024:
        return 4, 1
    return 2, 2


def get_target_cache_loc(
    num_correct_drafts, to_free_num_slots, out_cache_loc, num_verify_tokens
):
    bs = num_correct_drafts.shape[0]
    device = out_cache_loc.device
    nvt = int(num_verify_tokens)

    if bs == 0:
        return (
            torch.zeros(1, dtype=out_cache_loc.dtype, device=device),
            torch.zeros(1, dtype=out_cache_loc.dtype, device=device),
        )

    # ---- worst-case 缓冲: n_tgt <= bs*nvt, n_free <= bs*nvt, 尾部留 2 槽放 sums ----
    BLOCK = triton.next_power_of_2(bs)
    KBLOCK = triton.next_power_of_2(nvt)
    rows, num_warps = _pick_launch(bs)
    buf = torch.empty(
        2 * bs * nvt + 2, dtype=out_cache_loc.dtype, device=device
    )
    tgt_max = buf[: bs * nvt]
    free_max = buf[bs * nvt : 2 * bs * nvt + 2]

    # ---- 单次标准 dispatch: 大 bs 用多行 program 削减 exec ----
    grid = (triton.cdiv(bs, rows),)
    _get_target_cache_loc_kernel[grid](
        num_correct_drafts,
        to_free_num_slots,
        out_cache_loc,
        tgt_max,
        free_max,
        nvt,
        BS=bs,
        BLOCK=BLOCK,
        KBLOCK=KBLOCK,
        ROWS=rows,
        num_warps=num_warps,
        num_stages=1,
    )

    # ---- 一次 D2H 同步取回两个输出长度, narrow 出精确尺寸 ----
    c_tot, s_tot = free_max[bs * nvt : bs * nvt + 2].tolist()
    tgt = buf[: c_tot + bs]
    free = buf[bs * nvt : bs * nvt + (s_tot if s_tot > 0 else 1)]
    return tgt, free


__all__ = ["get_target_cache_loc"]
