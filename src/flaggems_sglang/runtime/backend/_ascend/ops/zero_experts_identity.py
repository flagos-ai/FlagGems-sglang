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

"""zero_experts_identity 的纯 Triton 实现 (v5)."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_tokens", "top_k", "num_experts"])
def _zei_kernel(
    idx_ptr,
    scale_ptr,
    h_ptr,
    o_ptr,
    num_tokens,
    top_k,
    num_experts,
    hidden,
    BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    ROWS: tl.constexpr,
    EVEN_ROWS: tl.constexpr,
    EVEN_COLS: tl.constexpr,
):
    """(行块, 列块) 二维 grid: 每 program 一次 load/store 整个 (ROWS, BLOCK) tile."""
    row0 = tl.program_id(0) * ROWS
    rows = row0 + tl.arange(0, ROWS)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    koffs = tl.arange(0, TOPK)

    if EVEN_ROWS:
        # 归约本行块各 token 路由到 zero expert 的 scale 总和 (fp32, 形状 (ROWS,)).
        idx = tl.load(idx_ptr + rows[:, None] * top_k + koffs[None, :])
        sc = tl.load(scale_ptr + rows[:, None] * top_k + koffs[None, :])
        total = tl.sum(
            tl.where(idx >= num_experts, sc.to(tl.float32), 0.0), axis=1
        )
        if EVEN_COLS:
            h = tl.load(h_ptr + rows[:, None] * hidden + cols[None, :])
            out = h.to(tl.float32) * total[:, None]
            tl.store(
                o_ptr + rows[:, None] * hidden + cols[None, :],
                out.to(o_ptr.dtype.element_ty),
            )
        else:
            cm = cols[None, :] < hidden
            h = tl.load(
                h_ptr + rows[:, None] * hidden + cols[None, :],
                mask=cm,
                other=0.0,
            )
            out = h.to(tl.float32) * total[:, None]
            tl.store(
                o_ptr + rows[:, None] * hidden + cols[None, :],
                out.to(o_ptr.dtype.element_ty),
                mask=cm,
            )
    else:
        km = (rows < num_tokens)[:, None] & (koffs[None, :] < top_k)
        idx = tl.load(
            idx_ptr + rows[:, None] * top_k + koffs[None, :], mask=km, other=0
        )
        sc = tl.load(
            scale_ptr + rows[:, None] * top_k + koffs[None, :],
            mask=km,
            other=0.0,
        )
        total = tl.sum(
            tl.where(idx >= num_experts, sc.to(tl.float32), 0.0), axis=1
        )
        m2 = (rows < num_tokens)[:, None]
        if EVEN_COLS:
            h = tl.load(
                h_ptr + rows[:, None] * hidden + cols[None, :],
                mask=m2,
                other=0.0,
            )
            out = h.to(tl.float32) * total[:, None]
            tl.store(
                o_ptr + rows[:, None] * hidden + cols[None, :],
                out.to(o_ptr.dtype.element_ty),
                mask=m2,
            )
        else:
            cm = m2 & (cols[None, :] < hidden)
            h = tl.load(
                h_ptr + rows[:, None] * hidden + cols[None, :],
                mask=cm,
                other=0.0,
            )
            out = h.to(tl.float32) * total[:, None]
            tl.store(
                o_ptr + rows[:, None] * hidden + cols[None, :],
                out.to(o_ptr.dtype.element_ty),
                mask=cm,
            )


# hidden 超过此值时整行 tile 过大, 回退到分块 pow2 路径.
_WHOLE_ROW_MAX = 8192


# (ROWS, BLOCK, num_warps, num_stages) 按 (num_tokens, hidden) 分档;
# 本机 (ascend 910) direct-launch 下 do_bench(median) 逐档实测标定.
def _pick_config(num_tokens, hidden):
    if hidden <= _WHOLE_ROW_MAX:
        # 列方向一个 program 覆盖整行: 无列切块开销, 无列 mask.
        if num_tokens < 4:
            rows = 1
        elif num_tokens < 32:
            rows = 2
        elif num_tokens < 256:
            rows = 4
        elif num_tokens < 1024:
            rows = 2
        else:
            rows = 4
        return rows, hidden, 8, 1
    # 大 hidden: 分块 pow2 (BLOCK 2 的幂), 列方向带 mask.
    if num_tokens < 64:
        return 1, 4096, 8, 1
    if num_tokens < 256:
        return 16, 1024, 8, 1
    if num_tokens < 1024:
        return 4, 2048, 8, 2
    return 4, 4096, 8, 2


def zero_experts_identity(
    expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states
):
    num_tokens, top_k = expert_indices.shape
    hidden = hidden_states.shape[1]

    out = torch.empty_like(hidden_states)
    if num_tokens == 0 or hidden == 0:
        return out

    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not expert_indices.is_contiguous():
        expert_indices = expert_indices.contiguous()
    if not expert_scales.is_contiguous():
        expert_scales = expert_scales.contiguous()

    rows, block, warps, stages = _pick_config(num_tokens, hidden)
    topk = triton.next_power_of_2(top_k)
    grid = (triton.cdiv(num_tokens, rows), triton.cdiv(hidden, block))
    _zei_kernel[grid](
        expert_indices,
        expert_scales,
        hidden_states,
        out,
        num_tokens,
        top_k,
        num_experts,
        hidden,
        BLOCK=block,
        TOPK=topk,
        ROWS=rows,
        EVEN_ROWS=num_tokens % rows == 0,
        EVEN_COLS=hidden % block == 0,
        num_warps=warps,
        num_stages=stages,
    )
    return out


__all__ = ["zero_experts_identity"]
