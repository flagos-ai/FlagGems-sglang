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

import triton
import triton.language as tl

_INT32_MAX = 2**31 - 1
_INT32_MIN = -(2**31)

# tile 分档: hidden >= 4096 且 1792 整除 -> 16x1792; 1024 整除 -> 8x1024;
# 其它 -> 16x1024. 均为本机实测可编译且性能最优的组合.
_LARGE_H_UNIT = 1792
_SMALL_H_UNIT = 1024
_NUM_WARPS = 4
_NUM_STAGES = 1
_MAX_PLANS = 64


@triton.jit(
    do_not_specialize=["n_rows"],
    do_not_specialize_on_alignment=["input_ptr", "weight_ptr", "out_ptr"],
)
def _vocab_parallel_embedding_kernel(
    input_ptr,  # token id, 任意 int dtype, flat [n_rows]
    weight_ptr,  # [local_vocab, hidden], stride(1) == 1
    out_ptr,  # [n_rows, hidden]
    n_rows,
    ORG_START: tl.constexpr,
    ORG_END: tl.constexpr,
    ADDED_START: tl.constexpr,
    ADDED_END: tl.constexpr,
    ADDED_OFFSET: tl.constexpr,
    HIDDEN: tl.constexpr,
    W_STRIDE0: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN_T: tl.constexpr,
    EVEN_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    # 窗口边界是 constexpr: 范围检查折成单次比较. EVEN_T 时连行边界 mask
    # 一起去掉 (见文件头).
    if EVEN_T:
        tok = tl.load(input_ptr + rows).to(tl.int32)
    else:
        tok = tl.load(input_ptr + rows, mask=rows < n_rows, other=0).to(
            tl.int32
        )
    in_org = (tok >= ORG_START) & (tok < ORG_END)
    in_added = (tok >= ADDED_START) & (tok < ADDED_END)
    valid = in_org | in_added

    # 落窗外的行 (含 masked 行) 索引夹紧到 0 行, 读一个合法行后再 select
    # 成 0 —— load 上不挂 valid mask (见 v2 注释).
    local_id = tl.where(in_org, tok - ORG_START, tok - ADDED_OFFSET)
    local_id = tl.where(valid, local_id, 0)

    # 行内列地址连续 (cols 系数 1): 权重行段/输出行段均为连续向量访问.
    src = local_id[:, None] * W_STRIDE0 + cols[None, :]
    if EVEN_T and EVEN_H:
        w = tl.load(weight_ptr + src)
    else:
        w = tl.load(
            weight_ptr + src,
            mask=(rows < n_rows)[:, None] & (cols < HIDDEN)[None, :],
            other=0.0,
        )
    w = tl.where(valid[:, None], w, 0.0)

    dst = rows[:, None] * HIDDEN + cols[None, :]
    if EVEN_T and EVEN_H:
        tl.store(out_ptr + dst, w)
    else:
        tl.store(
            out_ptr + dst,
            w,
            mask=(rows < n_rows)[:, None] & (cols < HIDDEN)[None, :],
        )


def vocab_parallel_embedding(
    input_,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    hidden = weight.shape[1]
    n_rows = input_.numel()
    out = weight.new_empty(input_.shape + (hidden,))
    if n_rows == 0 or hidden == 0:
        return out

    org_start = int(org_vocab_start_index)
    org_end = int(org_vocab_end_index)
    added_start = int(added_vocab_start_index)
    added_end = int(added_vocab_end_index)
    padding = int(num_org_vocab_padding)
    added_offset = added_start - (org_end - org_start) - padding

    # tile 分档内联 (见文件头): hidden >= 4096 且 %1792 == 0 -> 16x1792;
    # %1024 == 0 -> 8x1024; 其它 -> 16x1024 兜底.
    if hidden >= 4096 and hidden % _LARGE_H_UNIT == 0:
        bt = 16
        bh = _LARGE_H_UNIT
    elif hidden % _SMALL_H_UNIT == 0:
        bt = 8
        bh = _SMALL_H_UNIT
    else:
        bt = 16
        bh = _SMALL_H_UNIT
    even_t = n_rows % bt == 0
    even_h = hidden % bh == 0

    grid = ((n_rows + bt - 1) // bt, (hidden + bh - 1) // bh)
    _vocab_parallel_embedding_kernel[grid](
        input_,
        weight,
        out,
        n_rows,
        ORG_START=org_start,
        ORG_END=org_end,
        ADDED_START=added_start,
        ADDED_END=added_end,
        ADDED_OFFSET=added_offset,
        HIDDEN=hidden,
        W_STRIDE0=weight.stride(0),
        BLOCK_T=bt,
        BLOCK_H=bh,
        EVEN_T=even_t,
        EVEN_H=even_h,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return out


__all__ = ["vocab_parallel_embedding"]
