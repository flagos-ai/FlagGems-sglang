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

# 每选一个元素只做一次归约：tl.max(return_indices=True, tie_break_left=True) 同时给出最大值和最小列下标
# （平手取小下标，和题面一致），然后把选中的列置 -inf。输出逐列直接写出，不维护 [BLOCK_M, KP] 累加块，
# k=32 时寄存器压力小得多。


@triton.jit
def _gate_topk_kernel(x_ptr, val_ptr, idx_ptr, M, N, stride_xm,
                      K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < M
    inrow = rmask[:, None] & (cols[None, :] < N)
    x = tl.load(x_ptr + rows[:, None].to(tl.int64) * stride_xm + cols[None, :],
                mask=inrow, other=float("-inf")).to(tl.float32)
    obase = rows.to(tl.int64) * K
    for i in tl.static_range(K):
        cur, sel = tl.max(x, axis=1, return_indices=True, return_indices_tie_break_left=True)
        tl.store(val_ptr + obase + i, cur.to(val_ptr.dtype.element_ty), mask=rmask)
        tl.store(idx_ptr + obase + i, sel.to(tl.int32), mask=rmask)
        x = tl.where(cols[None, :] == sel[:, None], float("-inf"), x)


def _config(N, k, BLOCK_N):
    # 平台计时探针 13187（天数 / 沐曦 / 海光）和本机 4070：一行一个 program、一个 warp 在所有用例上最好
    return 1, 1


def gate_topk(x, k):
    M, N = x.shape
    values = torch.empty((M, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=x.device)
    BLOCK_N = max(16, triton.next_power_of_2(N))
    BLOCK_M, num_warps = _config(N, k, BLOCK_N)
    grid = (triton.cdiv(M, BLOCK_M),)
    _gate_topk_kernel[grid](x, values, indices, M, N, x.stride(0),
                            K=k, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps)
    return values, indices
