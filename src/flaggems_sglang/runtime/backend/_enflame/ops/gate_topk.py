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

# 燧原专用。v1 在燧原上只挂了 case 2（M=512 N=64 k=1 fp32）：编译期 make_gcuir 的 PassManager 失败。
# 这个用例独有的是 k=1（输出块只有 KP=2 列）和 N=64（行块 64 列）。所以这里：
#   - k=1 走一维路径（就是 argmax，平手取最小下标），不构造 [BLOCK_M, KP] 输出块
#   - 其余情况输出块至少 16 列、行块至少 128 列，形状贴近已经通过的用例


@triton.jit
def _gate_topk_kernel(x_ptr, val_ptr, idx_ptr, M, N, stride_xm,
                      K: tl.constexpr, KP: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    rows0 = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    num_blocks = tl.cdiv(M, BLOCK_M)
    for blk in range(pid, num_blocks, nprog):
        rows = blk * BLOCK_M + rows0
        rmask = rows < M
        live = rmask[:, None] & (cols[None, :] < N)
        x = tl.load(x_ptr + rows[:, None].to(tl.int64) * stride_xm + cols[None, :],
                    mask=live, other=0.0).to(tl.float32)
        if K == 1:
            cur = tl.max(tl.where(live, x, float("-inf")), axis=1)
            sel = tl.min(tl.where(live & (x == cur[:, None]), cols[None, :], BLOCK_N), axis=1)
            tl.store(val_ptr + rows, cur.to(val_ptr.dtype.element_ty), mask=rmask)
            tl.store(idx_ptr + rows, sel, mask=rmask)
        else:
            kc = tl.arange(0, KP)
            out_v = tl.zeros((BLOCK_M, KP), dtype=tl.float32)
            out_i = tl.zeros((BLOCK_M, KP), dtype=tl.int32)
            for i in tl.static_range(K):
                cur = tl.max(tl.where(live, x, float("-inf")), axis=1)
                cand = tl.where(live & (x == cur[:, None]), cols[None, :], BLOCK_N)
                sel = tl.min(cand, axis=1)
                live = live & (cols[None, :] != sel[:, None])
                out_v = tl.where(kc[None, :] == i, cur[:, None], out_v)
                out_i = tl.where(kc[None, :] == i, sel[:, None], out_i)
            omask = rmask[:, None] & (kc[None, :] < K)
            off = rows[:, None].to(tl.int64) * K + kc[None, :]
            tl.store(val_ptr + off, out_v.to(val_ptr.dtype.element_ty), mask=omask)
            tl.store(idx_ptr + off, out_i, mask=omask)


def gate_topk(x, k):
    M, N = x.shape
    values = torch.empty((M, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=x.device)
    BLOCK_N = max(128, triton.next_power_of_2(N))
    KP = max(16, triton.next_power_of_2(k))
    BLOCK_M = max(1, min(32, 4096 // BLOCK_N))
    grid = (max(1, min(triton.cdiv(M, BLOCK_M), 32768)),)
    _gate_topk_kernel[grid](x, values, indices, M, N, x.stride(0),
                            K=k, KP=KP, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return values, indices
