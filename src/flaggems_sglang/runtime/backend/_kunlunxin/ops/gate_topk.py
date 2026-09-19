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

# 昆仑专用。我们之前的昆仑版（v1 二维 tile / v8 一维但一个 program 扫 256–512 个元素）交了 6 次，昆仑一次都没出分
# （「服务线程卡死」或一直 waiting_callback），而同期 deepep 的昆仑每次都正常出分。
# 这里照搬官方仓库 PR #64（sitraliqui，sigmoid_gate_topk_renorm 的昆仑实现，该题 8 芯全过）的写法：
#   - 每个 program 只处理 ≤128 个元素的一段（grid = (M, 段数)），段内 K 次「取最大 + 平手取最小下标」
#   - 结果先攒在 [BLOCK_K] 寄存器向量里，最后一次性向量写出；N、K 都是 constexpr；有限的大负数当掩码值
#   - 第二个 kernel 每行把各段候选再做一次同样的选择（平手按原始列下标取小）
# 全部 1-D，无 int64，无 2D load。


@triton.jit
def _tile_topk_kernel(x_ptr, cval_ptr, cidx_ptr, stride_xm,
                      N: tl.constexpr, K: tl.constexpr, NUM_TILES: tl.constexpr,
                      BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    neg_large = -3.4028234663852886e38
    offs_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    vals = tl.load(x_ptr + row * stride_xm + offs_n, mask=mask_n, other=neg_large).to(tl.float32)
    vals = tl.where(mask_n, vals, neg_large)
    offs_k = tl.arange(0, BLOCK_K)
    top_v = tl.full((BLOCK_K,), neg_large, tl.float32)
    top_i = tl.full((BLOCK_K,), N, tl.int32)
    for slot in tl.static_range(0, K):
        best = tl.max(vals, axis=0)
        cand = tl.where((vals == best) & mask_n, offs_n, N)
        bidx = tl.min(cand, axis=0)
        top_v = tl.where(offs_k == slot, best, top_v)
        top_i = tl.where(offs_k == slot, bidx.to(tl.int32), top_i)
        vals = tl.where(offs_n == bidx, neg_large, vals)
    base = (row * NUM_TILES + tile) * K + offs_k
    tl.store(cval_ptr + base, top_v, mask=offs_k < K)
    tl.store(cidx_ptr + base, top_i, mask=offs_k < K)


@triton.jit
def _merge_topk_kernel(cval_ptr, cidx_ptr, val_ptr, idx_ptr,
                       N: tl.constexpr, K: tl.constexpr, NUM_TILES: tl.constexpr,
                       BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    neg_large = -3.4028234663852886e38
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < NUM_TILES * K
    vals = tl.load(cval_ptr + row * (NUM_TILES * K) + offs_c, mask=mask_c, other=neg_large)
    idxs = tl.load(cidx_ptr + row * (NUM_TILES * K) + offs_c, mask=mask_c, other=N)
    vals = tl.where(mask_c & (idxs < N), vals, neg_large)
    offs_k = tl.arange(0, BLOCK_K)
    top_v = tl.full((BLOCK_K,), neg_large, tl.float32)
    top_i = tl.full((BLOCK_K,), N, tl.int32)
    for slot in tl.static_range(0, K):
        best = tl.max(vals, axis=0)
        bidx = tl.min(tl.where((vals == best) & mask_c, idxs, N), axis=0)
        top_v = tl.where(offs_k == slot, best, top_v)
        top_i = tl.where(offs_k == slot, bidx, top_i)
        vals = tl.where(idxs == bidx, neg_large, vals)
    tl.store(val_ptr + row * K + offs_k, top_v.to(val_ptr.dtype.element_ty), mask=offs_k < K)
    tl.store(idx_ptr + row * K + offs_k, top_i, mask=offs_k < K)


def gate_topk(x, k):
    M, N = x.shape
    values = torch.empty((M, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((M, k), dtype=torch.int32, device=x.device)
    block_n = min(128, max(16, triton.next_power_of_2(N)))
    block_k = max(2, triton.next_power_of_2(k))
    num_tiles = triton.cdiv(N, block_n)
    cval = torch.empty((M, num_tiles, k), dtype=torch.float32, device=x.device)
    cidx = torch.empty((M, num_tiles, k), dtype=torch.int32, device=x.device)
    _tile_topk_kernel[(M, num_tiles)](x, cval, cidx, x.stride(0), N=N, K=k, NUM_TILES=num_tiles,
                                      BLOCK_N=block_n, BLOCK_K=block_k, num_warps=1, num_stages=1)
    block_c = max(16, triton.next_power_of_2(num_tiles * k))
    _merge_topk_kernel[(M,)](cval, cidx, values, indices, N=N, K=k, NUM_TILES=num_tiles,
                             BLOCK_C=block_c, BLOCK_K=block_k, num_warps=1, num_stages=1)
    return values, indices
