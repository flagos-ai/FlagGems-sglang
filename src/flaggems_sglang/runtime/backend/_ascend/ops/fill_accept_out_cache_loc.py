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

"""fill_accept_out_cache_loc: 纯 Triton 实现 (v5: 标准 dispatch)."""


import triton
import triton.language as tl


@triton.jit
def _fill_accept_kernel(
    accept_index_ptr,
    out_cache_loc_ptr,
    out_ptr,
    size,
    PRI: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, PRI)
    ai = tl.load(accept_index_ptr + offs, mask=offs < pid * BLOCK, other=-1)
    prefix = tl.sum((ai.to(tl.int32) > -1).to(tl.int32), axis=0)

    # 阶段 2: 本 chunk 的 exclusive prefix sum + gather + scatter。
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    ai = tl.load(accept_index_ptr + offs, mask=mask, other=-1)
    keep = ai.to(tl.int32) > -1
    # gather: 被接受的 slot 从 out_cache_loc 取值; masked-out lane 不访存。
    val = tl.load(out_cache_loc_ptr + ai, mask=keep, other=0)
    keep32 = keep.to(tl.int32)
    # 块内 exclusive prefix + 全局前缀 -> 目的下标 (块间区间不重叠)。
    dst = prefix + tl.cumsum(keep32, axis=0) - keep32
    tl.store(out_ptr + dst.to(tl.int64), val, mask=keep)


def _pow2(n):
    """最小的 2 的幂 >= n。"""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def fill_accept_out_cache_loc(
    accept_index, out_cache_loc, accept_out_cache_loc, size
):
    size = int(size)
    # reference 的 accept_index[:size] 切片语义: size 超长时按实际长度截断。
    if size > accept_index.shape[0]:
        size = accept_index.shape[0]
    if size <= 0:
        return accept_out_cache_loc

    # PRI: 前缀扫描区域的 pad 宽度 (2 的幂, >= 任意 program 的前缀长度
    # (nprog-1)*BLOCK); 所有 program 共用同一个 constexpr, 只编译一份。
    span = _pow2(size)
    # (BLOCK, num_warps, num_stages) 按 size 分档, 每档已保证 BLOCK*nprog >= size。
    if size <= 8:
        block, warps, stages = 16, 1, 1
    elif size <= 64:
        block, warps, stages = 32, 1, 1
    elif size <= 512:
        block, warps, stages = 64, 1, 1
    else:
        nprog = (size + 127) // 128
        if nprog > 32:
            nprog = 32
        block = _pow2((size + nprog - 1) // nprog)
        nprog = (size + block - 1) // block
        warps, stages = 1, 2
    nprog = (size + block - 1) // block

    _fill_accept_kernel[(nprog,)](
        accept_index,
        out_cache_loc,
        accept_out_cache_loc,
        size,
        PRI=span,
        BLOCK=block,
        num_warps=warps,
        num_stages=stages,
    )
    return accept_out_cache_loc


__all__ = ["fill_accept_out_cache_loc"]
