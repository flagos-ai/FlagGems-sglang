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

# 昇腾专用。主机侧按探针 13233 的拆解压缩（昇腾上每次新分配 ≈ 25–30 µs、JIT 调度 ≈ 25–30 µs 都计入计时）：
#   - 两个输出合成一次分配（同一块内存切两个视图返回；两个输出元素都是 4 字节）
#   - 第一次走 JIT 编译，之后同一特化直接用编译好的句柄发射（只复用编译产物，不缓存任何结果，每次都在 kernel 里完整重算）
#   - 整型形状参数 do_not_specialize；主机侧只用内置整数运算，不调 data_ptr

_OLD_LAUNCH = tuple(int(v) for v in triton.__version__.split(".")[:2]) < (3, 3)   # <3.3 直接发射不带 constexpr


@triton.jit(do_not_specialize=["M", "N", "stride_xm"])
def _gate_topk_kernel(x_ptr, val_ptr, idx_ptr, M, N, stride_xm,
                      K: tl.constexpr, KP: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 每个 program 处理 BLOCK_M 行；k 次「取最大 + 掩掉」，平手取最小列下标，
    # 精确复现 torch.topk(sorted=True) + 「tie-break 取较小列下标」。
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    rows0 = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    kc = tl.arange(0, KP)
    num_blocks = tl.cdiv(M, BLOCK_M)
    for blk in range(pid, num_blocks, nprog):
        rows = blk * BLOCK_M + rows0
        rmask = rows < M
        live = rmask[:, None] & (cols[None, :] < N)
        x = tl.load(x_ptr + rows[:, None].to(tl.int64) * stride_xm + cols[None, :],
                    mask=live, other=0.0).to(tl.float32)
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
    bn = (1 << (N - 1).bit_length()) if N > 16 else 16
    kp = (1 << (k - 1).bit_length()) if k > 2 else 2
    bm = 4096 // bn
    bm = 32 if bm > 32 else (bm if bm > 0 else 1)
    if x.dtype == torch.float32:
        buf = x.new_empty((2, M, k), dtype=torch.int32)
        indices = buf[0]
        values = buf[1].view(torch.float32)
    else:
        values = x.new_empty((M, k))
        indices = x.new_empty((M, k), dtype=torch.int32)
    g = (M + bm - 1) // bm
    grid = (g if g < 32768 else 32768, 1, 1)
    args = (x, values, indices, M, N, x.stride(0))
    key = (k, bn, x.dtype)
    cached = getattr(_gate_topk_kernel, "_s2_handle", None)
    if cached is not None and cached[0] == key:
        if _OLD_LAUNCH:
            cached[1][grid](*args)
        else:
            cached[1][grid](*args, k, kp, bm, bn)
    else:
        handle = _gate_topk_kernel[grid](*args, K=k, KP=kp, BLOCK_M=bm, BLOCK_N=bn)
        _gate_topk_kernel._s2_handle = (key, handle)
    return values, indices
