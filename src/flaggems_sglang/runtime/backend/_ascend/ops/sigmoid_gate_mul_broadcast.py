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
def _sgmb_even(
    x_ptr,
    g_ptr,
    o_ptr,
    CPR: tl.constexpr,
    BLOCK: tl.constexpr,
    U: tl.constexpr,
):
    """无 mask 版本: 要求 n % (BLOCK*U) == 0 且 BLOCK 整除 D (chunk 不跨行).

    CPR = D // BLOCK (chunks per row). 每个 chunk 恰好落在一行内, 行号是
    标量, gate 做 1 次标量 load + 标量 sigmoid, 再广播到整个 chunk.
    """
    pid = tl.program_id(0)
    ty = o_ptr.dtype.element_ty
    col = tl.arange(0, BLOCK)
    for i in tl.static_range(U):
        chunk = pid * U + i
        row = chunk // CPR
        g = tl.sigmoid(tl.load(g_ptr + row).to(tl.float32))
        offs = chunk * BLOCK + col
        v = tl.load(x_ptr + offs).to(tl.float32)
        tl.store(o_ptr + offs, (v * g).to(ty))


@triton.jit(do_not_specialize=["n", "d"])
def _sgmb_masked(x_ptr, g_ptr, o_ptr, n, d, BLOCK: tl.constexpr):
    """通用尾掩码版本: 任意 n / d (含 chunk 跨行, 行号逐元素计算)."""
    pid = tl.program_id(0)
    ty = o_ptr.dtype.element_ty
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    row = offs // d
    g = tl.load(g_ptr + row, mask=m, other=0.0)
    v = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(o_ptr + offs, (v * tl.sigmoid(g.to(tl.float32))).to(ty), mask=m)


def sigmoid_gate_mul_broadcast(x, gate):
    # 瘦 host 路径: 避免 reshape(-1) / triton.next_power_of_2 / triton.cdiv
    # 等 Python 开销 (本机 host CPU 上每个 2.5~5us, 小 shape 时与 kernel
    # 时间同量级). 连续 gate ([N,1] 元素 i 在偏移 i) 直接传原张量.
    if not x.is_contiguous():
        x = x.contiguous()
    if not gate.is_contiguous():
        gate = gate.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    d = x.shape[-1]

    if n <= 8192:
        epp = 1 << (n - 1).bit_length()
        if epp > 2048:
            epp = 2048
        w = 1
    elif n < 262144:
        epp, w = 4096, 2
    elif n < 2097152:
        epp, w = 16384, 8
    else:
        epp, w = 32768, 8

    # 不超过 epp 且能整除 d 的最大 2 的幂 (>=128), 无则走 masked; 与 v4 的
    # _block_for 等价但用内联移位代替函数调用 + while.
    block = epp
    while block > 128 and d % block:
        block >>= 1
    if block > 128 or d % block == 0:
        if n % epp == 0:
            _sgmb_even[(n // epp,)](
                x,
                gate,
                out,
                CPR=d // block,
                BLOCK=block,
                U=epp // block,
                num_warps=w,
            )
            return out
    _sgmb_masked[((n + 2047) >> 11,)](
        x, gate, out, n, d, BLOCK=2048, num_warps=2
    )
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]
