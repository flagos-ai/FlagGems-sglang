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

"""dcp_lse_combine: 合并 N 个 decode-context-parallel rank 的部分注意力输出."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dcp_lse_combine_kernel(
    out_ptr,  # [rows*D] 合并输出
    lse_out_ptr,  # [rows] 合并 LSE (RETURN_LSE=False 时为占位, 不会写入)
    output_ptr,  # [N, rows*D] 各 rank 部分输出
    lse_ptr,  # [N, rows] 各 rank LSE (fp32)
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    ROWS: tl.constexpr,  # b*h (行数), 兼作 recv_lse 的 rank 维 stride
    G: tl.constexpr,  # 每行元素数 D, 兼作 output 的行内 stride
    W: tl.constexpr,  # num_warps (位置 constexpr 传入, 省 kwargs 开销)
    N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_POW2: tl.constexpr,  # 0 表示 D == BLOCK_D (无 D-mask 快速路径), 否则 = D
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid * R
    offs_r = row0 + tl.arange(0, R)
    mask_r = offs_r < ROWS
    offs_d = tl.arange(0, BLOCK_D)
    base = offs_r[:, None] * G + offs_d[None, :]
    if D_POW2 == 0:
        mask_rd = mask_r[:, None] & tl.full([1, BLOCK_D], True, tl.int1)
    else:
        mask_rd = mask_r[:, None] & (offs_d < G)[None, :]

    # ---- pass 1: m = max_n lse[n], sanitize: NaN / +inf 视为无贡献 ----
    m = tl.full([R], float("-inf"), tl.float32)
    for k in tl.static_range(N):
        lse_k = tl.load(
            lse_ptr + k * ROWS + offs_r, mask=mask_r, other=float("-inf")
        )
        lse_k = tl.where(
            (lse_k != lse_k) | (lse_k == float("inf")), float("-inf"), lse_k
        )
        m = tl.maximum(m, lse_k)
    # 全 -inf 时强制 m = 0 (权重保持有限, 与 reference 一致)
    m = tl.where(m == float("-inf"), 0.0, m)

    # ---- pass 2: 逐 rank 载入 [R, BLOCK_D] span, fp32 就地加权累加 ----
    wsum = tl.zeros([R], tl.float32)
    acc = tl.zeros([R, BLOCK_D], tl.float32)
    for k in tl.static_range(N):
        lse_k = tl.load(
            lse_ptr + k * ROWS + offs_r, mask=mask_r, other=float("-inf")
        )
        lse_k = tl.where(
            (lse_k != lse_k) | (lse_k == float("inf")), float("-inf"), lse_k
        )
        centered = lse_k - m  # lse_k=-inf 时 centered=-inf, exp/exp2=0, 无贡献
        if IS_BASE_E:
            w = tl.exp(centered)
        else:
            w = tl.exp2(centered)
        wsum += w
        tile = tl.load(
            output_ptr + k * (ROWS * G) + base, mask=mask_rd, other=0.0
        ).to(tl.float32)
        acc += tile * w[:, None]

    acc = acc / wsum[:, None]

    tl.store(
        out_ptr + base,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_rd,
    )

    if RETURN_LSE:
        if IS_BASE_E:
            merged = tl.log(wsum) + m
        else:
            merged = tl.log2(wsum) + m
        tl.store(
            lse_out_ptr + offs_r,
            merged.to(lse_out_ptr.dtype.element_ty),
            mask=mask_r,
        )


def _next_pow2(x):
    """triton.next_power_of_2 的内联等价 (省 ~5us/次 的纯 Python 包装开销)."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def dcp_lse_combine(recv_output, recv_lse, is_lse_base_on_e, return_lse):
    n, b, h, d = recv_output.shape
    if not recv_output.is_contiguous():
        recv_output = recv_output.contiguous()
    if not recv_lse.is_contiguous():
        recv_lse = recv_lse.contiguous()

    rows = b * h
    out = torch.empty(
        (b, h, d), dtype=recv_output.dtype, device=recv_output.device
    )
    # RETURN_LSE=False 时用 recv_lse 占位, kernel 内对应分支为 constexpr 死代码.
    lse_out = (
        torch.empty((b, h), dtype=recv_lse.dtype, device=recv_lse.device)
        if return_lse
        else recv_lse
    )

    block_d = _next_pow2(d)
    # 启发式 (910B 实测对齐): r 上限 16384/block_d (block_d=128 时到 128),
    # num_warps 按 tile 大小 8/4/2 分档.
    r = 16384 // block_d
    if r > 128:
        r = 128
    if r < 1:
        r = 1
    r = _next_pow2(r)
    pr = _next_pow2(rows)
    if r > pr:
        r = pr
    tile = r * block_d
    if tile >= 8192:
        num_warps = 8
    elif tile >= 2048:
        num_warps = 4
    else:
        num_warps = 2

    grid0 = (rows + r - 1) // r
    d_pow2 = 0 if block_d == d else d
    _dcp_lse_combine_kernel[(grid0,)](
        out,
        lse_out,
        recv_output,
        recv_lse,
        is_lse_base_on_e,
        return_lse,
        rows,
        d,
        num_warps,
        n,
        block_d,
        d_pow2,
        r,
    )
    return out, (lse_out if return_lse else None)


__all__ = ["dcp_lse_combine"]
