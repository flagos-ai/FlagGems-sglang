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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32
_GROUP_M = 8


@triton.jit
def _chunk_scaled_dot_kkt_xpu_kernel(
    k_ptr,
    beta_ptr,
    g_ptr,
    out_ptr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_T: tl.constexpr,
    STRIDE_K_HG: tl.constexpr,
    STRIDE_K_K: tl.constexpr,
    STRIDE_BETA_B: tl.constexpr,
    STRIDE_BETA_T: tl.constexpr,
    STRIDE_BETA_H: tl.constexpr,
    STRIDE_G_B: tl.constexpr,
    STRIDE_G_T: tl.constexpr,
    STRIDE_G_H: tl.constexpr,
    STRIDE_O_B: tl.constexpr,
    STRIDE_O_T: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    STRIDE_O_BT: tl.constexpr,
    H: tl.constexpr,
    HG: tl.constexpr,
    NT: tl.constexpr,
    BT: tl.constexpr,
    K: tl.constexpr,
    USE_G: tl.constexpr,
    IEEE: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    grid_m = tl.cdiv(BT, BLOCK_M)
    grid_n = tl.cdiv(BT, BLOCK_N)
    tiles_per_mat = grid_m * grid_n
    program_id = tl.program_id(0)
    mat_id = program_id // tiles_per_mat
    tile_id = program_id % tiles_per_mat
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + tile_id % group_size
    pid_n = tile_id % width // group_size
    i_h = mat_id % H
    i_t = mat_id // H % NT
    i_b = mat_id // (H * NT)
    i_hg = i_h // (H // HG)
    t0 = i_t * BT
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_base = k_ptr + i_b * STRIDE_K_B + i_hg * STRIDE_K_HG
    a_ptrs = (
        k_base
        + (t0 + offs_m[:, None]) * STRIDE_K_T
        + offs_k[None, :] * STRIDE_K_K
    )
    b_ptrs = (
        k_base
        + offs_k[:, None] * STRIDE_K_K
        + (t0 + offs_n[None, :]) * STRIDE_K_T
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_ok = offs_k < K - k_block * BLOCK_K
            a = tl.load(a_ptrs, mask=k_ok[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_ok[:, None], other=0.0)
        if IEEE:
            acc = tl.dot(
                a, b, acc, input_precision="ieee", out_dtype=tl.float32
            )
        else:
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * STRIDE_K_K
        b_ptrs += BLOCK_K * STRIDE_K_K
    beta = tl.load(
        beta_ptr
        + i_b * STRIDE_BETA_B
        + (t0 + offs_m) * STRIDE_BETA_T
        + i_h * STRIDE_BETA_H
    ).to(tl.float32)
    acc *= beta[:, None]
    if USE_G:
        g_i = tl.load(
            g_ptr
            + i_b * STRIDE_G_B
            + (t0 + offs_m) * STRIDE_G_T
            + i_h * STRIDE_G_H
        ).to(tl.float32)
        g_j = tl.load(
            g_ptr
            + i_b * STRIDE_G_B
            + (t0 + offs_n) * STRIDE_G_T
            + i_h * STRIDE_G_H
        ).to(tl.float32)
        g_diff = g_i[:, None] - g_j[None, :]
        acc *= tl.where(g_diff <= 0, tl.exp(g_diff), 0.0)
    acc = tl.where(offs_m[:, None] > offs_n[None, :], acc, 0.0)
    tl.store(
        out_ptr
        + i_b * STRIDE_O_B
        + (t0 + offs_m[:, None]) * STRIDE_O_T
        + i_h * STRIDE_O_H
        + offs_n[None, :] * STRIDE_O_BT,
        acc,
    )


def _pad_chunks(x, chunk_size, block):
    B, T = (x.shape[0], x.shape[1])
    NT = T // chunk_size
    tail = x.shape[2:]
    viewed = x.reshape(B, NT, chunk_size, *tail)
    if chunk_size == block:
        return x
    padded = x.new_zeros((B, NT, block) + tail)
    padded[:, :, :chunk_size] = viewed
    return padded.reshape((B, NT * block) + tail)


def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
    k = k.contiguous()
    beta = beta.contiguous()
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = int(chunk_size)
    NT = T // BT
    block = _BLOCK_M
    bt_pad = -(-BT // block) * block
    k_w = _pad_chunks(k, BT, bt_pad)
    beta_w = _pad_chunks(beta, BT, bt_pad)
    use_g = g_cumsum is not None
    if use_g:
        g_w = _pad_chunks(g_cumsum.contiguous(), BT, bt_pad)
        sg = g_w.stride()
        g_ptr = g_w
    else:
        sg = (0, 0, 0)
        g_ptr = k_w
    T_pad = NT * bt_pad
    out_w = torch.empty(
        B, T_pad, H, bt_pad, device=k.device, dtype=torch.float32
    )
    sk = k_w.stride()
    sb = beta_w.stride()
    so = out_w.stride()
    ieee = k.dtype == torch.float32
    block_k = _BLOCK_K if K % _BLOCK_K == 0 else 16
    num_mats = B * NT * H
    grid = (num_mats * -(-bt_pad // _BLOCK_M) * -(-bt_pad // _BLOCK_N),)
    _chunk_scaled_dot_kkt_xpu_kernel[grid](
        k_w,
        beta_w,
        g_ptr,
        out_w,
        sk[0],
        sk[1],
        sk[2],
        sk[3],
        sb[0],
        sb[1],
        sb[2],
        sg[0],
        sg[1],
        sg[2],
        so[0],
        so[1],
        so[2],
        so[3],
        H,
        Hg,
        NT,
        bt_pad,
        K,
        use_g,
        ieee,
        K % block_k == 0,
        _BLOCK_M,
        _BLOCK_N,
        block_k,
        _GROUP_M,
        num_warps=4,
        num_stages=4,
    )
    if BT == bt_pad:
        return out_w
    return (
        out_w.view(B, NT, bt_pad, H, bt_pad)[:, :, :BT, :, :BT]
        .reshape(B, T, H, BT)
        .contiguous()
    )


__all__ = ["chunk_scaled_dot_kkt"]
