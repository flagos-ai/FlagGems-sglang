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


@triton.jit
def _chunk_scaled_dot_kkt_kernel(
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
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IEEE: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_b = i_bh // H
    i_h = i_bh % H
    i_hg = i_h // (H // HG)
    offs_i = tl.arange(0, BT)
    offs_j = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    t0 = i_t * BT
    acc = tl.zeros((BT, BT), dtype=tl.float32)
    k_base = k_ptr + i_b * STRIDE_K_B + i_hg * STRIDE_K_HG
    for k0 in range(0, K, BK):
        in_k = offs_k < K - k0
        k_tile = tl.load(
            k_base
            + (t0 + offs_i[:, None]) * STRIDE_K_T
            + (k0 + offs_k[None, :]) * STRIDE_K_K,
            mask=in_k[None, :],
            other=0.0,
        )
        kt_tile = tl.load(
            k_base
            + (k0 + offs_k[:, None]) * STRIDE_K_K
            + (t0 + offs_j[None, :]) * STRIDE_K_T,
            mask=in_k[:, None],
            other=0.0,
        )
        if IEEE:
            acc = tl.dot(
                k_tile,
                kt_tile,
                acc,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
        else:
            acc = tl.dot(k_tile, kt_tile, acc, out_dtype=tl.float32)
    beta = tl.load(
        beta_ptr
        + i_b * STRIDE_BETA_B
        + (t0 + offs_i) * STRIDE_BETA_T
        + i_h * STRIDE_BETA_H
    ).to(tl.float32)
    acc *= beta[:, None]
    if USE_G:
        g = tl.load(
            g_ptr
            + i_b * STRIDE_G_B
            + (t0 + offs_i) * STRIDE_G_T
            + i_h * STRIDE_G_H
        ).to(tl.float32)
        g_diff = g[:, None] - g[None, :]
        acc *= tl.where(g_diff <= 0, tl.exp(g_diff), 0.0)
    acc = tl.where(offs_i[:, None] > offs_j[None, :], acc, 0.0)
    tl.store(
        out_ptr
        + i_b * STRIDE_O_B
        + (t0 + offs_i[:, None]) * STRIDE_O_T
        + i_h * STRIDE_O_H
        + offs_j[None, :] * STRIDE_O_BT,
        acc,
    )


def _block_k(k_dim):
    if k_dim <= 16:
        return 16
    if k_dim <= 32:
        return 32
    if k_dim <= 64:
        return 64
    return 128


def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
    k = k.contiguous()
    beta = beta.contiguous()
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = int(chunk_size)
    NT = T // BT
    out = torch.empty(B, T, H, BT, device=k.device, dtype=torch.float32)
    use_g = g_cumsum is not None
    if use_g:
        g_cumsum = g_cumsum.contiguous()
        sg = g_cumsum.stride()
        g_ptr = g_cumsum
    else:
        sg = (0, 0, 0)
        g_ptr = k
    sk = k.stride()
    sb = beta.stride()
    so = out.stride()
    ieee = k.dtype == torch.float32
    bk = _block_k(K)
    _chunk_scaled_dot_kkt_kernel[NT, B * H](
        k,
        beta,
        g_ptr,
        out,
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
        K,
        BT,
        bk,
        use_g,
        ieee,
        num_warps=4,
        num_stages=3,
    )
    return out


__all__ = ["chunk_scaled_dot_kkt"]
