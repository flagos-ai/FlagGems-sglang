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
def _gqa_gstore_kernel(
    k_ptr,
    beta_ptr,
    g_ptr,
    out_ptr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_T: tl.constexpr,
    STRIDE_K_HG: tl.constexpr,
    STRIDE_K_K: tl.constexpr,
    STRIDE_BETA_H: tl.constexpr,
    STRIDE_BETA_B: tl.constexpr,
    STRIDE_BETA_T: tl.constexpr,
    STRIDE_G_H: tl.constexpr,
    STRIDE_G_B: tl.constexpr,
    STRIDE_G_T: tl.constexpr,
    STRIDE_O_B: tl.constexpr,
    STRIDE_O_T: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    STRIDE_O_BT: tl.constexpr,
    B,
    H: tl.constexpr,
    HG: tl.constexpr,
    RATIO: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IEEE: tl.constexpr,
    USE_TRANS: tl.constexpr,
):
    pid = tl.program_id(0)
    i_hg = pid % HG
    i_tb = pid // HG
    i_b = i_tb % B
    i_t = i_tb // B
    t0 = i_t * BT
    offs_i = tl.arange(0, BT)
    offs_j = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    o_g = tl.arange(0, RATIO)
    k_base = k_ptr + i_b * STRIDE_K_B + i_hg * STRIDE_K_HG
    gram = tl.zeros((BT, BT), dtype=tl.float32)
    for k0 in range(0, K, BK):
        in_k = offs_k < K - k0
        k_tile = tl.load(
            k_base
            + (t0 + offs_i[:, None]) * STRIDE_K_T
            + (k0 + offs_k[None, :]) * STRIDE_K_K,
            mask=in_k[None, :],
            other=0.0,
        )
        if USE_TRANS:
            kt_tile = tl.trans(k_tile)
        else:
            kt_tile = tl.load(
                k_base
                + (k0 + offs_k[:, None]) * STRIDE_K_K
                + (t0 + offs_j[None, :]) * STRIDE_K_T,
                mask=in_k[:, None],
                other=0.0,
            )
        if IEEE:
            gram = tl.dot(
                k_tile,
                kt_tile,
                gram,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
        else:
            gram = tl.dot(k_tile, kt_tile, gram, out_dtype=tl.float32)
    gram = tl.where(offs_i[:, None] > offs_j[None, :], gram, 0.0)
    h0 = i_hg * RATIO
    b_beta = tl.load(
        beta_ptr
        + (h0 + o_g)[None, :] * STRIDE_BETA_H
        + i_b * STRIDE_BETA_B
        + (t0 + offs_i)[:, None] * STRIDE_BETA_T
    ).to(tl.float32)
    a = gram[:, None, :] * b_beta[:, :, None]
    if USE_G:
        b_g = tl.load(
            g_ptr
            + (h0 + o_g)[None, :] * STRIDE_G_H
            + i_b * STRIDE_G_B
            + (t0 + offs_i)[:, None] * STRIDE_G_T
        ).to(tl.float32)
        g_diff = b_g[:, :, None] - tl.trans(b_g, (1, 0))[None, :, :]
        a = a * tl.where(g_diff <= 0, tl.exp(g_diff), 0.0)
    a_flat = tl.reshape(a, (BT, RATIO * BT))
    tl.store(
        out_ptr
        + i_b * STRIDE_O_B
        + (t0 + offs_i[:, None]) * STRIDE_O_T
        + h0 * STRIDE_O_H
        + tl.arange(0, RATIO * BT)[None, :] * STRIDE_O_BT,
        a_flat,
    )


def _block_k(k_dim):
    if k_dim <= 16:
        return 16
    if k_dim <= 32:
        return 32
    return 64


def _run(k, beta, g_cumsum, chunk_size, use_trans, warps, stages):
    k = k.contiguous()
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = int(chunk_size)
    NT = T // BT
    ratio = H // Hg
    out = torch.empty(B, T, H, BT, device=k.device, dtype=torch.float32)
    beta_c = beta.contiguous()
    use_g = g_cumsum is not None
    if use_g:
        g_c = g_cumsum.contiguous()
        sg = (g_c.stride(2), g_c.stride(0), g_c.stride(1))
        g_ptr = g_c
    else:
        sg = (0, 0, 0)
        g_ptr = k
    sb = (beta_c.stride(2), beta_c.stride(0), beta_c.stride(1))
    sk, so = (k.stride(), out.stride())
    ieee = k.dtype == torch.float32
    bk = _block_k(K)
    _gqa_gstore_kernel[NT * B * Hg,](
        k,
        beta_c,
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
        B,
        H,
        Hg,
        ratio,
        K,
        BT,
        bk,
        use_g,
        ieee,
        use_trans,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
    return _run(
        k, beta, g_cumsum, chunk_size, use_trans=True, warps=4, stages=1
    )


__all__ = ["chunk_scaled_dot_kkt"]
