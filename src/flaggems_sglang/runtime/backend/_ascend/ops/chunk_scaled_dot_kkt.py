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
def _chunk_scaled_dot_kkt_ascend_persist_kernel(
    k_ptr,
    beta_ptr,
    g_ptr,
    out_ptr,
    T,
    B,
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    NT,
    NUM_PROGRAMS: tl.constexpr,
):
    bt_stride = B * T
    pid = tl.program_id(0)
    n_tasks = NT * B * H
    for task in tl.range(pid, n_tasks, NUM_PROGRAMS, num_stages=1):
        i_t = task // (B * H)
        i_bh = task % (B * H)
        i_b = i_bh // H
        i_h = i_bh % H
        i_hg = i_h // (H // HG)
        bos = i_b * T
        o_t = tl.arange(0, BT)
        in_i = i_t * BT + o_t < T
        b_beta = tl.load(
            beta_ptr + i_h * bt_stride + bos + i_t * BT + o_t,
            mask=in_i,
            other=0.0,
        ).to(tl.float32)
        acc = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)
            in_k = offs_k < K
            b_k = tl.load(
                k_ptr
                + ((bos + i_t * BT + o_t[:, None]) * HG + i_hg) * K
                + offs_k[None, :],
                mask=in_i[:, None] & in_k[None, :],
                other=0.0,
            )
            acc = tl.dot(b_k, tl.trans(b_k), acc, out_dtype=tl.float32)
        if USE_G:
            b_g = tl.load(
                g_ptr + i_h * bt_stride + bos + i_t * BT + o_t,
                mask=in_i,
                other=0.0,
            ).to(tl.float32)
            g_diff = b_g[:, None] - b_g[None, :]
            acc *= tl.exp(tl.where(g_diff <= 0, g_diff, float("-inf")))
        acc *= b_beta[:, None]
        acc = tl.where(o_t[:, None] > o_t[None, :], acc, 0.0)
        tl.store(
            out_ptr
            + ((bos + i_t * BT + o_t[:, None]) * H + i_h) * BT
            + o_t[None, :],
            acc,
            mask=in_i[:, None],
        )


def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = int(chunk_size)
    NT = triton.cdiv(T, BT)
    out = torch.empty(B, T, H, BT, device=k.device, dtype=torch.float32)
    use_g = g_cumsum is not None
    beta_t = torch.permute(beta, (2, 0, 1)).contiguous()
    if use_g:
        g_ptr = torch.permute(g_cumsum, (2, 0, 1)).contiguous()
    else:
        g_ptr = beta_t
    num_programs = 40
    _chunk_scaled_dot_kkt_ascend_persist_kernel[num_programs,](
        k.contiguous(),
        beta_t,
        g_ptr,
        out,
        T,
        B,
        H,
        Hg,
        K,
        BT,
        128,
        use_g,
        NT,
        num_programs,
        num_warps=8,
        num_stages=1,
    )
    return out


__all__ = ["chunk_scaled_dot_kkt"]
