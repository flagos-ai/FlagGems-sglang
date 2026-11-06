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
def _fill_zero_output(OUT, META: tl.constexpr):
    COUNT: tl.constexpr = META[0]
    B: tl.constexpr = META[1]
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(OUT + i, 0.0, i < COUNT)


@triton.jit
def _step_a_v_kernel(X, A, OUT, SEG, WIDX, RANKS, PERM, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    SC: tl.constexpr = META[5]
    XS0: tl.constexpr = META[6]
    XS1: tl.constexpr = META[7]
    XS2: tl.constexpr = META[8]
    AS0: tl.constexpr = META[9]
    AS1: tl.constexpr = META[10]
    AS2: tl.constexpr = META[11]
    BM: tl.constexpr = META[12]
    BN: tl.constexpr = META[13]
    BK: tl.constexpr = META[14]
    EVEN_K: tl.constexpr = META[15]
    FLAT: tl.constexpr = META[16]
    HAS_PERM: tl.constexpr = META[17]
    PERM_SIZE: tl.constexpr = META[18]
    m0 = tl.program_id(0) * BM
    m = m0 + tl.arange(0, BM)
    keep = m < M
    if FLAT:
        pos = m
        p0 = m0
        p1 = m0 + BM - 1
        xoff = m * XS1
        ooff = m * N
    else:
        idx = m // H
        head = m - idx * H
        if HAS_PERM:
            keep = keep & (idx < PERM_SIZE)
            rows = tl.load(PERM + idx, mask=keep, other=0).to(tl.int32)
        else:
            rows = idx
        pos = idx
        p0 = m0 // H
        p1 = (m0 + BM - 1) // H
        xoff = rows * XS0 + head * XS1
        ooff = (rows * H + head) * N
    n0 = tl.program_id(1) * BN if N > BN else 0
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for b in range(BS):
        lo = tl.load(SEG + b) * SC
        hi = tl.load(SEG + b + 1) * SC
        if (lo < hi) & (lo <= p1) & (hi > p0):
            w = tl.load(WIDX + b)
            if tl.load(RANKS + w) > 0:
                rmask = keep & (pos >= lo) & (pos < hi)
                if H % BM == 0:
                    partial = acc
                else:
                    partial = tl.zeros((BM, BN), dtype=tl.float32)
                xp = X + xoff[:, None] + offs_k[None, :] * XS2
                ap = (
                    A + w * AS0 + offs_n[None, :] * AS1 + offs_k[:, None] * AS2
                )
                for k_start in range(0, K, BK):
                    if EVEN_K:
                        x = tl.load(xp, mask=rmask[:, None], other=0.0)
                        a = tl.load(ap, mask=nmask[None, :], other=0.0)
                    else:
                        x = tl.load(
                            xp,
                            mask=rmask[:, None]
                            & (k_start + offs_k[None, :] < K),
                            other=0.0,
                        )
                        a = tl.load(
                            ap,
                            mask=nmask[None, :]
                            & (k_start + offs_k[:, None] < K),
                            other=0.0,
                        )
                    partial = tl.dot(x, a, partial)
                    xp += BK * XS2
                    ap += BK * AS2
                if H % BM == 0:
                    acc = partial
                else:
                    acc = tl.where(rmask[:, None], partial, acc)
    tl.store(
        OUT + ooff[:, None] + offs_n[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=keep[:, None] & nmask[None, :],
    )


def step_a_v(attn_output, A_buf, batch_info):
    S, H, K = attn_output.shape
    N = A_buf.shape[1]
    xs0, xs1, xs2 = attn_output.stride()
    seg = batch_info.seg_indptr
    widx = batch_info.weight_indices
    ranks = batch_info.lora_ranks
    perm = batch_info.permutation
    if seg.stride(0) != 1:
        seg = seg.contiguous()
    if widx.stride(0) != 1:
        widx = widx.contiguous()
    if ranks.stride(0) != 1:
        ranks = ranks.contiguous()
    if perm is not None and perm.stride(0) != 1:
        perm = perm.contiguous()
    has_perm = perm is not None
    if has_perm and perm.numel() == 0:
        out = attn_output.new_empty((S, H, N))
        if S * H * N > 0:
            _fill_zero_output.run(
                out,
                (S * H * N, 256),
                grid=((S * H * N + 256 - 1) // 256,),
                warmup=False,
            )
        return out
    flat = not has_perm and xs0 == H * xs1
    if has_perm and perm.numel() != S:
        out = torch.zeros(
            (S, H, N), dtype=attn_output.dtype, device=attn_output.device
        )
    else:
        out = attn_output.new_empty((S, H, N))
        if not has_perm:
            perm = None
    m_total = S * H
    bn = min(32, max(16, 0 if N == 0 else 1 << (N - 1).bit_length()))
    kp = max(16, 0 if K == 0 else 1 << (K - 1).bit_length())
    if m_total <= 4096:
        bm = 16
        bk = min(512, kp)
        stages = 2
    else:
        bm = 64
        bk = min(256, kp)
        stages = 1
    if attn_output.dtype == torch.float32:
        bk = min(64, bk)
    grid = ((m_total + bm - 1) // bm, (N + bn - 1) // bn)
    _step_a_v_kernel.run(
        attn_output,
        A_buf,
        out,
        seg,
        widx,
        ranks,
        perm,
        (
            H,
            N,
            m_total,
            K,
            batch_info.bs,
            H if flat else 1,
            xs0,
            xs1,
            xs2,
            *A_buf.stride(),
            bm,
            bn,
            bk,
            K % bk == 0,
            flat,
            has_perm,
            perm.numel() if has_perm else 0,
        ),
        grid=grid,
        warmup=False,
        num_warps=4,
        num_stages=stages,
    )
    return out


__all__ = ["step_a_v"]
