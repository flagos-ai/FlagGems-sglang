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
def _segment_kernel(X, A, OUT, SEG, WIDX, RANKS, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    TILES: tl.constexpr = META[5]
    XS1: tl.constexpr = META[6]
    XS2: tl.constexpr = META[7]
    AS0: tl.constexpr = META[8]
    AS1: tl.constexpr = META[9]
    AS2: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    EVEN_K: tl.constexpr = META[14]
    pid = tl.program_id(0)
    b = pid // TILES
    tile = pid % TILES
    if b == BS:
        if N > BN and (N > 32 or X.dtype.element_ty == tl.float32):
            if tl.program_id(1) > 0:
                return
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        z = tl.arange(0, 1024)
        for start in range(0, first, 1024):
            p = start + z
            tl.store(OUT + p % N * M + p // N, 0.0, mask=p < first)
        for start in range(last, M * N, 1024):
            p = start + z
            tl.store(OUT + p % N * M + p // N, 0.0, mask=p < M * N)
        return
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    m0 = lo + tile * BM
    if m0 >= hi:
        return
    m = m0 + tl.arange(0, BM)
    keep = m < hi
    n0 = (
        0
        if N <= 32 and X.dtype.element_ty != tl.float32
        else tl.program_id(1) * BN if N > BN else 0
    )
    ns = n0 + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        tl.store(
            OUT + ns[:, None] * M + m[None, :],
            0.0,
            mask=(ns[:, None] < N) & keep[None, :],
        )
        if N <= 32 and X.dtype.element_ty != tl.float32 and (N > BN):
            tl.store(
                OUT + (ns + BN)[:, None] * M + m[None, :],
                0.0,
                mask=((ns + BN)[:, None] < N) & keep[None, :],
            )
        return
    if N <= 32 and X.dtype.element_ty != tl.float32:
        xp = tl.make_block_ptr(
            base=X + lo * XS1,
            shape=(K, hi - lo),
            strides=(XS2, XS1),
            offsets=(0, tile * BM),
            block_shape=(BK, BM),
            order=(0, 1),
        )
        ap = tl.make_block_ptr(
            base=A + w * AS0,
            shape=(N, K),
            strides=(AS1, AS2),
            offsets=(n0, 0),
            block_shape=(BN, BK),
            order=(1, 0),
        )
    else:
        xp = X + m[None, :] * XS1 + ks[:, None] * XS2
        ap = A + w * AS0 + ks[None, :] * AS2 + ns[:, None] * AS1
    acc = tl.zeros((BN, BM), tl.float32)
    if N <= 32 and X.dtype.element_ty != tl.float32 and (N > BN):
        ap2 = tl.make_block_ptr(
            base=A + w * AS0,
            shape=(N, K),
            strides=(AS1, AS2),
            offsets=(n0 + BN, 0),
            block_shape=(BN, BK),
            order=(1, 0),
        )
        acc2 = tl.zeros((BN, BM), tl.float32)
    for k0 in tl.static_range(0, K, BK):
        if N <= 32 and X.dtype.element_ty != tl.float32:
            if H % BM == 0 and N <= 32 and (X.dtype.element_ty != tl.float32):
                if EVEN_K:
                    x = tl.load(xp)
                else:
                    x = tl.load(xp, boundary_check=(0,), padding_option="zero")
            elif EVEN_K:
                x = tl.load(xp, boundary_check=(1,), padding_option="zero")
            else:
                x = tl.load(xp, boundary_check=(0, 1), padding_option="zero")
            if EVEN_K:
                a = tl.load(ap, boundary_check=(0,), padding_option="zero")
            else:
                a = tl.load(ap, boundary_check=(0, 1), padding_option="zero")
        elif EVEN_K:
            if H % BM == 0 and N <= 32 and (X.dtype.element_ty != tl.float32):
                x = tl.load(xp)
            else:
                x = tl.load(xp, mask=keep[None, :], other=0.0)
            a = tl.load(ap, mask=ns[:, None] < N, other=0.0)
        else:
            if H % BM == 0 and N <= 32 and (X.dtype.element_ty != tl.float32):
                x = tl.load(xp, mask=k0 + ks[:, None] < K, other=0.0)
            else:
                x = tl.load(
                    xp, mask=keep[None, :] & (k0 + ks[:, None] < K), other=0.0
                )
            a = tl.load(
                ap, mask=(ns[:, None] < N) & (k0 + ks[None, :] < K), other=0.0
            )
        if N <= 32 and X.dtype.element_ty != tl.float32 and (N > BN):
            if EVEN_K:
                a2 = tl.load(ap2, boundary_check=(0,), padding_option="zero")
            else:
                a2 = tl.load(ap2, boundary_check=(0, 1), padding_option="zero")
        if X.dtype.element_ty == tl.float32:
            acc = tl.dot(a, x, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, x, acc)
        if N <= 32 and X.dtype.element_ty != tl.float32 and (N > BN):
            acc2 = tl.dot(a2, x, acc2)
            ap2 = tl.advance(ap2, (0, BK))
        if N <= 32 and X.dtype.element_ty != tl.float32:
            xp = tl.advance(xp, (BK, 0))
            ap = tl.advance(ap, (0, BK))
        else:
            xp += BK * XS2
            ap += BK * AS2
    tl.store(
        OUT + ns[:, None] * M + m[None, :],
        acc,
        mask=(ns[:, None] < N) & keep[None, :],
    )
    if N <= 32 and X.dtype.element_ty != tl.float32 and (N > BN):
        tl.store(
            OUT + (ns + BN)[:, None] * M + m[None, :],
            acc2,
            mask=((ns + BN)[:, None] < N) & keep[None, :],
        )


@triton.jit
def _small_vector(X, A, OUT, SEG, WIDX, RANKS, PERM, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    BS: tl.constexpr = META[3]
    HAS_PERM: tl.constexpr = META[4]
    X0: tl.constexpr = META[5]
    X1: tl.constexpr = META[6]
    X2: tl.constexpr = META[7]
    A0: tl.constexpr = META[8]
    A1: tl.constexpr = META[9]
    A2: tl.constexpr = META[10]
    BK: tl.constexpr = META[11]
    GN: tl.constexpr = META[12]
    m = tl.program_id(0)
    pos = m // H
    head = m % H
    row = tl.load(PERM + pos).to(tl.int32) if HAS_PERM else pos
    ns = tl.program_id(1) * GN + tl.arange(0, GN)
    ks = tl.arange(0, BK)
    active = False
    adapter = 0
    for b in range(BS):
        lo = tl.load(SEG + b)
        hi = tl.load(SEG + b + 1)
        if (lo <= pos) & (pos < hi):
            adapter = tl.load(WIDX + b)
            active = tl.load(RANKS + adapter) > 0
    if active:
        x = tl.load(X + row * X0 + head * X1 + ks * X2, ks < K, 0).to(
            tl.float32
        )
        a = tl.load(
            A + adapter * A0 + ns[:, None] * A1 + ks[None, :] * A2,
            (ns[:, None] < N) & (ks[None, :] < K),
            0,
        ).to(tl.float32)
        value = tl.sum(a * x[None, :], 1)
        tl.store(OUT + (row * H + head) * N + ns, value, ns < N)
    else:
        tl.store(OUT + (row * H + head) * N + ns, 0.0, ns < N)


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
    acc = tl.zeros((BN, BM), dtype=tl.float32)
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
                    partial = tl.zeros((BN, BM), dtype=tl.float32)
                xp = X + xoff[None, :] + offs_k[:, None] * XS2
                ap = (
                    A + w * AS0 + offs_n[:, None] * AS1 + offs_k[None, :] * AS2
                )
                for k_start in range(0, K, BK):
                    if EVEN_K:
                        x = tl.load(xp, mask=rmask[None, :], other=0.0)
                        a = tl.load(ap, mask=nmask[:, None], other=0.0)
                    else:
                        x = tl.load(
                            xp,
                            mask=rmask[None, :]
                            & (k_start + offs_k[:, None] < K),
                            other=0.0,
                        )
                        a = tl.load(
                            ap,
                            mask=nmask[:, None]
                            & (k_start + offs_k[None, :] < K),
                            other=0.0,
                        )
                    partial = tl.dot(a, x, partial)
                    xp += BK * XS2
                    ap += BK * AS2
                if H % BM == 0:
                    acc = partial
                else:
                    acc = tl.where(rmask[None, :], partial, acc)
    tl.store(
        OUT + ooff[:, None] + offs_n[None, :],
        tl.trans(acc).to(OUT.dtype.element_ty),
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
        if flat and S * H > 4096:
            out = torch.empty_strided(
                (S, H, N),
                (H, 1, S * H),
                dtype=attn_output.dtype,
                device=attn_output.device,
            )
        else:
            out = attn_output.new_empty((S, H, N))
        if not has_perm:
            perm = None
    m_total = S * H
    if m_total <= 4096 and 0 < K <= 128:
        rows = perm.numel() if has_perm else S
        if rows * H > 0 and N > 0:
            _small_vector.run(
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
                    K,
                    batch_info.bs,
                    has_perm,
                    xs0,
                    xs1,
                    xs2,
                    *A_buf.stride(),
                    0 if K == 0 else 1 << (K - 1).bit_length(),
                    4,
                ),
                grid=(rows * H, (N + 4 - 1) // 4),
                warmup=False,
                num_warps=4,
            )
        return out
    bn = (
        16
        if flat
        and m_total > 4096
        and (N <= 32)
        and (attn_output.dtype != torch.float32)
        else min(32, max(16, 0 if N == 0 else 1 << (N - 1).bit_length()))
    )
    kp = max(16, 0 if K == 0 else 1 << (K - 1).bit_length())
    if m_total <= 4096:
        bm = 16
        bk = min(512, kp)
        stages = 2
    else:
        bm = (
            128
            if (N <= 16 or (flat and N <= 32))
            and attn_output.dtype != torch.float32
            else 256
        )
        bk = min(64, kp)
        stages = 1
    if flat and m_total > 4096:
        ml = batch_info.max_len
        if not isinstance(ml, int) or ml <= 0 or ml > S:
            ml = S
        tiles = max(1, (ml * H + bm - 1) // bm)
        _segment_kernel.run(
            attn_output,
            A_buf,
            out,
            seg,
            widx,
            ranks,
            (
                H,
                N,
                m_total,
                K,
                batch_info.bs,
                tiles,
                xs1,
                xs2,
                *A_buf.stride(),
                bm,
                bn,
                bk,
                K % bk == 0,
            ),
            grid=(
                (tiles * batch_info.bs + 1, 1)
                if N <= 32 and attn_output.dtype != torch.float32
                else (tiles * batch_info.bs + 1, (N + bn - 1) // bn)
            ),
            warmup=False,
            num_warps=8,
            num_stages=stages,
        )
        return out
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
        num_warps=8,
        num_stages=stages,
    )
    return out


__all__ = ["step_a_v"]
