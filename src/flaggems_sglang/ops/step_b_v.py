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
def _step_b_v_kernel(
    X, W, BASE, OUT, SEG, WIDX, RANKS, SCALES, PERM, META: tl.constexpr
):
    N: tl.constexpr = META[1]
    S: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    F: tl.constexpr = META[4]
    Q: tl.constexpr = META[5]
    BS: tl.constexpr = META[6]
    XS0: tl.constexpr = META[7]
    XS1: tl.constexpr = META[8]
    XS2: tl.constexpr = META[9]
    WS0: tl.constexpr = META[10]
    WS1: tl.constexpr = META[11]
    WS2: tl.constexpr = META[12]
    GS0: tl.constexpr = META[13]
    GS1: tl.constexpr = META[14]
    GS2: tl.constexpr = META[15]
    BM: tl.constexpr = META[16]
    BN: tl.constexpr = META[17]
    BK: tl.constexpr = META[18]
    HAS_PERM: tl.constexpr = META[19]
    PL: tl.constexpr = META[20]
    EXACT: tl.constexpr = META[21]
    NFULL: tl.constexpr = META[22]
    KFULL: tl.constexpr = META[23]
    OS0: tl.constexpr = META[24]
    OS1: tl.constexpr = META[25]
    OS2: tl.constexpr = META[26]
    USE_BLOCK: tl.constexpr = META[27]
    h = tl.program_id(0)
    i0 = tl.program_id(1) * BM
    idx = i0 + tl.arange(0, BM)
    if HAS_PERM:
        keep = idx < PL
        rows = tl.load(PERM + idx, mask=keep, other=0).to(tl.int32)
    else:
        keep = idx < S
        rows = idx
    offs_k = tl.arange(0, BK)
    offs_n = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    rscale = tl.zeros((BM,), dtype=tl.float32)
    owner = tl.full((), -1, tl.int32)
    if not HAS_PERM and S >= BM:
        for segment in range(BS):
            lower = tl.load(SEG + segment)
            upper = tl.load(SEG + segment + 1)
            if (lower <= i0) & (upper >= i0 + BM):
                owner = segment
    if owner >= 0:
        w = tl.load(WIDX + owner)
        if tl.load(RANKS + w) > 0:
            if USE_BLOCK:
                xp = tl.make_block_ptr(
                    base=X + h * XS1,
                    shape=(S, K),
                    strides=(XS0, XS2),
                    offsets=(i0, 0),
                    block_shape=(BM, BK),
                    order=(1, 0),
                )
                wp = tl.make_block_ptr(
                    base=W + w * WS0 + (h * F + Q) * WS1,
                    shape=(K, N),
                    strides=(WS2, WS1),
                    offsets=(0, 0),
                    block_shape=(BK, BN),
                    order=(0, 1),
                )
                if KFULL:
                    x = tl.load(xp, cache_modifier=".cg")
                    if NFULL:
                        wb = tl.load(wp, eviction_policy="evict_last")
                    else:
                        wb = tl.load(
                            wp,
                            boundary_check=(1,),
                            padding_option="zero",
                            eviction_policy="evict_last",
                        )
                else:
                    x = tl.load(
                        xp,
                        boundary_check=(1,),
                        padding_option="zero",
                        cache_modifier=".cg",
                    )
                    wb = tl.load(
                        wp,
                        boundary_check=(0, 1),
                        padding_option="zero",
                        eviction_policy="evict_last",
                    )
            else:
                xp = X + rows[:, None] * XS0 + h * XS1 + offs_k[None, :] * XS2
                wp = (
                    W
                    + w * WS0
                    + (h * F + Q + offs_n[None, :]) * WS1
                    + offs_k[:, None] * WS2
                )
                if KFULL:
                    x = tl.load(xp, cache_modifier=".cg")
                    if NFULL:
                        wb = tl.load(wp, eviction_policy="evict_last")
                    else:
                        wb = tl.load(
                            wp,
                            mask=(offs_n < N)[None, :],
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                else:
                    x = tl.load(
                        xp,
                        mask=(offs_k < K)[None, :],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                    wb = tl.load(
                        wp,
                        mask=(offs_k < K)[:, None] & (offs_n < N)[None, :],
                        other=0.0,
                        eviction_policy="evict_last",
                    )
            if X.dtype.element_ty == tl.float32:
                acc = tl.dot(x, wb, input_precision="ieee")
            else:
                acc = tl.dot(x, wb)
            rscale = tl.full((BM,), 1.0, tl.float32) * tl.load(SCALES + w)
    else:
        for b in range(BS):
            lo = tl.load(SEG + b)
            hi = tl.load(SEG + b + 1)
            if (lo < i0 + BM) & (hi > i0):
                w = tl.load(WIDX + b)
                if tl.load(RANKS + w) > 0:
                    rmask = (idx >= lo) & (idx < hi)
                    if not EXACT:
                        rmask = rmask & keep
                    xp = (
                        X
                        + rows[:, None] * XS0
                        + h * XS1
                        + offs_k[None, :] * XS2
                    )
                    wp = (
                        W
                        + w * WS0
                        + (h * F + Q + offs_n[None, :]) * WS1
                        + offs_k[:, None] * WS2
                    )
                    if KFULL:
                        x = tl.load(
                            xp,
                            mask=rmask[:, None],
                            other=0.0,
                            cache_modifier=".cg",
                        )
                        if NFULL:
                            wb = tl.load(wp, eviction_policy="evict_last")
                        else:
                            wb = tl.load(
                                wp,
                                mask=(offs_n < N)[None, :],
                                other=0.0,
                                eviction_policy="evict_last",
                            )
                    else:
                        x = tl.load(
                            xp,
                            mask=rmask[:, None] & (offs_k < K)[None, :],
                            other=0.0,
                            cache_modifier=".cg",
                        )
                        wb = tl.load(
                            wp,
                            mask=(offs_k < K)[:, None] & (offs_n < N)[None, :],
                            other=0.0,
                            eviction_policy="evict_last",
                        )
                    if (lo <= i0) & (hi >= i0 + BM):
                        if X.dtype.element_ty == tl.float32:
                            acc = tl.dot(x, wb, acc, input_precision="ieee")
                        else:
                            acc = tl.dot(x, wb, acc)
                    else:
                        if X.dtype.element_ty == tl.float32:
                            partial = tl.dot(x, wb, input_precision="ieee")
                        else:
                            partial = tl.dot(x, wb)
                        acc += tl.where(rmask[:, None], partial, 0.0)
                    rscale = tl.where(rmask, tl.load(SCALES + w), rscale)
    bp = BASE + rows[:, None] * GS0 + h * GS1 + offs_n[None, :] * GS2
    op = OUT + rows[:, None] * OS0 + h * OS1 + offs_n[None, :] * OS2
    if EXACT:
        if USE_BLOCK:
            bp_tile = tl.make_block_ptr(
                base=BASE + h * GS1,
                shape=(S, N),
                strides=(GS0, GS2),
                offsets=(i0, 0),
                block_shape=(BM, BN),
                order=(1, 0),
            )
            op_tile = tl.make_block_ptr(
                base=OUT + h * OS1,
                shape=(S, N),
                strides=(OS0, OS2),
                offsets=(i0, 0),
                block_shape=(BM, BN),
                order=(1, 0),
            )
            base = tl.load(bp_tile, cache_modifier=".cg").to(tl.float32)
            tl.store(
                op_tile,
                (base + rscale[:, None] * acc).to(OUT.dtype.element_ty),
            )
        else:
            base = tl.load(bp, cache_modifier=".cg").to(tl.float32)
            tl.store(
                op, (base + rscale[:, None] * acc).to(OUT.dtype.element_ty)
            )
    else:
        if NFULL:
            m2 = keep[:, None]
        else:
            m2 = keep[:, None] & (offs_n < N)[None, :]
        base = tl.load(bp, mask=m2, other=0.0, cache_modifier=".cg").to(
            tl.float32
        )
        tl.store(
            op,
            (base + rscale[:, None] * acc).to(OUT.dtype.element_ty),
            mask=m2,
        )


def _optimized_step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    S, H, K = attn_lora_a.shape
    N = v_head_dim
    perm = batch_info.permutation
    if perm is not None:
        perm = perm.contiguous()
    has_perm = perm is not None
    bn = max(16, 0 if N == 0 else 1 << (N - 1).bit_length())
    bk = max(16, 0 if K == 0 else 1 << (K - 1).bit_length())
    bm = max(16, 8192 // bn)
    bm = (
        min(bm, 16 if K < 32 else 32)
        if torch.version.hip is not None
        and (not has_perm)
        and (S >= 64)
        and (K >= 16)
        else bm
    )
    if has_perm and perm.numel() != S:
        out = base_output.clone(memory_format=torch.contiguous_format)
    else:
        out = (
            base_output.new_empty((H, S, N)).permute(1, 0, 2)
            if torch.version.hip is not None and (not has_perm) and (S >= 64)
            else base_output.new_empty((S, H, N))
        )
    if not has_perm:
        perm = None
    _step_b_v_kernel.run(
        attn_lora_a,
        B_buf,
        base_output,
        out,
        batch_info.seg_indptr.contiguous(),
        batch_info.weight_indices.contiguous(),
        batch_info.lora_ranks.contiguous(),
        batch_info.scalings.contiguous(),
        perm,
        (
            H,
            N,
            S,
            K,
            qk_nope_head_dim + N,
            qk_nope_head_dim,
            batch_info.bs,
            *attn_lora_a.stride(),
            *B_buf.stride(),
            *base_output.stride(),
            bm,
            bn,
            bk,
            has_perm,
            perm.numel() if has_perm else S,
            not has_perm and S % bm == 0 and (N == bn),
            N == bn,
            K == bk,
            *out.stride(),
            torch.version.hip is not None
            and (not has_perm)
            and (S >= 64)
            and (K >= 32),
        ),
        num_warps=4,
        num_stages=2,
        grid=(H, (S + bm - 1) // bm),
        warmup=False,
    )
    return out


@triton.jit
def _bounded_rows(X, W, OUT, SEG, WI, RANKS, SCALES, PERM, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    K: tl.constexpr = META[2]
    F: tl.constexpr = META[3]
    Q: tl.constexpr = META[4]
    ML: tl.constexpr = META[5]
    NT: tl.constexpr = META[6]
    TOTAL: tl.constexpr = META[7]
    GRID: tl.constexpr = META[8]
    PL: tl.constexpr = META[9]
    HAS_PERM: tl.constexpr = META[10]
    X0: tl.constexpr = META[11]
    X1: tl.constexpr = META[12]
    X2: tl.constexpr = META[13]
    W0: tl.constexpr = META[14]
    W1: tl.constexpr = META[15]
    W2: tl.constexpr = META[16]
    BK: tl.constexpr = META[17]
    BN: tl.constexpr = META[18]
    for work in range(tl.program_id(0), TOTAL, GRID):
        panel = work % NT
        h = work // NT % H
        m = work // (NT * H) % ML
        b = work // (NT * H * ML)
        lo = tl.load(SEG + b)
        hi = tl.load(SEG + b + 1)
        logical = lo + m
        if (logical < hi) & (logical < PL):
            w = tl.load(WI + b)
            if tl.load(RANKS + w) != 0:
                if HAS_PERM:
                    row = tl.load(PERM + logical).to(tl.int32)
                else:
                    row = logical
                n = panel * BN + tl.arange(0, BN)
                ks = tl.arange(0, BK)
                correction = tl.zeros((BN,), tl.float32)
                for chunk in range(tl.cdiv(K, BK)):
                    k = chunk * BK + ks
                    x = tl.load(
                        X + row * X0 + h * X1 + k * X2, k < K, other=0.0
                    ).to(tl.float32)
                    weight = tl.load(
                        W
                        + w * W0
                        + (h * F + Q + n[None, :]) * W1
                        + k[:, None] * W2,
                        (k < K)[:, None] & (n < N)[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    correction += tl.sum(x[:, None] * weight, axis=0)
                ptr = OUT + (row * H + h) * N + n
                base = tl.load(ptr, n < N, other=0.0).to(tl.float32)
                scale = tl.load(SCALES + w).to(tl.float32)
                tl.store(ptr, base + correction * scale, n < N)


def step_b_v(
    attn_lora_a, B_buf, batch_info, base_output, qk_nope_head_dim, v_head_dim
):
    if (
        v_head_dim <= 128
        and attn_lora_a.shape[-1] * attn_lora_a.element_size() <= 256
        or base_output.numel() == 0
        or batch_info.bs == 0
        or (B_buf.shape[0] == 0)
    ):
        return _optimized_step_b_v(
            attn_lora_a,
            B_buf,
            batch_info,
            base_output,
            qk_nope_head_dim,
            v_head_dim,
        )
    x = attn_lora_a.contiguous()
    weight = B_buf.contiguous()
    S, H, K = x.shape
    N = v_head_dim
    ml = batch_info.max_len or S
    nt = (N + 32 - 1) // 32
    total = ml * H * batch_info.bs * nt
    grid = max(1, min(8192, total))
    out = base_output.clone(memory_format=torch.contiguous_format)
    perm = batch_info.permutation
    p = perm.contiguous() if perm is not None else batch_info.seg_indptr
    _bounded_rows.run(
        x,
        weight,
        out,
        batch_info.seg_indptr.contiguous(),
        batch_info.weight_indices.contiguous(),
        batch_info.lora_ranks.contiguous(),
        batch_info.scalings.contiguous(),
        p,
        (
            H,
            N,
            K,
            qk_nope_head_dim + N,
            qk_nope_head_dim,
            ml,
            nt,
            total,
            grid,
            perm.numel() if perm is not None else S,
            perm is not None,
            *x.stride(),
            *weight.stride(),
            32,
            32,
        ),
        grid=(grid,),
        warmup=False,
    )
    return out


__all__ = ["step_b_v"]
