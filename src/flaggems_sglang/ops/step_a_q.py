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

"""step_a_q implementation for NVIDIA."""

import torch
import triton
import triton.language as tl


def _step_a_q_metadata(batch_info):
    return tuple(
        (
            x if x.stride(0) == 1 else x.contiguous()
            for x in (
                batch_info.seg_indptr,
                batch_info.weight_indices,
                batch_info.lora_ranks,
            )
        )
    )


@triton.jit
def _step_a_q_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    seg_indptr,
    weight_indices_ptr,
    ranks_ptr,
    perm_ptr,
    H: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    BS: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NK: tl.constexpr,
    BRANCHED: tl.constexpr,
    HAS_PERM: tl.constexpr,
    PERM_SIZE: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    NT: tl.constexpr,
    OUT_STRIDE_0: tl.constexpr,
    OUT_STRIDE_1: tl.constexpr,
    OUT_STRIDE_2: tl.constexpr,
):
    if NT == 1:
        h = tl.program_id(0)
        n0 = 0
    else:
        h = tl.program_id(0) % H
        n0 = tl.program_id(0) // H * BLOCK_N
    i0 = tl.program_id(1) * BLOCK_M
    idx = i0 + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    if HAS_PERM:
        keep = idx < PERM_SIZE
        rows = tl.load(perm_ptr + idx * PERM_STRIDE, mask=keep, other=0).to(
            tl.int32
        )
    else:
        keep = idx < S
        rows = idx
    xb = (
        x_ptr
        + rows[:, None] * X_STRIDE_0
        + h * X_STRIDE_1
        + offs_k[None, :] * X_STRIDE_2
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for b in range(BS):
        lo = tl.load(seg_indptr + b)
        hi = tl.load(seg_indptr + b + 1)
        if BRANCHED:
            if (lo < i0 + BLOCK_M) & (hi > i0):
                w = tl.load(weight_indices_ptr + b)
                if tl.load(ranks_ptr + w) > 0:
                    rmask = keep & (idx >= lo) & (idx < hi)
                    acc += tl.where(
                        rmask[:, None],
                        _step_a_q_tile(
                            x_ptr,
                            weight_ptr,
                            xb,
                            w,
                            h,
                            offs_k,
                            offs_n,
                            keep,
                            H,
                            N,
                            S,
                            K,
                            FULL_K_PER_HEAD,
                            X_STRIDE_2,
                            W_STRIDE_0,
                            W_STRIDE_1,
                            W_STRIDE_2,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_K,
                            NK,
                            HAS_PERM,
                        ),
                        0.0,
                    )
        else:
            w = tl.load(weight_indices_ptr + b)
            rmask = (
                keep & (idx >= lo) & (idx < hi) & (tl.load(ranks_ptr + w) > 0)
            )
            acc += tl.where(
                rmask[:, None],
                _step_a_q_tile(
                    x_ptr,
                    weight_ptr,
                    xb,
                    w,
                    h,
                    offs_k,
                    offs_n,
                    keep,
                    H,
                    N,
                    S,
                    K,
                    FULL_K_PER_HEAD,
                    X_STRIDE_2,
                    W_STRIDE_0,
                    W_STRIDE_1,
                    W_STRIDE_2,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    NK,
                    HAS_PERM,
                ),
                0.0,
            )
    op = (
        out_ptr
        + rows[:, None] * OUT_STRIDE_0
        + h * OUT_STRIDE_1
        + offs_n[None, :] * OUT_STRIDE_2
    )
    val = acc.to(out_ptr.dtype.element_ty)
    if BLOCK_N == N:
        if not HAS_PERM and S % BLOCK_M == 0:
            tl.store(op, val)
        else:
            tl.store(op, val, mask=keep[:, None])
    else:
        tl.store(op, val, mask=keep[:, None] & (offs_n < N)[None, :])


@triton.jit
def _step_a_q_tile(
    x_ptr,
    weight_ptr,
    xb,
    w,
    h,
    offs_k,
    offs_n,
    keep,
    H: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NK: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    wbase = (
        weight_ptr
        + w * W_STRIDE_0
        + (h * FULL_K_PER_HEAD + offs_k[:, None]) * W_STRIDE_1
        + offs_n[None, :] * W_STRIDE_2
    )
    part = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    CACHE: tl.constexpr = (
        ".cg" if NK == 1 and x_ptr.dtype.element_ty != tl.float32 else ".ca"
    )
    for ki in range(NK):
        kk = ki * BLOCK_K
        xp = xb + kk * X_STRIDE_2
        wp = wbase + kk * W_STRIDE_1
        if NK * BLOCK_K == K:
            if HAS_PERM or S % BLOCK_M == 0:
                x = tl.load(xp, cache_modifier=CACHE)
            else:
                x = tl.load(
                    xp, mask=keep[:, None], other=0.0, cache_modifier=CACHE
                )
            if BLOCK_N == N:
                wt = tl.load(wp)
            else:
                wt = tl.load(wp, mask=(offs_n < N)[None, :], other=0.0)
        else:
            km = kk + offs_k < K
            x = tl.load(
                xp,
                mask=keep[:, None] & km[None, :],
                other=0.0,
                cache_modifier=CACHE,
            )
            wt = tl.load(
                wp, mask=km[:, None] & (offs_n < N)[None, :], other=0.0
            )
        if x_ptr.dtype.element_ty == tl.float32:
            part = tl.dot(x, wt, part, input_precision="ieee")
        else:
            part = tl.dot(x, wt, part)
    return part


@triton.jit
def _step_a_q_m_pipeline_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    seg_indptr,
    weight_indices_ptr,
    ranks_ptr,
    H: tl.constexpr,
    N: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    BS: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_STRIDE_0: tl.constexpr,
    OUT_STRIDE_1: tl.constexpr,
    OUT_STRIDE_2: tl.constexpr,
):
    pid = tl.program_id(0)
    h = tl.program_id(1)
    b = pid // TILES
    tile = pid % TILES
    n = tl.arange(0, BLOCK_N)
    m = tl.arange(0, BLOCK_M)
    if b == BS:
        first = tl.load(seg_indptr)
        for begin in range(0, first, BLOCK_M):
            rows = begin + m
            tl.store(
                out_ptr
                + rows[:, None] * OUT_STRIDE_0
                + h * OUT_STRIDE_1
                + n[None, :] * OUT_STRIDE_2,
                tl.full((BLOCK_M, BLOCK_N), 0, tl.float32),
                (rows < first)[:, None] & (n < N)[None, :],
            )
        last = tl.load(seg_indptr + BS)
        for begin in range(last, S, BLOCK_M):
            rows = begin + m
            tl.store(
                out_ptr
                + rows[:, None] * OUT_STRIDE_0
                + h * OUT_STRIDE_1
                + n[None, :] * OUT_STRIDE_2,
                tl.full((BLOCK_M, BLOCK_N), 0, tl.float32),
                (rows < S)[:, None] & (n < N)[None, :],
            )
        return
    lo = tl.load(seg_indptr + b)
    hi = tl.load(seg_indptr + b + 1)
    start = lo + tile * 2 * BLOCK_M
    if start >= hi:
        return
    w = tl.load(weight_indices_ptr + b)
    if tl.load(ranks_ptr + w) == 0:
        for turn in range(2):
            rows = start + turn * BLOCK_M + m
            tl.store(
                out_ptr
                + rows[:, None] * OUT_STRIDE_0
                + h * OUT_STRIDE_1
                + n[None, :] * OUT_STRIDE_2,
                tl.full((BLOCK_M, BLOCK_N), 0, tl.float32),
                (rows < hi)[:, None] & (n < N)[None, :],
            )
        return
    k = tl.arange(0, BLOCK_K)
    wt = tl.load(
        weight_ptr
        + w * W_STRIDE_0
        + (h * FULL_K_PER_HEAD + k[:, None]) * W_STRIDE_1
        + n[None, :] * W_STRIDE_2,
        (k < K)[:, None] & (n < N)[None, :],
        0,
    )
    for turn in tl.range(0, 2, num_stages=1):
        rows = start + turn * BLOCK_M + m
        x = tl.load(
            x_ptr
            + rows[:, None] * X_STRIDE_0
            + h * X_STRIDE_1
            + k[None, :] * X_STRIDE_2,
            (rows < hi)[:, None] & (k < K)[None, :],
            0,
            cache_modifier=".cg",
        )
        value = tl.dot(x, wt)
        tl.store(
            out_ptr
            + rows[:, None] * OUT_STRIDE_0
            + h * OUT_STRIDE_1
            + n[None, :] * OUT_STRIDE_2,
            value,
            (rows < hi)[:, None] & (n < N)[None, :],
        )


@triton.jit
def _step_a_q_bounded_fp32_cuda_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    seg_indptr,
    weight_indices_ptr,
    ranks_ptr,
    perm_ptr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    NT: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    HAS_PERM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2) // NT
    n0 = tl.program_id(2) % NT * BLOCK_N
    lo = tl.load(seg_indptr + b)
    hi = tl.load(seg_indptr + b + 1)
    if lo + tile * BLOCK_M >= hi:
        return
    w = tl.load(weight_indices_ptr + b)
    if tl.load(ranks_ptr + w) == 0:
        return
    idx = lo + tile * BLOCK_M + tl.arange(0, BLOCK_M)
    keep = idx < hi
    if HAS_PERM:
        row = tl.load(perm_ptr + idx * PERM_STRIDE, keep, 0).to(tl.int32)
    else:
        row = idx
    n = n0 + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for start in range(0, K, BLOCK_K):
        kk = start + k
        x = tl.load(
            x_ptr
            + row[:, None] * X_STRIDE_0
            + h * X_STRIDE_1
            + kk[None, :] * X_STRIDE_2,
            keep[:, None] & (kk < K)[None, :],
            0,
        )
        wt = tl.load(
            weight_ptr
            + w * W_STRIDE_0
            + (h * FULL_K_PER_HEAD + kk[:, None]) * W_STRIDE_1
            + n[None, :] * W_STRIDE_2,
            (kk < K)[:, None] & (n < N)[None, :],
            0,
        )
        acc = tl.dot(x, wt, acc, input_precision="tf32x3")
    tl.store(
        out_ptr + (row[:, None] * H + h) * N + n[None, :],
        acc,
        keep[:, None] & (n < N)[None, :],
    )


def _step_a_q_cuda_float32(q_nope, B_buf, batch_info, full_K_per_head):
    (S, H, K) = q_nope.shape
    N = B_buf.shape[-1]
    out = torch.zeros((S, H, N), dtype=q_nope.dtype, device=q_nope.device)
    if (
        B_buf.shape[0] == 0
        or q_nope.numel() == 0
        or N == 0
        or (batch_info.bs == 0)
    ):
        return out
    max_len = getattr(batch_info, "max_len", None)
    if max_len is None:
        max_len = S
    if max_len == 0:
        return out
    (seg_indptr, weight_indices, ranks) = _step_a_q_metadata(batch_info)
    q_nope = q_nope.contiguous()
    B_buf = B_buf.contiguous()
    permutation = batch_info.permutation
    has_permutation = permutation is not None
    p = permutation if has_permutation else seg_indptr
    block_m = 16 if max_len <= 16 else 32
    (block_n, block_k) = (32, 64)
    nt = triton.cdiv(N, block_n)
    _step_a_q_bounded_fp32_cuda_kernel[
        triton.cdiv(max_len, block_m), H, batch_info.bs * nt
    ](
        q_nope,
        B_buf,
        out,
        seg_indptr,
        weight_indices,
        ranks,
        p,
        H,
        N,
        K,
        full_K_per_head,
        nt,
        *q_nope.stride(),
        *B_buf.stride(),
        p.stride(0),
        has_permutation,
        block_m,
        block_n,
        block_k,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _step_a_q_small_zero_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    seg_indptr,
    weight_indices_ptr,
    ranks_ptr,
    perm_ptr,
    S: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    FULL_K_PER_HEAD: tl.constexpr,
    BS: tl.constexpr,
    ML: tl.constexpr,
    HAS_PERM: tl.constexpr,
    X_STRIDE_0: tl.constexpr,
    X_STRIDE_1: tl.constexpr,
    X_STRIDE_2: tl.constexpr,
    W_STRIDE_0: tl.constexpr,
    W_STRIDE_1: tl.constexpr,
    W_STRIDE_2: tl.constexpr,
    PERM_STRIDE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    NT: tl.constexpr,
    PERM_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == BS * ML * H * NT:
        first = tl.load(seg_indptr)
        last = tl.load(seg_indptr + BS)
        if HAS_PERM:
            off = tl.arange(0, BLOCK_ROWS)
            for i in range(0, first):
                row = tl.load(perm_ptr + i * PERM_STRIDE).to(tl.int32)
                for col in range(0, H * N, BLOCK_ROWS):
                    pos = col + off
                    tl.store(out_ptr + row * H * N + pos, 0.0, pos < H * N)
            for i in range(last, PERM_SIZE):
                row = tl.load(perm_ptr + i * PERM_STRIDE).to(tl.int32)
                for col in range(0, H * N, BLOCK_ROWS):
                    pos = col + off
                    tl.store(out_ptr + row * H * N + pos, 0.0, pos < H * N)
        else:
            off = tl.arange(0, 1024)
            for i in range(0, first * H * N, 1024):
                pos = i + off
                tl.store(out_ptr + pos, 0.0, pos < first * H * N)
            for i in range(last * H * N, S * H * N, 1024):
                pos = i + off
                tl.store(out_ptr + pos, 0.0, pos < S * H * N)
        return
    b = pid // (ML * H * NT)
    m = pid // (H * NT) % ML
    h = pid % H
    n0 = pid // H % NT * BLOCK_N
    lo = tl.load(seg_indptr + b)
    hi = tl.load(seg_indptr + b + 1)
    if lo + m >= hi:
        return
    if HAS_PERM:
        row = tl.load(perm_ptr + (lo + m) * PERM_STRIDE).to(tl.int32)
    else:
        row = lo + m
    n = n0 + tl.arange(0, BLOCK_N)
    op = out_ptr + (row * H + h) * N + n
    w = tl.load(weight_indices_ptr + b)
    if tl.load(ranks_ptr + w) == 0:
        tl.store(op, tl.full((BLOCK_N,), 0, tl.float32), n < N)
        return
    k = tl.arange(0, BLOCK_K)
    value = tl.zeros((BLOCK_N,), tl.float32)
    for start in range(0, K, BLOCK_K):
        kk = start + k
        x = tl.load(
            x_ptr + row * X_STRIDE_0 + h * X_STRIDE_1 + kk * X_STRIDE_2,
            kk < K,
            0,
        ).to(tl.float32)
        wt = tl.load(
            weight_ptr
            + w * W_STRIDE_0
            + (h * FULL_K_PER_HEAD + kk[None, :]) * W_STRIDE_1
            + n[:, None] * W_STRIDE_2,
            (kk < K)[None, :] & (n < N)[:, None],
            0,
        ).to(tl.float32)
        value += tl.sum(x[None, :] * wt, 1)
    tl.store(op, value, n < N)


def _step_a_q_small_zero(q_nope, B_buf, batch_info, full_K_per_head, max_len):
    (S, H, K) = q_nope.shape
    N = B_buf.shape[-1]
    (seg_indptr, weight_indices, ranks) = _step_a_q_metadata(batch_info)
    permutation = batch_info.permutation
    has_permutation = permutation is not None
    p = permutation if has_permutation else seg_indptr
    if has_permutation and permutation.numel() != S:
        out = torch.zeros((S, H, N), dtype=q_nope.dtype, device=q_nope.device)
    else:
        out = torch.empty((S, H, N), dtype=q_nope.dtype, device=q_nope.device)
    block_k = min(256, triton.next_power_of_2(K))
    block_n = min(triton.next_power_of_2(N), max(1, 4096 // block_k))
    nt = triton.cdiv(N, block_n)
    _step_a_q_small_zero_kernel[batch_info.bs * max_len * H * nt + 1,](
        q_nope,
        B_buf,
        out,
        seg_indptr,
        weight_indices,
        ranks,
        p,
        S,
        H,
        K,
        N,
        full_K_per_head,
        batch_info.bs,
        max_len,
        has_permutation,
        *q_nope.stride(),
        *B_buf.stride(),
        p.stride(0),
        block_k,
        block_n,
        min(1024, triton.next_power_of_2(H * N)),
        nt,
        permutation.numel() if has_permutation else S,
        num_warps=4,
    )
    return out


@triton.jit
def _step_a_q_constant_zero_kernel(
    out_ptr,
    NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID: tl.constexpr,
):
    pid = tl.program_id(0)
    ix = tl.arange(0, BLOCK_SIZE)
    if triton.cdiv(NUM, BLOCK_SIZE) <= 65535:
        off = pid * BLOCK_SIZE + ix
        tl.store(out_ptr + off, 0.0, off < NUM)
    else:
        for block in range(pid, triton.cdiv(NUM, BLOCK_SIZE), GRID):
            off = block * BLOCK_SIZE + ix
            tl.store(out_ptr + off, 0.0, off < NUM)


def _step_a_q_zero_output(q_nope, rank):
    out = torch.empty(
        (*q_nope.shape[:2], rank), dtype=q_nope.dtype, device=q_nope.device
    )
    size = out.numel()
    if size:
        grid = min(65535, triton.cdiv(size, 1024))
        _step_a_q_constant_zero_kernel[grid,](
            out, size, 1024, grid, num_warps=4, num_stages=1
        )
    return out


def step_a_q(q_nope, B_buf, batch_info, full_K_per_head):
    if (
        B_buf.shape[0] == 0
        or q_nope.numel() == 0
        or B_buf.shape[-1] == 0
        or (batch_info.bs == 0)
        or (getattr(batch_info, "max_len", None) == 0)
    ):
        return _step_a_q_zero_output(q_nope, B_buf.shape[-1])
    if q_nope.dtype == torch.float32 and torch.version.hip is None:
        max_len = getattr(batch_info, "max_len", None)
        if max_len is None:
            max_len = q_nope.shape[0]
        if batch_info.bs > 0 and 0 < max_len <= 16:
            return _step_a_q_small_zero(
                q_nope, B_buf, batch_info, full_K_per_head, max_len
            )
        return _step_a_q_cuda_float32(
            q_nope, B_buf, batch_info, full_K_per_head
        )
    (seg_indptr, weight_indices, ranks) = _step_a_q_metadata(batch_info)
    (S, H, K) = q_nope.shape
    N = B_buf.shape[2]
    permutation = batch_info.permutation
    has_permutation = permutation is not None
    if (
        torch.version.hip is None
        and (not has_permutation)
        and (S >= 64)
        and (128 <= K <= 256)
        and (N <= 64)
        and (q_nope.dtype != torch.float32)
    ):
        max_len = getattr(batch_info, "max_len", None)
        if max_len is None:
            max_len = S
        block_m = 32 if max_len <= 64 else 64
        tiles = max(1, triton.cdiv(max_len, block_m * 2))
        out = (
            torch.empty_strided(
                (S, H, N),
                (N, S * N, 1),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
            if not has_permutation
            and S >= 64
            and (q_nope.dtype != torch.float32)
            and (K == 128 and N == 32 or (K == 64 and N == 16))
            else torch.empty(
                (S, H, N), dtype=q_nope.dtype, device=q_nope.device
            )
        )
        _step_a_q_m_pipeline_kernel[tiles * batch_info.bs + 1, H](
            q_nope,
            B_buf,
            out,
            seg_indptr,
            weight_indices,
            ranks,
            H,
            N,
            S,
            K,
            full_K_per_head,
            batch_info.bs,
            *q_nope.stride(),
            *B_buf.stride(),
            tiles,
            block_m,
            max(16, triton.next_power_of_2(N)),
            max(16, triton.next_power_of_2(K)),
            *out.stride(),
            num_warps=4,
            num_stages=3,
        )
        return out
    if has_permutation:
        covered = permutation.numel() == S
    else:
        permutation = q_nope
        covered = True
    if covered:
        out = (
            torch.empty_strided(
                (S, H, N),
                (N, S * N, 1),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
            if not has_permutation
            and S >= 64
            and (q_nope.dtype != torch.float32)
            and (K == 128 and N == 32 or (K == 64 and N == 16))
            else torch.empty(
                (S, H, N), dtype=q_nope.dtype, device=q_nope.device
            )
        )
    else:
        out = torch.zeros((S, H, N), dtype=q_nope.dtype, device=q_nope.device)
    block_m = 16 if S < 64 else 64
    block_n = min(64, max(16, triton.next_power_of_2(N)))
    nt = triton.cdiv(N, block_n)
    block_k = min(128, max(16, triton.next_power_of_2(K)))
    if q_nope.dtype == torch.float32:
        block_k = min(64, block_k)
    nk = triton.cdiv(K, block_k)
    _step_a_q_kernel[H * nt, triton.cdiv(S, block_m)](
        q_nope,
        B_buf,
        out,
        seg_indptr,
        weight_indices,
        ranks,
        permutation,
        H,
        N,
        S,
        K,
        full_K_per_head,
        batch_info.bs,
        *q_nope.stride(),
        *B_buf.stride(),
        block_m,
        block_n,
        block_k,
        nk,
        S >= 64,
        has_permutation,
        permutation.numel() if has_permutation else S,
        permutation.stride(0) if has_permutation else 1,
        nt,
        *out.stride(),
        num_warps=4,
        num_stages=3,
    )
    return out


__all__ = ["step_a_q"]
