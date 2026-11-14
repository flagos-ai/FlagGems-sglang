# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task 96 follow-up: one allocation and five tensor arguments."""

import os

import torch
import triton
import triton.language as tl

# GCU launch limit: grid.x <= 65535 (grid.y/z would be <= 255).
_GRID_X_MAX = 65535


def _int64_env_enabled():
    return os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "0") not in (
        "",
        "0",
    )


@triton.jit
def _offset(
    token, column, S: tl.constexpr, D: tl.constexpr, STRIDES: tl.constexpr
):
    return (
        (token // S) * STRIDES[0]
        + (token % S) * STRIDES[1]
        + (column // D) * STRIDES[2]
        + (column % D) * STRIDES[3]
    )


@triton.jit
def _pack(
    Q,
    K,
    V,
    Index,
    Out,
    N: tl.constexpr,
    TOKENS: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    IS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Index is either a true int32 tensor or a fake-int64 one (int64
    # metadata over int32 physical storage); both are read through an
    # int32 pointer view.
    ip = Index.to(tl.pointer_type(tl.int32))
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, column = i // (H * D), i % (H * D)
    token = tl.load(ip + row * IS, i < N, other=0).to(tl.int64)
    token = tl.where(token < 0, token + TOKENS, token)
    q = tl.load(Q + _offset(token, column, S, D, QS), i < N, other=0)
    k = tl.load(K + _offset(token, column, S, D, KS), i < N, other=0)
    v = tl.load(V + _offset(token, column, S, D, VS), i < N, other=0)
    tl.store(Out + i, q, i < N)
    tl.store(Out + (N + i), k, i < N)
    tl.store(Out + (2 * N + i), v, i < N)


@triton.jit
def _pack_group4(
    Q,
    K,
    V,
    Index,
    Out,
    ROWS,
    N,
    TOKENS,
    HD: tl.constexpr,
):
    # Contiguous fast path: each program packs 4 whole output rows with an
    # affine token*HD source offset (no div/mod). The scalar token loads
    # are issued back-to-back and every wide vector load is issued before
    # the first store: on L600 independent wide loads inside a straight
    # line segment pipeline, while interleaving loads with stores serialises
    # them (-29~37% measured on this op). Dead lanes of the tail group read
    # source row 0 (token masked to 0) and only the stores are masked.
    # Index arrives as int32 or as fake-int64 (int64 metadata over int32
    # storage); the int32 pointer view covers both.
    ip = Index.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    cols = tl.arange(0, HD)
    full = cols < HD
    base = pid * 4
    t0 = tl.load(ip + base, mask=base < ROWS, other=0)
    t1 = tl.load(ip + base + 1, mask=base + 1 < ROWS, other=0)
    t2 = tl.load(ip + base + 2, mask=base + 2 < ROWS, other=0)
    t3 = tl.load(ip + base + 3, mask=base + 3 < ROWS, other=0)
    t0 = tl.where(t0 < 0, t0 + TOKENS, t0)
    t1 = tl.where(t1 < 0, t1 + TOKENS, t1)
    t2 = tl.where(t2 < 0, t2 + TOKENS, t2)
    t3 = tl.where(t3 < 0, t3 + TOKENS, t3)
    q0 = tl.load(Q + t0 * HD + cols)
    k0 = tl.load(K + t0 * HD + cols)
    v0 = tl.load(V + t0 * HD + cols)
    q1 = tl.load(Q + t1 * HD + cols)
    k1 = tl.load(K + t1 * HD + cols)
    v1 = tl.load(V + t1 * HD + cols)
    q2 = tl.load(Q + t2 * HD + cols)
    k2 = tl.load(K + t2 * HD + cols)
    v2 = tl.load(V + t2 * HD + cols)
    q3 = tl.load(Q + t3 * HD + cols)
    k3 = tl.load(K + t3 * HD + cols)
    v3 = tl.load(V + t3 * HD + cols)
    m0 = base < ROWS
    m1 = base + 1 < ROWS
    m2 = base + 2 < ROWS
    m3 = base + 3 < ROWS
    tl.store(Out + base * HD + cols, q0, mask=m0 & full)
    tl.store(Out + N + base * HD + cols, k0, mask=m0 & full)
    tl.store(Out + 2 * N + base * HD + cols, v0, mask=m0 & full)
    tl.store(Out + (base + 1) * HD + cols, q1, mask=m1 & full)
    tl.store(Out + N + (base + 1) * HD + cols, k1, mask=m1 & full)
    tl.store(Out + 2 * N + (base + 1) * HD + cols, v1, mask=m1 & full)
    tl.store(Out + (base + 2) * HD + cols, q2, mask=m2 & full)
    tl.store(Out + N + (base + 2) * HD + cols, k2, mask=m2 & full)
    tl.store(Out + 2 * N + (base + 2) * HD + cols, v2, mask=m2 & full)
    tl.store(Out + (base + 3) * HD + cols, q3, mask=m3 & full)
    tl.store(Out + N + (base + 3) * HD + cols, k3, mask=m3 & full)
    tl.store(Out + 2 * N + (base + 3) * HD + cols, v3, mask=m3 & full)


@triton.jit
def _pack_group8(
    Q,
    K,
    V,
    Index,
    Out,
    ROWS,
    N,
    TOKENS,
    HD: tl.constexpr,
):
    # Same contract as _pack_group4 with 8 rows per program (rows >= 64).
    ip = Index.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    cols = tl.arange(0, HD)
    full = cols < HD
    base = pid * 8
    t0 = tl.load(ip + base, mask=base < ROWS, other=0)
    t1 = tl.load(ip + base + 1, mask=base + 1 < ROWS, other=0)
    t2 = tl.load(ip + base + 2, mask=base + 2 < ROWS, other=0)
    t3 = tl.load(ip + base + 3, mask=base + 3 < ROWS, other=0)
    t4 = tl.load(ip + base + 4, mask=base + 4 < ROWS, other=0)
    t5 = tl.load(ip + base + 5, mask=base + 5 < ROWS, other=0)
    t6 = tl.load(ip + base + 6, mask=base + 6 < ROWS, other=0)
    t7 = tl.load(ip + base + 7, mask=base + 7 < ROWS, other=0)
    t0 = tl.where(t0 < 0, t0 + TOKENS, t0)
    t1 = tl.where(t1 < 0, t1 + TOKENS, t1)
    t2 = tl.where(t2 < 0, t2 + TOKENS, t2)
    t3 = tl.where(t3 < 0, t3 + TOKENS, t3)
    t4 = tl.where(t4 < 0, t4 + TOKENS, t4)
    t5 = tl.where(t5 < 0, t5 + TOKENS, t5)
    t6 = tl.where(t6 < 0, t6 + TOKENS, t6)
    t7 = tl.where(t7 < 0, t7 + TOKENS, t7)
    q0 = tl.load(Q + t0 * HD + cols)
    k0 = tl.load(K + t0 * HD + cols)
    v0 = tl.load(V + t0 * HD + cols)
    q1 = tl.load(Q + t1 * HD + cols)
    k1 = tl.load(K + t1 * HD + cols)
    v1 = tl.load(V + t1 * HD + cols)
    q2 = tl.load(Q + t2 * HD + cols)
    k2 = tl.load(K + t2 * HD + cols)
    v2 = tl.load(V + t2 * HD + cols)
    q3 = tl.load(Q + t3 * HD + cols)
    k3 = tl.load(K + t3 * HD + cols)
    v3 = tl.load(V + t3 * HD + cols)
    q4 = tl.load(Q + t4 * HD + cols)
    k4 = tl.load(K + t4 * HD + cols)
    v4 = tl.load(V + t4 * HD + cols)
    q5 = tl.load(Q + t5 * HD + cols)
    k5 = tl.load(K + t5 * HD + cols)
    v5 = tl.load(V + t5 * HD + cols)
    q6 = tl.load(Q + t6 * HD + cols)
    k6 = tl.load(K + t6 * HD + cols)
    v6 = tl.load(V + t6 * HD + cols)
    q7 = tl.load(Q + t7 * HD + cols)
    k7 = tl.load(K + t7 * HD + cols)
    v7 = tl.load(V + t7 * HD + cols)
    m0 = base < ROWS
    m1 = base + 1 < ROWS
    m2 = base + 2 < ROWS
    m3 = base + 3 < ROWS
    m4 = base + 4 < ROWS
    m5 = base + 5 < ROWS
    m6 = base + 6 < ROWS
    m7 = base + 7 < ROWS
    tl.store(Out + base * HD + cols, q0, mask=m0 & full)
    tl.store(Out + N + base * HD + cols, k0, mask=m0 & full)
    tl.store(Out + 2 * N + base * HD + cols, v0, mask=m0 & full)
    tl.store(Out + (base + 1) * HD + cols, q1, mask=m1 & full)
    tl.store(Out + N + (base + 1) * HD + cols, k1, mask=m1 & full)
    tl.store(Out + 2 * N + (base + 1) * HD + cols, v1, mask=m1 & full)
    tl.store(Out + (base + 2) * HD + cols, q2, mask=m2 & full)
    tl.store(Out + N + (base + 2) * HD + cols, k2, mask=m2 & full)
    tl.store(Out + 2 * N + (base + 2) * HD + cols, v2, mask=m2 & full)
    tl.store(Out + (base + 3) * HD + cols, q3, mask=m3 & full)
    tl.store(Out + N + (base + 3) * HD + cols, k3, mask=m3 & full)
    tl.store(Out + 2 * N + (base + 3) * HD + cols, v3, mask=m3 & full)
    tl.store(Out + (base + 4) * HD + cols, q4, mask=m4 & full)
    tl.store(Out + N + (base + 4) * HD + cols, k4, mask=m4 & full)
    tl.store(Out + 2 * N + (base + 4) * HD + cols, v4, mask=m4 & full)
    tl.store(Out + (base + 5) * HD + cols, q5, mask=m5 & full)
    tl.store(Out + N + (base + 5) * HD + cols, k5, mask=m5 & full)
    tl.store(Out + 2 * N + (base + 5) * HD + cols, v5, mask=m5 & full)
    tl.store(Out + (base + 6) * HD + cols, q6, mask=m6 & full)
    tl.store(Out + N + (base + 6) * HD + cols, k6, mask=m6 & full)
    tl.store(Out + 2 * N + (base + 6) * HD + cols, v6, mask=m6 & full)
    tl.store(Out + (base + 7) * HD + cols, q7, mask=m7 & full)
    tl.store(Out + N + (base + 7) * HD + cols, k7, mask=m7 & full)
    tl.store(Out + 2 * N + (base + 7) * HD + cols, v7, mask=m7 & full)


@triton.jit
def _pack_rows(
    Q,
    K,
    V,
    Index,
    Out,
    JOBS,
    N,
    TOKENS,
    HD,
    S,
    D,
    QS0,
    QS1,
    QS2,
    QS3,
    KS0,
    KS1,
    KS2,
    KS3,
    VS0,
    VS1,
    VS2,
    VS3,
    BLOCK: tl.constexpr,
    CB: tl.constexpr,
    FULL: tl.constexpr,
    STEPS,
    PROGRAMS,
):
    # 1-D per-row fallback (strided sources or non-power-of-two row width).
    # One job == one (row, column block); jobs beyond the grid.x cap fold
    # into a grid-stride loop. 2-D gather tiles (per-row token base
    # broadcast against a column vector) are avoided on purpose: they trip
    # the memref_ext.slice_start 7-vs-6 regression (compiler_issues/
    # sglang_fused_pack_qkv_slice_start.md); the 1-D per-row form is immune.
    # Index arrives as int32 or fake-int64; read through an int32 view.
    ip = Index.to(tl.pointer_type(tl.int32))
    start = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    for step in range(STEPS):
        pid = start + step * PROGRAMS
        if pid < JOBS:
            row = pid // CB
            cb = pid % CB
            col = cb * BLOCK + cols
            t = tl.load(ip + row)
            t = tl.where(t < 0, t + TOKENS, t)
            batch = t // S
            pos = t - batch * S
            head = col // D
            dim = col - head * D
            qb = batch * QS0 + pos * QS1
            kb = batch * KS0 + pos * KS1
            vb = batch * VS0 + pos * VS1
            if FULL:
                q = tl.load(Q + qb + head * QS2 + dim * QS3)
                k = tl.load(K + kb + head * KS2 + dim * KS3)
                v = tl.load(V + vb + head * VS2 + dim * VS3)
                dst = row * HD + col
                tl.store(Out + dst, q)
                tl.store(Out + N + dst, k)
                tl.store(Out + 2 * N + dst, v)
            else:
                cmask = col < HD
                q = tl.load(
                    Q + qb + head * QS2 + dim * QS3, mask=cmask, other=0
                )
                k = tl.load(
                    K + kb + head * KS2 + dim * KS3, mask=cmask, other=0
                )
                v = tl.load(
                    V + vb + head * VS2 + dim * VS3, mask=cmask, other=0
                )
                dst = row * HD + col
                tl.store(Out + dst, q, mask=cmask)
                tl.store(Out + N + dst, k, mask=cmask)
                tl.store(Out + 2 * N + dst, v, mask=cmask)


def fused_pack_qkv(q, k, v, indices):
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("q, k, v must have the same [B,S,H,D] shape")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, v must have the same dtype")
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be a one-dimensional int32/int64 tensor")
    if any(t.device != q.device for t in (k, v, indices)):
        raise ValueError("all inputs must be on the same device")
    if indices.dtype == torch.int64:
        # torch_gcu replaces Long storage with Int ("GCU not support Long use
        # Int replace"): an int64 load then reads interleaved garbage
        # (physically 4-byte elements at an 8-byte stride) and the bogus
        # token ids fault the device (Sip error -> SIGABRT at sync).
        # Without the int64 env the tensor is therefore just int64 metadata
        # over int32 storage: a contiguous, offset-0 tensor is handed to the
        # kernel unchanged and read through an int32 pointer view (zero
        # cast). True 8-byte storage (env on) or strided/offset views keep
        # the host-side cast, whose output is always physical int32.
        # Token ids live in [-bs*seq, bs*seq), exactly representable in
        # int32 (ATen cast verified exact).
        if (
            _int64_env_enabled()
            or not indices.is_contiguous()
            or indices.storage_offset() != 0
        ):
            indices = indices.to(torch.int32)
    rows = indices.numel()
    heads, head_dim = q.shape[2], q.shape[3]
    row_width = heads * head_dim
    storage = torch.empty(
        (3, rows, heads, head_dim), dtype=q.dtype, device=q.device
    )
    outputs = storage.unbind(0)
    n = rows * row_width
    if n == 0:
        return outputs
    tokens = q.shape[0] * q.shape[1]
    # int32 addressing domain: source offsets stay below q.numel(),
    # destination offsets below 3*n (plus tail-group dead-lane slack of at
    # most U*HD <= 32768), and the negative-index wrap add stays below 2**31.
    i32 = (
        q.numel() < 2**31
        and 3 * n + 32768 < 2**31
        and tokens < 2**30
        and (rows == 1 or indices.stride(0) == 1)
    )
    if i32:
        if (
            row_width <= 4096
            and row_width == triton.next_power_of_2(row_width)
            and q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
        ):
            unit = 8 if rows >= 64 and row_width <= 2048 else 4
            if triton.cdiv(rows, unit) <= _GRID_X_MAX:
                grid = (triton.cdiv(rows, unit),)
                if unit == 8:
                    _pack_group8[grid](
                        q,
                        k,
                        v,
                        indices,
                        storage,
                        rows,
                        n,
                        tokens,
                        HD=row_width,
                        num_warps=2,
                    )
                else:
                    _pack_group4[grid](
                        q,
                        k,
                        v,
                        indices,
                        storage,
                        rows,
                        n,
                        tokens,
                        HD=row_width,
                        num_warps=2,
                    )
                return outputs
        block = min(triton.next_power_of_2(row_width), 4096)
        col_blocks = triton.cdiv(row_width, block)
        jobs = rows * col_blocks
        launched = min(jobs, _GRID_X_MAX)
        _pack_rows[(launched,)](
            q,
            k,
            v,
            indices,
            storage,
            jobs,
            n,
            tokens,
            row_width,
            q.shape[1],
            head_dim,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            BLOCK=block,
            CB=col_blocks,
            FULL=row_width % block == 0,
            STEPS=triton.cdiv(jobs, launched),
            PROGRAMS=launched,
            num_warps=2 if block <= 2048 else 4,
        )
        return outputs
    block = max(
        256 if n <= 1024 else 1024,
        triton.next_power_of_2(triton.cdiv(n, _GRID_X_MAX)),
    )
    _pack[(triton.cdiv(n, block),)](
        q,
        k,
        v,
        indices,
        storage,
        n,
        q.shape[0] * q.shape[1],
        q.shape[1],
        q.shape[2],
        q.shape[3],
        q.stride(),
        k.stride(),
        v.stride(),
        indices.stride(0),
        block,
    )
    return outputs


__all__ = ["fused_pack_qkv"]
