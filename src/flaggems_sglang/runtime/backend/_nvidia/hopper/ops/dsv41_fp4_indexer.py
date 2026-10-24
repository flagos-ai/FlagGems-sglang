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

# Portions adapted from SGLang (Copyright 2023-2026 SGLang Team),
# including the validated CherryLemon DeepSeek V4.1 H100 serving fork.
# See docs/dsv41-h100-operators.md for source identities and numerical scope.
"""Decode-time low-ratio indexer logits for Hopper.

DeepGEMM's fp8_fp4 mqa-logits kernels need SM100/SM120. This Triton kernel covers
the same decode step on SM90: one query token per request, scored against the
request's visible compressed positions read straight out of the fp4 indexer
pool (e2m1 payload + e8m0 per-32 block scales, page layout of
store_fp4_index_k_cache), summed over heads with relu and the per-head weights.

Numerics follow the torch reference path: bf16 dot, bf16 relu/weight product,
bf16 head reduction, fp32 logits out.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl

_e2m1_decode = None
fp4_weighted_head_score = None


def _load_fp4_math():
    # Keep the framework package importable without its optional dsv41 extra.
    # Resolve common JIT helpers before compiling/capturing the fused kernel.
    global _e2m1_decode, fp4_weighted_head_score
    if _e2m1_decode is None:
        from flag_gems.fused.dsv41 import fp4_indexer

        if fp4_indexer.FP4_INDEX_SCORE_ABI != 1:
            raise RuntimeError("incompatible FlagGems FP4 index-score ABI")
        _e2m1_decode = fp4_indexer._e2m1_decode
        fp4_weighted_head_score = fp4_indexer.fp4_weighted_head_score


INDEX_HEAD_DIM = 128
PAYLOAD_BYTES = tl.constexpr(64)
SCALE_BYTES = tl.constexpr(4)


@triton.jit
def _fp4_index_logits_kernel(
    q_ptr,  # [B, H, D] bf16, fq4 queries (already rope'd)
    w_ptr,  # [B, H] bf16 head weights (softmax scale folded in)
    slots_ptr,  # [B, L] pool slots per (request, compressed position)
    req_to_token_ptr,  # [num_reqs, max_context_len] full-token pool slots
    req_ptr,  # [B] request-pool row for each query
    candidate_blocks_ptr,
    lens_ptr,  # [B] int64 visible compressed positions per request
    table_ptr,  # [num_pages, page_size * 64 + page_size * 4] uint8
    out_ptr,  # [B, L] fp32 logits, -inf beyond lens
    candidate_scores_ptr,
    candidate_lens_ptr,
    L,
    page_size,
    row_stride,
    stride_qb,
    stride_qh,
    stride_wb,
    stride_req,
    candidate_block_stride,
    stride_out,
    candidate_score_stride,
    H: tl.constexpr,
    HALF_D: tl.constexpr,  # D // 2 == 64 nibble-pairs per row
    BLOCK_L: tl.constexpr,
    RATIO: tl.constexpr,
    USE_REQ_TO_TOKEN: tl.constexpr,
    USE_CANDIDATE_BLOCKS: tl.constexpr,
    CANDIDATE_BLOCK_SIZE: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
):
    b = tl.program_id(0)
    lb = tl.program_id(1)
    offs_l = lb * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_h = tl.arange(0, H)
    offs_i = tl.arange(
        0, HALF_D
    )  # byte index i holds elements 2i (low nibble), 2i+1 (high nibble)

    n_vis = tl.load(lens_ptr + b)
    if lb * BLOCK_L >= tl.minimum(n_vis, L):
        if WRITE_LOGITS:
            tl.store(
                out_ptr + b * stride_out + offs_l,
                -float("inf"),
                offs_l < L,
            )
        if WRITE_CANDIDATES:
            nblocks = (L + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
            block_ids = lb * (BLOCK_L // CANDIDATE_BLOCK_SIZE) + tl.arange(
                0, BLOCK_L // CANDIDATE_BLOCK_SIZE
            )
            tl.store(
                candidate_scores_ptr + b * candidate_score_stride + block_ids,
                -float("inf"),
                block_ids < nblocks,
            )
            tl.store(candidate_lens_ptr + b, 0, mask=lb == 0)
        return
    valid = offs_l < tl.minimum(n_vis, L)
    if USE_CANDIDATE_BLOCKS:
        block_col = offs_l // CANDIDATE_BLOCK_SIZE
        within = offs_l % CANDIDATE_BLOCK_SIZE
        block = tl.load(
            candidate_blocks_ptr + b * candidate_block_stride + block_col,
            mask=valid,
            other=0,
        )
        logical_position = block.to(tl.int64) * CANDIDATE_BLOCK_SIZE + within
        req = tl.load(req_ptr + b).to(tl.int64)
        slot = tl.load(
            req_to_token_ptr + req * stride_req + logical_position * RATIO,
            mask=valid,
            other=0,
        ).to(tl.int64)
        slot = slot // RATIO
    elif USE_REQ_TO_TOKEN:
        req = tl.load(req_ptr + b).to(tl.int64)
        slot = tl.load(
            req_to_token_ptr + req * stride_req + offs_l.to(tl.int64) * RATIO,
            mask=valid,
            other=0,
        ).to(tl.int64)
        slot = slot // RATIO
    else:
        slot = tl.load(
            slots_ptr + b * L + offs_l, mask=offs_l < L, other=0
        ).to(tl.int64)
    page = slot // page_size
    off = slot % page_size
    row_base = page * row_stride

    # K payload: [BLOCK_L, HALF_D] uint8
    pay = tl.load(
        table_ptr
        + row_base[:, None]
        + off[:, None] * PAYLOAD_BYTES
        + offs_i[None, :],
        mask=valid[:, None],
        other=0,
    )
    low = _e2m1_decode(pay & 0x0F)
    high = _e2m1_decode((pay >> 4) & 0x0F)
    # e8m0 block scales: element j uses block j // 32 -> byte i uses block i // 16.
    sc_idx = offs_i // 16
    exps = tl.load(
        table_ptr
        + row_base[:, None]
        + page_size * PAYLOAD_BYTES
        + off[:, None] * SCALE_BYTES
        + sc_idx[None, :],
        mask=valid[:, None],
        other=127,
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    k_low = (low * scale).to(tl.bfloat16)  # [BLOCK_L, HALF_D] elements 2i
    k_high = (high * scale).to(tl.bfloat16)  # elements 2i+1

    # queries: even / odd elements, [H, HALF_D] bf16
    q_even = tl.load(
        q_ptr
        + b * stride_qb
        + offs_h[:, None] * stride_qh
        + 2 * offs_i[None, :]
    )
    q_odd = tl.load(
        q_ptr
        + b * stride_qb
        + offs_h[:, None] * stride_qh
        + 2 * offs_i[None, :]
        + 1
    )

    w = tl.load(w_ptr + b * stride_wb + offs_h).to(tl.float32)
    logit = fp4_weighted_head_score(q_even, q_odd, k_low, k_high, w)
    logit = tl.where(valid, logit, float("-inf"))
    if WRITE_LOGITS:
        tl.store(out_ptr + b * stride_out + offs_l, logit, mask=offs_l < L)
    if WRITE_CANDIDATES:
        blocks_per_tile: tl.constexpr = BLOCK_L // CANDIDATE_BLOCK_SIZE
        block_scores = tl.reshape(
            logit, (blocks_per_tile, CANDIDATE_BLOCK_SIZE)
        )
        block_scores = tl.max(block_scores, axis=1)
        block_ids = lb * blocks_per_tile + tl.arange(0, blocks_per_tile)
        last_block = (n_vis - 1) // CANDIDATE_BLOCK_SIZE
        block_scores = tl.where(
            (n_vis > 0) & (block_ids == last_block), float("inf"), block_scores
        )
        num_blocks = (L + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
        tl.store(
            candidate_scores_ptr + b * candidate_score_stride + block_ids,
            block_scores,
            mask=block_ids < num_blocks,
        )
        tl.store(
            candidate_lens_ptr + b,
            (n_vis + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE,
            mask=lb == 0,
        )


# Each CTA shares a request's decoded K tile across six verify queries. Keep
# each query's original 32-head MMA and reduction layout: increasing the MMA M
# dimension can change floating-point reduction order. Request equality is
# checked on device on every replay; visible lengths and source writes remain
# per query. Compact candidate lists deliberately use the original kernel.
@triton.jit
def _fp4_group_load_keys(
    RT,
    TABLE,
    req,
    nvis,
    L,
    STRIDE_REQ,
    ROW_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    BL: tl.constexpr,
):
    l = tl.program_id(1) * BL + tl.arange(0, BL)  # noqa: E741
    i = tl.arange(0, 64)
    valid = l < tl.minimum(nvis, L)
    slot = (
        tl.load(
            RT + req.to(tl.int64) * STRIDE_REQ + l.to(tl.int64) * RATIO,
            valid,
            other=0,
        ).to(tl.int64)
        // RATIO
    )
    page = slot // PAGE
    off = slot % PAGE
    row = page * ROW_STRIDE
    pay = tl.load(
        TABLE + row[:, None] + off[:, None] * 64 + i[None, :],
        valid[:, None],
        other=0,
    )
    low = _e2m1_decode(pay & 15)
    high = _e2m1_decode(pay >> 4 & 15)
    exps = tl.load(
        TABLE
        + row[:, None]
        + PAGE * 64
        + off[:, None] * 4
        + (i // 16)[None, :],
        valid[:, None],
        other=127,
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    return ((low * scale).to(tl.bfloat16), (high * scale).to(tl.bfloat16))


@triton.jit
def _fp4_group_invalid(
    OUT,
    CS,
    CL,
    b,
    L,
    SO,
    SCS,
    BL: tl.constexpr,
    CB: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
):
    lb = tl.program_id(1)
    l = lb * BL + tl.arange(0, BL)  # noqa: E741
    if WRITE_LOGITS:
        tl.store(OUT + b * SO + l, -float("inf"), l < L)
    if WRITE_CANDIDATES:
        ids = lb * (BL // CB) + tl.arange(0, BL // CB)
        tl.store(CS + b * SCS + ids, -float("inf"), ids < tl.cdiv(L, CB))
        tl.store(CL + b, 0, lb == 0)


@triton.jit
def _fp4_group_score(
    Q,
    W,
    OUT,
    CS,
    CL,
    KL,
    KH,
    b,
    nvis,
    L,
    SQB,
    SQH,
    SWB,
    SO,
    SCS,
    BL: tl.constexpr,
    CB: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
):
    h = tl.arange(0, 32)
    i = tl.arange(0, 64)
    lb = tl.program_id(1)
    l = lb * BL + tl.arange(0, BL)  # noqa: E741
    qe = tl.load(Q + b * SQB + h[:, None] * SQH + 2 * i[None, :])
    qo = tl.load(Q + b * SQB + h[:, None] * SQH + 2 * i[None, :] + 1)
    w = tl.load(W + b * SWB + h).to(tl.float32)
    logit = fp4_weighted_head_score(qe, qo, KL, KH, w)
    logit = tl.where(l < tl.minimum(nvis, L), logit, -float("inf"))
    if WRITE_LOGITS:
        tl.store(OUT + b * SO + l, logit, l < L)
    if WRITE_CANDIDATES:
        scores = tl.max(tl.reshape(logit, (BL // CB, CB)), 1)
        ids = lb * (BL // CB) + tl.arange(0, BL // CB)
        scores = tl.where(
            (nvis > 0) & (ids == (nvis - 1) // CB), float("inf"), scores
        )
        tl.store(CS + b * SCS + ids, scores, ids < tl.cdiv(L, CB))
        tl.store(CL + b, tl.cdiv(nvis, CB), lb == 0)


@triton.jit
def _fp4_index_logits_grouped_kernel(
    Q,
    W,
    RT,
    REQ,
    LENS,
    TABLE,
    OUT,
    CS,
    CL,
    B,
    L,
    PAGE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    SQB,
    SQH,
    SWB,
    SR,
    SO,
    SCS,
    GROUP: tl.constexpr,
    PGROUP: tl.constexpr,
    RATIO: tl.constexpr,
    BL: tl.constexpr = 64,
    CB: tl.constexpr = 8,
    WRITE_LOGITS: tl.constexpr = True,
    WRITE_CANDIDATES: tl.constexpr = False,
):
    first = tl.program_id(0) * GROUP
    lb = tl.program_id(1)
    offset = tl.arange(0, PGROUP)
    rows = first + offset
    mask = (offset < GROUP) & (rows < B)
    lens = tl.load(LENS + rows, mask, other=0)
    maxlens = tl.max(lens, 0)
    if lb * BL >= tl.minimum(maxlens, L):
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < B:
                _fp4_group_invalid(
                    OUT,
                    CS,
                    CL,
                    b,
                    L,
                    SO,
                    SCS,
                    BL,
                    CB,
                    WRITE_LOGITS,
                    WRITE_CANDIDATES,
                )
        return
    req0 = tl.load(REQ + first)
    reqs = tl.load(REQ + rows, mask, other=req0)
    same = tl.sum((reqs != req0).to(tl.int32), 0) == 0
    if same:
        kl, kh = _fp4_group_load_keys(
            RT, TABLE, req0, maxlens, L, SR, ROW_STRIDE, PAGE, RATIO, BL
        )
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < B:
                nvis = tl.load(LENS + b)
                _fp4_group_score(
                    Q,
                    W,
                    OUT,
                    CS,
                    CL,
                    kl,
                    kh,
                    b,
                    nvis,
                    L,
                    SQB,
                    SQH,
                    SWB,
                    SO,
                    SCS,
                    BL,
                    CB,
                    WRITE_LOGITS,
                    WRITE_CANDIDATES,
                )
    else:
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < B:
                nvis = tl.load(LENS + b)
                if lb * BL < tl.minimum(nvis, L):
                    req = tl.load(REQ + b)
                    kl, kh = _fp4_group_load_keys(
                        RT,
                        TABLE,
                        req,
                        nvis,
                        L,
                        SR,
                        ROW_STRIDE,
                        PAGE,
                        RATIO,
                        BL,
                    )
                    _fp4_group_score(
                        Q,
                        W,
                        OUT,
                        CS,
                        CL,
                        kl,
                        kh,
                        b,
                        nvis,
                        L,
                        SQB,
                        SQH,
                        SWB,
                        SO,
                        SCS,
                        BL,
                        CB,
                        WRITE_LOGITS,
                        WRITE_CANDIDATES,
                    )
                else:
                    _fp4_group_invalid(
                        OUT,
                        CS,
                        CL,
                        b,
                        L,
                        SO,
                        SCS,
                        BL,
                        CB,
                        WRITE_LOGITS,
                        WRITE_CANDIDATES,
                    )


@triton.jit
def _fp4_group_load_keys_persistent_tile(
    TILE,
    RT,
    TABLE,
    req,
    nvis,
    L,
    STRIDE_REQ,
    ROW_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    RATIO: tl.constexpr,
    BL: tl.constexpr,
):
    l = TILE * BL + tl.arange(0, BL)  # noqa: E741
    i = tl.arange(0, 64)
    valid = l < tl.minimum(nvis, L)
    slot = (
        tl.load(
            RT + req.to(tl.int64) * STRIDE_REQ + l.to(tl.int64) * RATIO,
            valid,
            other=0,
        ).to(tl.int64)
        // RATIO
    )
    page = slot // PAGE
    off = slot % PAGE
    row = page * ROW_STRIDE
    pay = tl.load(
        TABLE + row[:, None] + off[:, None] * 64 + i[None, :],
        valid[:, None],
        other=0,
    )
    low = _e2m1_decode(pay & 15)
    high = _e2m1_decode(pay >> 4 & 15)
    exps = tl.load(
        TABLE
        + row[:, None]
        + PAGE * 64
        + off[:, None] * 4
        + (i // 16)[None, :],
        valid[:, None],
        other=127,
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    return ((low * scale).to(tl.bfloat16), (high * scale).to(tl.bfloat16))


@triton.jit
def _fp4_group_invalid_persistent_tile(
    TILE,
    OUT,
    CS,
    CL,
    b,
    L,
    SO,
    SCS,
    BL: tl.constexpr,
    CB: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
):
    lb = TILE
    l = lb * BL + tl.arange(0, BL)  # noqa: E741
    if WRITE_LOGITS:
        tl.store(OUT + b * SO + l, -float("inf"), l < L)
    if WRITE_CANDIDATES:
        ids = lb * (BL // CB) + tl.arange(0, BL // CB)
        tl.store(CS + b * SCS + ids, -float("inf"), ids < tl.cdiv(L, CB))
        tl.store(CL + b, 0, lb == 0)


@triton.jit
def _fp4_group_score_persistent_tile(
    TILE,
    Q,
    W,
    OUT,
    CS,
    CL,
    KL,
    KH,
    b,
    nvis,
    L,
    SQB,
    SQH,
    SWB,
    SO,
    SCS,
    BL: tl.constexpr,
    CB: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
):
    h = tl.arange(0, 32)
    i = tl.arange(0, 64)
    lb = TILE
    l = lb * BL + tl.arange(0, BL)  # noqa: E741
    qe = tl.load(Q + b * SQB + h[:, None] * SQH + 2 * i[None, :])
    qo = tl.load(Q + b * SQB + h[:, None] * SQH + 2 * i[None, :] + 1)
    w = tl.load(W + b * SWB + h).to(tl.float32)
    logit = fp4_weighted_head_score(qe, qo, KL, KH, w)
    logit = tl.where(l < tl.minimum(nvis, L), logit, -float("inf"))
    if WRITE_LOGITS:
        tl.store(OUT + b * SO + l, logit, l < L)
    if WRITE_CANDIDATES:
        scores = tl.max(tl.reshape(logit, (BL // CB, CB)), 1)
        ids = lb * (BL // CB) + tl.arange(0, BL // CB)
        scores = tl.where(
            (nvis > 0) & (ids == (nvis - 1) // CB), float("inf"), scores
        )
        tl.store(CS + b * SCS + ids, scores, ids < tl.cdiv(L, CB))
        tl.store(CL + b, tl.cdiv(nvis, CB), lb == 0)


@triton.jit
def _fp4_index_logits_grouped_kernel_persistent_tile(
    TILE,
    Q,
    W,
    RT,
    REQ,
    LENS,
    TABLE,
    OUT,
    CS,
    CL,
    B,
    L,
    PAGE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    SQB,
    SQH,
    SWB,
    SR,
    SO,
    SCS,
    GROUP: tl.constexpr,
    PGROUP: tl.constexpr,
    RATIO: tl.constexpr,
    BL: tl.constexpr = 64,
    CB: tl.constexpr = 8,
    WRITE_LOGITS: tl.constexpr = True,
    WRITE_CANDIDATES: tl.constexpr = False,
):
    first = tl.program_id(0) * GROUP
    lb = TILE
    offset = tl.arange(0, PGROUP)
    rows = first + offset
    mask = (offset < GROUP) & (rows < B)
    lens = tl.load(LENS + rows, mask, other=0)
    maxlens = tl.max(lens, 0)
    if lb * BL >= tl.minimum(maxlens, L):
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < B:
                _fp4_group_invalid_persistent_tile(
                    lb,
                    OUT,
                    CS,
                    CL,
                    b,
                    L,
                    SO,
                    SCS,
                    BL,
                    CB,
                    WRITE_LOGITS,
                    WRITE_CANDIDATES,
                )
        return
    req0 = tl.load(REQ + first)
    reqs = tl.load(REQ + rows, mask, other=req0)
    same = tl.sum((reqs != req0).to(tl.int32), 0) == 0
    if same:
        kl, kh = _fp4_group_load_keys_persistent_tile(
            lb, RT, TABLE, req0, maxlens, L, SR, ROW_STRIDE, PAGE, RATIO, BL
        )
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < B:
                nvis = tl.load(LENS + b)
                _fp4_group_score_persistent_tile(
                    lb,
                    Q,
                    W,
                    OUT,
                    CS,
                    CL,
                    kl,
                    kh,
                    b,
                    nvis,
                    L,
                    SQB,
                    SQH,
                    SWB,
                    SO,
                    SCS,
                    BL,
                    CB,
                    WRITE_LOGITS,
                    WRITE_CANDIDATES,
                )
    else:
        for j in tl.range(0, GROUP, loop_unroll_factor=1):
            b = first + j
            if b < B:
                nvis = tl.load(LENS + b)
                if lb * BL < tl.minimum(nvis, L):
                    req = tl.load(REQ + b)
                    kl, kh = _fp4_group_load_keys_persistent_tile(
                        lb,
                        RT,
                        TABLE,
                        req,
                        nvis,
                        L,
                        SR,
                        ROW_STRIDE,
                        PAGE,
                        RATIO,
                        BL,
                    )
                    _fp4_group_score_persistent_tile(
                        lb,
                        Q,
                        W,
                        OUT,
                        CS,
                        CL,
                        kl,
                        kh,
                        b,
                        nvis,
                        L,
                        SQB,
                        SQH,
                        SWB,
                        SO,
                        SCS,
                        BL,
                        CB,
                        WRITE_LOGITS,
                        WRITE_CANDIDATES,
                    )
                else:
                    _fp4_group_invalid_persistent_tile(
                        lb,
                        OUT,
                        CS,
                        CL,
                        b,
                        L,
                        SO,
                        SCS,
                        BL,
                        CB,
                        WRITE_LOGITS,
                        WRITE_CANDIDATES,
                    )


@triton.jit
def _fp4_persistent_initialize(
    OUT,
    CS,
    CL,
    LENS,
    B,
    SO,
    SCS,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
    CB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if WRITE_LOGITS:
        tl.store(OUT + offset, -float("inf"), offset < B * SO)
    if WRITE_CANDIDATES:
        tl.store(CS + offset, -float("inf"), offset < B * SCS)
        nvis = tl.load(LENS + offset, offset < B, other=0)
        tl.store(CL + offset, tl.cdiv(tl.maximum(nvis, 0), CB), offset < B)


@triton.jit
def _fp4_index_logits_grouped_persistent_kernel(
    Q,
    W,
    RT,
    REQ,
    LENS,
    TABLE,
    OUT,
    CS,
    CL,
    B,
    L,
    PAGE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    SQB,
    SQH,
    SWB,
    SR,
    SO,
    SCS,
    GROUP: tl.constexpr,
    PGROUP: tl.constexpr,
    RATIO: tl.constexpr,
    BL: tl.constexpr,
    CB: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
):
    first = tl.program_id(0) * GROUP
    offset = tl.arange(0, PGROUP)
    rows = first + offset
    nvis = tl.load(LENS + rows, (offset < GROUP) & (rows < B), other=0)
    limit = tl.cdiv(tl.minimum(tl.maximum(tl.max(nvis, 0), 0), L), BL)
    for tile in range(tl.program_id(1), limit, tl.num_programs(1)):
        _fp4_index_logits_grouped_kernel_persistent_tile(
            tile,
            Q,
            W,
            RT,
            REQ,
            LENS,
            TABLE,
            OUT,
            CS,
            CL,
            B,
            L,
            PAGE,
            ROW_STRIDE,
            SQB,
            SQH,
            SWB,
            SR,
            SO,
            SCS,
            GROUP,
            PGROUP,
            RATIO,
            BL,
            CB,
            WRITE_LOGITS,
            WRITE_CANDIDATES,
        )


def _use_native_fp8(
    q,
    page_size,
    ratio,
    width,
    candidate_block_size,
    write_logits,
    query_group_size,
):
    # These are the measured capacity/batch buckets, not a density assumption.
    # Runtime Q/K eligibility, requests and lengths are checked on the device.
    if (
        query_group_size != 6
        or tuple(q.shape[1:]) != (32, 128)
        or page_size != 64
        or triton.__version__.split("+")[0] not in ("3.6.0", "3.7.1")
        or torch.cuda.get_device_capability(q.device) != (9, 0)
    ):
        return False
    b = q.shape[0]
    return (
        ratio == 1
        and width == 655627
        and candidate_block_size == 8
        and not write_logits
        and b in (6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 192)
    ) or (
        ratio == 2
        and width == 327813
        and candidate_block_size == 0
        and write_logits
        and b in (6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 192)
    )


def _use_tle_native_fp8(batch, ratio):
    # B6/r2's approximately 1% result was not stable under eight paired runs.
    # This is a measured bucket dispatch, after the semantic/grouping gate.
    return (
        triton.__version__.split("+")[0] == "3.6.0"
        and batch in (6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 192)
        and ratio in (1, 2)
        and (batch, ratio) not in ((6, 1), (6, 2), (12, 1))
    )


@lru_cache(maxsize=1)
def _tle_native_fp8():
    # Resolve optional TLE/helper availability once, before Graph capture.
    # Upstream Triton 3.6 and older FlagGems installations retain Gluon.
    try:
        from .dsv41_fp4_tle_indexer import fp4_index_logits_req_to_token
    except ImportError:
        return None
    return fp4_index_logits_req_to_token


def fp4_index_logits_req_to_token(
    q: torch.Tensor,
    weights: torch.Tensor,
    req_to_token: torch.Tensor,
    req: torch.Tensor,
    lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    ratio: int,
    width: int,
    candidate_block_size: int = 0,
    write_logits: bool = True,
    query_group_size: int = 1,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor]
):
    """Score logical compressed positions without materializing their pool slots."""
    _load_fp4_math()
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    B, H, _ = q.shape
    assert req.shape == lens.shape == (B,)
    assert req_to_token.dim() == 2 and req_to_token.stride(1) == 1
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert ratio in (1, 2)
    if (
        table.stride(1) == 1
        and table.stride(0) % 2 == 0
        and table.data_ptr() % 2 == 0
        and _use_native_fp8(
            q,
            page_size,
            ratio,
            width,
            candidate_block_size,
            write_logits,
            query_group_size,
        )
    ):
        # A missing new optional helper preserves the pinned BF16 installation.
        try:
            from flag_gems.fused.dsv41.fp4_fp8 import FP4_FP8_SCORE_ABI
        except ImportError:
            FP4_FP8_SCORE_ABI = 0
        if FP4_FP8_SCORE_ABI == 2:
            native_fp8 = (
                _tle_native_fp8() if _use_tle_native_fp8(B, ratio) else None
            )
            if native_fp8 is None:
                from .dsv41_fp4_fp8_indexer import (
                    fp4_index_logits_req_to_token as native_fp8,
                )

            return native_fp8(
                q,
                weights,
                req_to_token,
                req,
                lens,
                table,
                page_size,
                ratio,
                width,
                candidate_block_size,
                write_logits,
                query_group_size,
            )
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    req = req.to(torch.int64).contiguous()
    lens = lens.to(torch.int64).contiguous()
    output_width = triton.cdiv(width, 4) * 4 if write_logits else 4
    out_storage = torch.empty(
        (B, output_width), dtype=torch.float32, device=q.device
    )
    out = out_storage[:, :width]
    block_l = 64
    if candidate_block_size:
        assert block_l % candidate_block_size == 0
        num_blocks = triton.cdiv(width, candidate_block_size)
        candidate_storage = torch.empty(
            (B, triton.cdiv(num_blocks, 4) * 4),
            dtype=torch.float32,
            device=q.device,
        )
        candidate_scores = candidate_storage[:, :num_blocks]
        candidate_lens = torch.empty(B, dtype=torch.int32, device=q.device)
    else:
        candidate_scores = out
        candidate_lens = lens
    if width == 0:
        if candidate_block_size:
            return out, candidate_scores, candidate_lens
        return out
    # The caller supplies a semantic target-verify hint, never a guess from B.
    # Unsupported heads, block layouts, and partial groups retain the old grid.
    use_middle = (
        query_group_size == 6
        and B in (6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96)
        and H == 32
    )
    use_persistent = (
        use_middle
        and candidate_block_size == 8
        and not write_logits
        and width >= 65536
    )
    use_grouped = (
        query_group_size == 6
        and (B in (120, 192) or (use_middle and candidate_block_size == 0))
        and H == 32
        and candidate_block_size in (0, 8)
    )
    if use_persistent:
        block_l = 128
        init_count = max(
            B * out.stride(0) if write_logits else 0,
            B * candidate_scores.stride(0) if candidate_block_size else 0,
            B if candidate_block_size else 0,
        )
        _fp4_persistent_initialize[(triton.cdiv(init_count, 1024),)](
            out,
            candidate_scores,
            candidate_lens,
            lens,
            B,
            out.stride(0),
            candidate_scores.stride(0),
            write_logits,
            bool(candidate_block_size),
            candidate_block_size or 1,
            1024,
            num_warps=4,
        )
        _fp4_index_logits_grouped_persistent_kernel[
            (B // 6, min(512, triton.cdiv(width, block_l)))
        ](
            q,
            weights,
            req_to_token,
            req,
            lens,
            table,
            out,
            candidate_scores,
            candidate_lens,
            B,
            width,
            page_size,
            table.stride(0),
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            req_to_token.stride(0),
            out.stride(0),
            candidate_scores.stride(0),
            GROUP=6,
            PGROUP=8,
            RATIO=ratio,
            BL=block_l,
            CB=candidate_block_size or 1,
            WRITE_LOGITS=write_logits,
            WRITE_CANDIDATES=bool(candidate_block_size),
            num_warps=4,
            maxnreg=160 if B >= 18 else None,
        )
    elif use_grouped:
        if use_middle:
            block_l = 128
        _fp4_index_logits_grouped_kernel[
            (B // 6, triton.cdiv(width, block_l))
        ](
            q,
            weights,
            req_to_token,
            req,
            lens,
            table,
            out,
            candidate_scores,
            candidate_lens,
            B,
            width,
            page_size,
            table.stride(0),
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            req_to_token.stride(0),
            out.stride(0),
            candidate_scores.stride(0),
            GROUP=6,
            PGROUP=8,
            RATIO=ratio,
            BL=block_l,
            CB=candidate_block_size or 1,
            WRITE_LOGITS=write_logits,
            WRITE_CANDIDATES=bool(candidate_block_size),
            num_warps=4,
            maxnreg=160 if use_middle and B >= 12 else None,
        )
    else:
        grid = (B, triton.cdiv(width, block_l))
        _fp4_index_logits_kernel[grid](
            q,
            weights,
            req_to_token,
            req_to_token,
            req,
            req_to_token,
            lens,
            table,
            out,
            candidate_scores,
            candidate_lens,
            width,
            page_size,
            table.stride(0),
            q.stride(0),
            q.stride(1),
            weights.stride(0),
            req_to_token.stride(0),
            req_to_token.stride(0),
            out.stride(0),
            candidate_scores.stride(0),
            H=H,
            HALF_D=INDEX_HEAD_DIM // 2,
            BLOCK_L=block_l,
            RATIO=ratio,
            USE_REQ_TO_TOKEN=True,
            USE_CANDIDATE_BLOCKS=False,
            CANDIDATE_BLOCK_SIZE=candidate_block_size or 1,
            WRITE_CANDIDATES=bool(candidate_block_size),
            WRITE_LOGITS=write_logits,
            num_warps=4,
        )
    if not write_logits:
        assert candidate_block_size
        return candidate_scores, candidate_lens
    if candidate_block_size:
        return out, candidate_scores, candidate_lens
    return out


def fp4_index_logits_candidate_blocks(
    q: torch.Tensor,
    weights: torch.Tensor,
    req_to_token: torch.Tensor,
    req: torch.Tensor,
    candidate_blocks: torch.Tensor,
    candidate_lens: torch.Tensor,
    table: torch.Tensor,
    page_size: int,
    ratio: int,
    candidate_block_size: int,
) -> torch.Tensor:
    """Score compact candidate blocks while resolving physical slots in-kernel."""
    _load_fp4_math()
    assert q.dtype == torch.bfloat16 and q.shape[-1] == INDEX_HEAD_DIM
    batch, heads, _ = q.shape
    assert req.shape == candidate_lens.shape == (batch,)
    assert candidate_blocks.shape[0] == batch
    assert req_to_token.dim() == 2 and req_to_token.stride(1) == 1
    assert table.dtype == torch.uint8 and table.dim() == 2
    assert ratio in (1, 2)
    q = q.contiguous()
    weights = weights.to(torch.bfloat16).contiguous()
    req = req.to(torch.int64).contiguous()
    candidate_lens = candidate_lens.to(torch.int32).contiguous()
    candidate_blocks = candidate_blocks.to(torch.int32).contiguous()
    width = candidate_blocks.shape[1] * candidate_block_size
    out_storage = torch.empty(
        (batch, triton.cdiv(width, 4) * 4),
        dtype=torch.float32,
        device=q.device,
    )
    out = out_storage[:, :width]
    if width == 0:
        return out
    block_l = 64
    _fp4_index_logits_kernel[(batch, triton.cdiv(width, block_l))](
        q,
        weights,
        candidate_blocks,
        req_to_token,
        req,
        candidate_blocks,
        candidate_lens,
        table,
        out,
        out,
        candidate_lens,
        width,
        page_size,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        req_to_token.stride(0),
        candidate_blocks.stride(0),
        out.stride(0),
        out.stride(0),
        H=heads,
        HALF_D=INDEX_HEAD_DIM // 2,
        BLOCK_L=block_l,
        RATIO=ratio,
        USE_REQ_TO_TOKEN=False,
        USE_CANDIDATE_BLOCKS=True,
        CANDIDATE_BLOCK_SIZE=candidate_block_size,
        WRITE_CANDIDATES=False,
        WRITE_LOGITS=True,
        num_warps=4,
    )
    return out


FP8_E4M3_MAX = 448.0


__all__ = [
    "fp4_index_logits_req_to_token",
    "fp4_index_logits_candidate_blocks",
]
