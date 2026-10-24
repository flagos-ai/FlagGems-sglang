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

"""Hopper native FP8 index-score implementation selected by measured dispatch.

Six queries share each decoded K tile. Three sequential M64 WGMMA groups keep
accumulator live ranges short. Dense outputs preload query scales and weights
and fill every tail position inside the native launch. The previous candidate
path is retained for its measured winners. Generic inputs use the original
device BF16 fallback; its flag resets on every invocation and graph replay.
"""

import torch
import triton
import triton.language as tl
from flag_gems.fused.dsv41.fp4_fp8 import (
    FP4_FP8_SCORE_ABI,
    block_scaled_fp8_dot,
    e2m1x4_to_fp8,
    fp8_weighted_head_score,
    normal_ue8m0_scale,
    prepare_fp4_query,
)
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared

from . import dsv41_fp4_indexer as bf16

assert FP4_FP8_SCORE_ABI == 2


@triton.jit
def _prepare_all(Q, QE, QO, QS, OK, NEED, B: tl.constexpr):
    prepare_fp4_query(Q, QE, QS, OK)
    if tl.program_id(0) % 6 == 0:
        tl.store(NEED + tl.program_id(0) // 6, 0)


@g.jit
def _native_r16(
    QE,
    QS,
    QOK,
    W,
    RT,
    REQ,
    LENS,
    TABLE,
    OUT,
    CS,
    FLAGS,
    NEED,
    B: gl.constexpr,
    L: gl.constexpr,
    PAGE: gl.constexpr,
    ROW: gl.constexpr,
    SR: gl.constexpr,
    SO: gl.constexpr,
    SCS: gl.constexpr,
    RATIO: gl.constexpr,
    BL: gl.constexpr,
    CB: gl.constexpr,
    WRITE_LOGITS: gl.constexpr,
    WRITE_CANDIDATES: gl.constexpr,
):
    group = gl.program_id(0)
    first = group * 6
    meta: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    x = gl.arange(0, 8, layout=meta)
    lens = gl.load(LENS + first + x, x < 6, other=0)
    limit = gl.minimum(gl.max(lens, 0), L)
    req = gl.load(REQ + first)
    reqs = gl.load(REQ + first + x, x < 6, other=req)
    heads = gl.arange(0, 256, layout=meta)
    qok = gl.load(QOK + first * 32 + heads, heads < 192, other=1)
    eligible = (gl.sum((reqs != req).to(gl.int32), 0) == 0) & (gl.min(qok, 0) == 1)
    qlayout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    qh = gl.arange(0, 64, layout=gl.SliceLayout(1, qlayout))
    qk = gl.arange(0, 32, layout=gl.SliceLayout(0, qlayout))
    qsm = gl.allocate_shared_memory(
        gl.float8e4nv, [16, 64, 32], gl.NVMMASharedLayout(32, 8, rank=2)
    )
    if eligible:
        for block in gl.static_range(4):
            for pair in gl.static_range(3):
                qv = gl.load(
                    QE
                    + ((first + pair * 2) * 32 + qh[:, None]) * 128
                    + block * 32
                    + qk[None, :]
                )
                qsm.index(block * 4 + pair).store(qv)
    ksm = gl.allocate_shared_memory(
        gl.float8e4nv, [4, BL, 32], gl.NVMMASharedLayout(32, 8, rank=2)
    )
    keylayout: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    il = gl.arange(0, BL, layout=gl.SliceLayout(1, keylayout))
    ki = gl.arange(0, 128, layout=gl.SliceLayout(0, keylayout))
    scalelayout: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [4, 1], [0, 1])
    sl = gl.arange(0, BL, layout=gl.SliceLayout(1, scalelayout))
    si = gl.arange(0, 4, layout=gl.SliceLayout(0, scalelayout))
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 32]
    )
    mh = gl.arange(0, 64, layout=gl.SliceLayout(0, mma))
    outlayout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    og = gl.arange(0, 2, layout=gl.SliceLayout(1, outlayout))
    ol = gl.arange(0, BL, layout=gl.SliceLayout(0, outlayout))
    for tile in range(
        gl.program_id(1), gl.cdiv(gl.maximum(limit, 0), BL), gl.num_programs(1)
    ):
        fast = eligible
        if eligible:
            l = tile * BL + il
            slot = (
                gl.load(
                    RT + req.to(gl.int64) * SR + l.to(gl.int64) * RATIO,
                    l < limit,
                    other=0,
                ).to(gl.int64)
                // RATIO
            )
            row = (slot // PAGE) * ROW
            off = slot % PAGE
            sr = gl.convert_layout(row, gl.SliceLayout(1, scalelayout))
            sf = gl.convert_layout(off, gl.SliceLayout(1, scalelayout))
            exps = gl.load(
                TABLE + sr[:, None] + PAGE * 64 + sf[:, None] * 4 + si[None, :],
                (tile * BL + sl[:, None]) < limit,
                other=127,
            ).to(gl.int32)
            emin = gl.min(exps, 1)
            emax = gl.max(exps, 1)
            fast = (
                gl.sum(((emax - emin > 4) | (emax > 191) | (emin < 64)).to(gl.int32), 0)
                == 0
            )
            if fast:
                pay = gl.load(
                    (TABLE + row[:, None] + off[:, None] * 64).to(
                        gl.pointer_type(gl.uint16)
                    )
                    + (ki // 4)[None, :],
                    l[:, None] < limit,
                    other=0,
                ).to(gl.uint32)
                packed = e2m1x4_to_fp8(pay)
                kv = gl.reshape(packed, (BL, 4, 32)).permute((0, 2, 1))
                kv = gl.convert_layout(
                    kv, gl.BlockedLayout([1, 4, 4], [4, 8, 1], [4, 1, 1], [1, 0, 2])
                ).reshape((BL, 32, 2, 2))
                ke, ko = gl.split(kv)
                k0, k2 = gl.split(ke)
                k1, k3 = gl.split(ko)
                ksm.index(0).store(k0)
                ksm.index(1).store(k1)
                ksm.index(2).store(k2)
                ksm.index(3).store(k3)
                fence_async_shared()
                for pair in range(3):
                    acc = gl.full((BL, 64), 0, gl.float32, mma)
                    for block in gl.static_range(4):
                        qs = gl.load(QS + ((first + pair * 2) * 32 + mh) * 4 + block)
                        ex = gl.sum(gl.where(si[None, :] == block, exps, 0), 1)
                        ks = gl.convert_layout(
                            gl.exp2((ex - 127).to(gl.float32)), gl.SliceLayout(1, mma)
                        )
                        acc += block_scaled_fp8_dot(
                            ksm.index(block),
                            qsm.index(block * 4 + pair).permute((1, 0)),
                            ks,
                            qs,
                            (BL, 64),
                            mma,
                        )
                    w = gl.load(W + (first + pair * 2) * 32 + mh).to(gl.float32)
                    logit = fp8_weighted_head_score(acc, w, BL)
                    logit = gl.convert_layout(logit, outlayout)
                    qb = first + pair * 2 + og
                    qlens = gl.load(LENS + qb)
                    lp = tile * BL + ol
                    logit = gl.where(
                        lp[None, :] < gl.minimum(qlens[:, None], L),
                        logit,
                        -float("inf"),
                    )
                    if WRITE_LOGITS:
                        gl.store(
                            OUT + qb[:, None] * SO + lp[None, :], logit, lp[None, :] < L
                        )
                    if WRITE_CANDIDATES:
                        scores = gl.max(gl.reshape(logit, (2, BL // CB, CB)), 2)
                        ids = tile * (BL // CB) + gl.arange(
                            0, BL // CB, layout=gl.SliceLayout(0, scores.type.layout)
                        )
                        qlc = gl.convert_layout(
                            qlens, gl.SliceLayout(1, scores.type.layout)
                        )
                        qbc = gl.convert_layout(
                            qb, gl.SliceLayout(1, scores.type.layout)
                        )
                        scores = gl.where(
                            (qlc[:, None] > 0)
                            & (ids[None, :] == (qlc[:, None] - 1) // CB),
                            float("inf"),
                            scores,
                        )
                        gl.store(
                            CS + qbc[:, None] * SCS + ids[None, :],
                            scores,
                            ids[None, :] < gl.cdiv(L, CB),
                        )
        gl.store(FLAGS + group * gl.cdiv(L, BL) + tile, fast.to(gl.int32))
        if not fast:
            gl.atomic_or(NEED + group, 1, sem="relaxed")


@g.jit
def _native(
    QE,
    QS,
    QOK,
    W,
    RT,
    REQ,
    LENS,
    TABLE,
    OUT,
    CS,
    FLAGS,
    NEED,
    B: gl.constexpr,
    L: gl.constexpr,
    PAGE: gl.constexpr,
    ROW: gl.constexpr,
    SR: gl.constexpr,
    SO: gl.constexpr,
    SCS: gl.constexpr,
    RATIO: gl.constexpr,
    BL: gl.constexpr,
    CB: gl.constexpr,
    WRITE_LOGITS: gl.constexpr,
    WRITE_CANDIDATES: gl.constexpr,
):
    group = gl.program_id(0)
    first = group * 6
    meta: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    x = gl.arange(0, 8, layout=meta)
    lens = gl.load(LENS + first + x, x < 6, other=0)
    limit = gl.minimum(gl.max(lens, 0), L)
    req = gl.load(REQ + first)
    reqs = gl.load(REQ + first + x, x < 6, other=req)
    heads = gl.arange(0, 256, layout=meta)
    qok = gl.load(QOK + first * 32 + heads, heads < 192, other=1)
    eligible = (gl.sum((reqs != req).to(gl.int32), 0) == 0) & (gl.min(qok, 0) == 1)
    qlayout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    qh = gl.arange(0, 64, layout=gl.SliceLayout(1, qlayout))
    qk = gl.arange(0, 32, layout=gl.SliceLayout(0, qlayout))
    qsm = gl.allocate_shared_memory(
        gl.float8e4nv, [16, 64, 32], gl.NVMMASharedLayout(32, 8, rank=2)
    )
    qmeta = gl.allocate_shared_memory(
        gl.float32, [16, 64], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    if eligible:
        for pair in gl.static_range(3):
            wq = gl.load(W + (first + pair * 2) * 32 + qh).to(gl.float32)
            qmeta.index(12 + pair).store(wq)
        for block in gl.static_range(4):
            for pair in gl.static_range(3):
                qv = gl.load(
                    QE
                    + ((first + pair * 2) * 32 + qh[:, None]) * 128
                    + block * 32
                    + qk[None, :]
                )
                qsm.index(block * 4 + pair).store(qv)
                sq = gl.load(QS + ((first + pair * 2) * 32 + qh) * 4 + block)
                qmeta.index(block * 3 + pair).store(sq)
    ksm = gl.allocate_shared_memory(
        gl.float8e4nv, [4, BL, 32], gl.NVMMASharedLayout(32, 8, rank=2)
    )
    keylayout: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
    il = gl.arange(0, BL, layout=gl.SliceLayout(1, keylayout))
    ki = gl.arange(0, 128, layout=gl.SliceLayout(0, keylayout))
    scalelayout: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [4, 1], [0, 1])
    sl = gl.arange(0, BL, layout=gl.SliceLayout(1, scalelayout))
    si = gl.arange(0, 4, layout=gl.SliceLayout(0, scalelayout))
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 32]
    )
    _mh = gl.arange(0, 64, layout=gl.SliceLayout(0, mma))
    outlayout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    og = gl.arange(0, 2, layout=gl.SliceLayout(1, outlayout))
    ol = gl.arange(0, BL, layout=gl.SliceLayout(0, outlayout))
    if WRITE_LOGITS and not WRITE_CANDIDATES:
        for tail in range(
            gl.cdiv(gl.maximum(limit, 0), BL) + gl.program_id(1),
            gl.cdiv(L, BL),
            gl.num_programs(1),
        ):
            lp_tail = tail * BL + ol
            for pair_tail in gl.static_range(3):
                qb_tail = first + pair_tail * 2 + og
                gl.store(
                    OUT + qb_tail[:, None] * SO + lp_tail[None, :],
                    -float("inf"),
                    lp_tail[None, :] < SO,
                )
    for tile in range(
        gl.program_id(1), gl.cdiv(gl.maximum(limit, 0), BL), gl.num_programs(1)
    ):
        fast = eligible
        if eligible:
            l = tile * BL + il
            slot = (
                gl.load(
                    RT + req.to(gl.int64) * SR + l.to(gl.int64) * RATIO,
                    l < limit,
                    other=0,
                ).to(gl.int64)
                // RATIO
            )
            row = (slot // PAGE) * ROW
            off = slot % PAGE
            sr = gl.convert_layout(row, gl.SliceLayout(1, scalelayout))
            sf = gl.convert_layout(off, gl.SliceLayout(1, scalelayout))
            exps = gl.load(
                TABLE + sr[:, None] + PAGE * 64 + sf[:, None] * 4 + si[None, :],
                (tile * BL + sl[:, None]) < limit,
                other=127,
            ).to(gl.int32)
            emin = gl.min(exps, 1)
            emax = gl.max(exps, 1)
            fast = (
                gl.sum(((emax - emin > 4) | (emax > 191) | (emin < 64)).to(gl.int32), 0)
                == 0
            )
            if fast:
                pay = gl.load(
                    (TABLE + row[:, None] + off[:, None] * 64).to(
                        gl.pointer_type(gl.uint16)
                    )
                    + (ki // 4)[None, :],
                    l[:, None] < limit,
                    other=0,
                ).to(gl.uint32)
                packed = e2m1x4_to_fp8(pay)
                kv = gl.reshape(packed, (BL, 4, 32)).permute((0, 2, 1))
                kv = gl.convert_layout(
                    kv, gl.BlockedLayout([1, 4, 4], [4, 8, 1], [4, 1, 1], [1, 0, 2])
                ).reshape((BL, 32, 2, 2))
                ke, ko = gl.split(kv)
                k0, k2 = gl.split(ke)
                k1, k3 = gl.split(ko)
                ksm.index(0).store(k0)
                ksm.index(1).store(k1)
                ksm.index(2).store(k2)
                ksm.index(3).store(k3)
                fence_async_shared()
                for pair in range(3):
                    acc = gl.full((BL, 64), 0, gl.float32, mma)
                    for block in gl.static_range(4):
                        qs = qmeta.index(block * 3 + pair).load(gl.SliceLayout(0, mma))
                        ex = gl.sum(gl.where(si[None, :] == block, exps, 0), 1)
                        ks = gl.convert_layout(
                            normal_ue8m0_scale(ex), gl.SliceLayout(1, mma)
                        )
                        acc += block_scaled_fp8_dot(
                            ksm.index(block),
                            qsm.index(block * 4 + pair).permute((1, 0)),
                            ks,
                            qs,
                            (BL, 64),
                            mma,
                        )
                    w = qmeta.index(12 + pair).load(gl.SliceLayout(0, mma))
                    logit = fp8_weighted_head_score(acc, w, BL)
                    logit = gl.convert_layout(logit, outlayout)
                    qb = first + pair * 2 + og
                    qlens = gl.load(LENS + qb)
                    lp = tile * BL + ol
                    logit = gl.where(
                        lp[None, :] < gl.minimum(qlens[:, None], L),
                        logit,
                        -float("inf"),
                    )
                    if WRITE_LOGITS:
                        gl.store(
                            OUT + qb[:, None] * SO + lp[None, :],
                            logit,
                            lp[None, :] < SO,
                        )
                    if WRITE_CANDIDATES:
                        scores = gl.max(gl.reshape(logit, (2, BL // CB, CB)), 2)
                        ids = tile * (BL // CB) + gl.arange(
                            0, BL // CB, layout=gl.SliceLayout(0, scores.type.layout)
                        )
                        qlc = gl.convert_layout(
                            qlens, gl.SliceLayout(1, scores.type.layout)
                        )
                        qbc = gl.convert_layout(
                            qb, gl.SliceLayout(1, scores.type.layout)
                        )
                        scores = gl.where(
                            (qlc[:, None] > 0)
                            & (ids[None, :] == (qlc[:, None] - 1) // CB),
                            float("inf"),
                            scores,
                        )
                        gl.store(
                            CS + qbc[:, None] * SCS + ids[None, :],
                            scores,
                            ids[None, :] < gl.cdiv(L, CB),
                        )
        gl.store(FLAGS + group * gl.cdiv(L, BL) + tile, fast.to(gl.int32))
        if not fast:
            gl.atomic_or(NEED + group, 1, sem="relaxed")


@triton.jit
def _fallback(
    Q,
    W,
    RT,
    REQ,
    LENS,
    TABLE,
    OUT,
    CS,
    CL,
    FLAGS,
    NEED,
    B: tl.constexpr,
    L: tl.constexpr,
    PAGE: tl.constexpr,
    ROW: tl.constexpr,
    SR: tl.constexpr,
    SO: tl.constexpr,
    SCS: tl.constexpr,
    RATIO: tl.constexpr,
    BL: tl.constexpr,
    CB: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
):
    if tl.load(NEED + tl.program_id(0)) == 0:
        return
    g = tl.arange(0, 8)
    b = tl.program_id(0) * 6 + g
    lens = tl.load(LENS + b, g < 6, other=0)
    limit = tl.cdiv(tl.minimum(tl.maximum(tl.max(lens, 0), 0), L), BL)
    for tile in range(tl.program_id(1), limit, tl.num_programs(1)):
        flag = tl.load(FLAGS + tl.program_id(0) * tl.cdiv(L, BL) + tile)
        if flag == 0:
            bf16._fp4_index_logits_grouped_kernel_persistent_tile(
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
                ROW,
                4096,
                128,
                32,
                SR,
                SO,
                SCS,
                6,
                8,
                RATIO,
                BL,
                CB,
                WRITE_LOGITS,
                WRITE_CANDIDATES,
            )


def fp4_index_logits_req_to_token(
    q,
    weights,
    req_to_token,
    req,
    lens,
    table,
    page_size,
    ratio,
    width,
    candidate_block_size=0,
    write_logits=True,
    query_group_size=1,
):
    bf16._load_fp4_math()
    B, H, _D = q.shape
    if query_group_size != 6 or B % 6 or H != 32:
        return bf16.fp4_index_logits_req_to_token(
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
    out_storage = torch.empty(
        (B, triton.cdiv(width, 4) * 4 if write_logits else 4),
        dtype=torch.float32,
        device=q.device,
    )
    out = out_storage[:, :width]
    if candidate_block_size:
        num_blocks = triton.cdiv(width, candidate_block_size)
        storage = torch.empty(
            (B, triton.cdiv(num_blocks, 4) * 4), dtype=torch.float32, device=q.device
        )
        cs = storage[:, :num_blocks]
        cl = torch.empty(B, dtype=torch.int32, device=q.device)
    else:
        cs = out
        cl = lens
    qe = torch.empty((B * 32, 128), dtype=torch.float8_e4m3fn, device=q.device)
    qo = qe
    qs = torch.empty((B * 32, 4), dtype=torch.float32, device=q.device)
    ok = torch.empty((B * 32,), dtype=torch.int32, device=q.device)
    needed = torch.empty((B // 6,), dtype=torch.int32, device=q.device)
    _prepare_all[(B,)](q, qe, qo, qs, ok, needed, B, num_warps=4)
    count = max(
        B * out.stride(0) if write_logits else 0,
        B * cs.stride(0) if candidate_block_size else 0,
        B,
    )
    dense = write_logits and not candidate_block_size
    if not dense:
        bf16._fp4_persistent_initialize[(triton.cdiv(count, 1024),)](
            out,
            cs,
            cl,
            lens,
            B,
            out.stride(0),
            cs.stride(0),
            write_logits,
            bool(candidate_block_size),
            candidate_block_size or 1,
            1024,
            num_warps=4,
        )
    bl = 128
    # Two 128-thread CTAs fit each of the 132 SMs at 255 registers/thread.
    # Round down: a single CTA beyond 264 forces a second persistent wave.
    use_r16 = ratio == 1 and width == 655627 and not dense
    grid = (264 if B == 6 else 128) if use_r16 else max(1, 264 // (B // 6))
    kernel = _native_r16 if use_r16 else _native
    flags = torch.empty(
        (B // 6, triton.cdiv(width, bl)), device=q.device, dtype=torch.int32
    )
    if width:
        kernel[(B // 6, min(grid, triton.cdiv(width, bl)))](
            qe,
            qs,
            ok,
            weights,
            req_to_token,
            req,
            lens,
            table,
            out,
            cs,
            flags,
            needed,
            B,
            width,
            page_size,
            table.stride(0),
            req_to_token.stride(0),
            out.stride(0),
            cs.stride(0),
            ratio,
            bl,
            candidate_block_size or 1,
            write_logits,
            bool(candidate_block_size),
            num_warps=4,
        )
        _fallback[(B // 6, min(grid, triton.cdiv(width, bl)))](
            q,
            weights,
            req_to_token,
            req,
            lens,
            table,
            out,
            cs,
            cl,
            flags,
            needed,
            B,
            width,
            page_size,
            table.stride(0),
            req_to_token.stride(0),
            out.stride(0),
            cs.stride(0),
            ratio,
            bl,
            candidate_block_size or 1,
            write_logits,
            bool(candidate_block_size),
            num_warps=4,
        )
    if not write_logits:
        return cs, cl
    if candidate_block_size:
        return out, cs, cl
    return out
