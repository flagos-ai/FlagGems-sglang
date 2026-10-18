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

"""Packed serial FP4-storage/FP8-compute indexer for FlagTree 3.6 SM90.

The default launcher is serial. Warp specialization is retained as an internal
experimental configuration; it is never selected by the public dispatch.
"""

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from flag_gems.fused.dsv41.fp4_fp8_tle import (
    decode_fp4_e2m1_words_to_fp8x4 as _decode_u32,
)

from .dsv41_fp4_fp8_indexer import _fallback, _prepare_all, bf16


@triton.jit
def _load_tile(
    RT,
    TABLE,
    req,
    eligible,
    limit,
    tile,
    k0,
    k1,
    k2,
    k3,
    exponents,
    validity,
    PAGE: tl.constexpr,
    ROW: tl.constexpr,
    SR: tl.constexpr,
    RATIO: tl.constexpr,
    BL: tl.constexpr,
):
    il = tl.arange(0, BL)
    _ik = tl.arange(0, 128)
    si = tl.arange(0, 4)
    logical = tile * BL + il
    slot = (
        tl.load(
            RT + req.to(tl.int64) * SR + logical.to(tl.int64) * RATIO,
            (logical < limit) & eligible,
            other=0,
        ).to(tl.int64)
        // RATIO
    )
    row, off = (slot // PAGE) * ROW, slot % PAGE
    exps = tl.load(
        TABLE + row[:, None] + PAGE * 64 + off[:, None] * 4 + si[None, :],
        (logical[:, None] < limit) & eligible,
        other=127,
    ).to(tl.int32)
    low, high = tl.min(exps, 1), tl.max(exps, 1)
    fast = eligible & (
        tl.sum(((high - low > 4) | (high > 191) | (low < 64)).to(tl.int32), 0) == 0
    )
    word_index = tl.arange(0, 8)
    shared_rows = tl.broadcast_to(il[:, None], (BL, 8))
    shared_cols = tl.broadcast_to((word_index * 4)[None, :], (BL, 8))
    for block in tl.static_range(4):
        words = tl.load(
            (TABLE + row[:, None] + off[:, None] * 64).to(tl.pointer_type(tl.uint16))
            + block * 8
            + word_index[None, :],
            (logical[:, None] < limit) & fast,
            other=0,
        ).to(tl.uint32)
        packed = _decode_u32(words)
        if block == 0:
            pointers = tle.gpu.local_ptr(k0, (shared_rows, shared_cols))
        elif block == 1:
            pointers = tle.gpu.local_ptr(k1, (shared_rows, shared_cols))
        elif block == 2:
            pointers = tle.gpu.local_ptr(k2, (shared_rows, shared_cols))
        else:
            pointers = tle.gpu.local_ptr(k3, (shared_rows, shared_cols))
        tl.store(pointers.to(tl.pointer_type(tl.uint32, 3)), packed)
    exponents.store(exps)
    validity.store(tl.full((1,), fast, tl.int32))


@triton.jit
def _compute_tile(
    k0,
    k1,
    k2,
    k3,
    exponents,
    validity,
    qsm,
    qmeta,
    LENS,
    OUT,
    CS,
    first,
    tile,
    L: tl.constexpr,
    SO: tl.constexpr,
    SCS: tl.constexpr,
    BL: tl.constexpr,
    CB: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
):
    fast = tl.sum(validity.load(), 0) != 0
    if fast:
        exps = exponents.load()
        _mh = tl.arange(0, 64)
        si = tl.arange(0, 4)
        lp = tile * BL + tl.arange(0, BL)
        og = tl.arange(0, 2)
        for pair in range(3):
            acc = tl.zeros((BL, 64), tl.float32)
            for block in tl.static_range(4):
                if block == 0:
                    operand = k0
                elif block == 1:
                    operand = k1
                elif block == 2:
                    operand = k2
                else:
                    operand = k3
                part = tle.gpu.wgmma(
                    operand,
                    qsm.slot(block * 3 + pair),
                    trans_b=True,
                    max_num_imprecise_acc=32,
                )
                part = tle.gpu.wgmma_wait(0, part)
                encoded = tl.sum(tl.where(si[None, :] == block, exps, 0), 1)
                ks = (encoded << 23).to(tl.float32, bitcast=True)
                qs = qmeta.slot(block * 3 + pair).load()
                acc += part * (ks[:, None] * qs[None, :])
            weights = qmeta.slot(12 + pair).load()
            score = tl.maximum(acc.to(tl.bfloat16).to(tl.float32), 0.0)
            score = (score * weights[None, :]).to(tl.bfloat16).to(tl.float32)
            logit = (
                tl.sum(tl.reshape(score, (BL, 2, 32)), 2)
                .trans()
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            qb = first + pair * 2 + og
            qlens = tl.load(LENS + qb)
            logit = tl.where(
                lp[None, :] < tl.minimum(qlens[:, None], L), logit, -float("inf")
            )
            if WRITE_LOGITS:
                tl.store(OUT + qb[:, None] * SO + lp[None, :], logit, lp[None, :] < SO)
            if WRITE_CANDIDATES:
                scores = tl.max(tl.reshape(logit, (2, BL // CB, CB)), 2)
                ids = tile * (BL // CB) + tl.arange(0, BL // CB)
                scores = tl.where(
                    (qlens[:, None] > 0) & (ids[None, :] == (qlens[:, None] - 1) // CB),
                    float("inf"),
                    scores,
                )
                tl.store(
                    CS + qb[:, None] * SCS + ids[None, :],
                    scores,
                    ids[None, :] < tl.cdiv(L, CB),
                )


@triton.jit
def _producer(
    writer,
    RT,
    TABLE,
    FLAGS,
    NEED,
    req,
    eligible,
    limit,
    L: tl.constexpr,
    PAGE: tl.constexpr,
    ROW: tl.constexpr,
    SR: tl.constexpr,
    RATIO: tl.constexpr,
    BL: tl.constexpr,
):
    sequence = 0
    for tile in range(
        tl.program_id(1), tl.cdiv(tl.maximum(limit, 0), BL), tl.num_programs(1)
    ):
        slot = writer.acquire(sequence)
        _load_tile(
            RT,
            TABLE,
            req,
            eligible,
            limit,
            tile,
            slot.k0,
            slot.k1,
            slot.k2,
            slot.k3,
            slot.exponents,
            slot.validity,
            PAGE,
            ROW,
            SR,
            RATIO,
            BL,
        )
        writer.commit(sequence)
        sequence += 1  # noqa: SIM113 -- TLE pipe sequence follows the device loop.


@triton.jit
def _consumer(
    reader,
    qsm,
    qmeta,
    LENS,
    OUT,
    CS,
    FLAGS,
    NEED,
    first,
    limit,
    L: tl.constexpr,
    SO: tl.constexpr,
    SCS: tl.constexpr,
    BL: tl.constexpr,
    CB: tl.constexpr,
    WRITE_LOGITS: tl.constexpr,
    WRITE_CANDIDATES: tl.constexpr,
):
    sequence = 0
    for tile in range(
        tl.program_id(1), tl.cdiv(tl.maximum(limit, 0), BL), tl.num_programs(1)
    ):
        current = reader.wait(sequence)
        slot = current.slot
        _compute_tile(
            slot.k0,
            slot.k1,
            slot.k2,
            slot.k3,
            slot.exponents,
            slot.validity,
            qsm,
            qmeta,
            LENS,
            OUT,
            CS,
            first,
            tile,
            L,
            SO,
            SCS,
            BL,
            CB,
            WRITE_LOGITS,
            WRITE_CANDIDATES,
        )
        fast = tl.sum(slot.validity.load(), 0) != 0
        tl.store(FLAGS + tl.program_id(0) * tl.cdiv(L, BL) + tile, fast.to(tl.int32))
        if not fast:
            tl.atomic_or(NEED + tl.program_id(0), 1, sem="relaxed")
        reader.release(sequence)
        sequence += 1  # noqa: SIM113 -- TLE pipe sequence follows the device loop.


@triton.jit
def native_tle(
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
    STAGES: tl.constexpr,
    USE_WS: tl.constexpr,
    PRODUCER_WARPS: tl.constexpr,
    PRODUCER_REGS: tl.constexpr,
):
    first = tl.program_id(0) * 6
    x = tl.arange(0, 8)
    lens = tl.load(LENS + first + x, x < 6, other=0)
    limit = tl.minimum(tl.max(lens, 0), L)
    req = tl.load(REQ + first)
    reqs = tl.load(REQ + first + x, x < 6, other=req)
    heads = tl.arange(0, 256)
    qok = tl.load(QOK + first * 32 + heads, heads < 192, other=1)
    eligible = (tl.sum((reqs != req).to(tl.int32), 0) == 0) & (tl.min(qok, 0) == 1)
    qsm = tle.gpu.alloc([12, 64, 32], tl.float8e4nv)
    qmeta = tle.gpu.alloc([16, 64], tl.float32, nv_mma_shared_layout=False)
    qh, qk = tl.arange(0, 64), tl.arange(0, 32)
    for pair in tl.static_range(3):
        qmeta.slot(12 + pair).store(
            tl.load(W + (first + pair * 2) * 32 + qh).to(tl.float32)
        )
        for block in tl.static_range(4):
            qsm.slot(block * 3 + pair).store(
                tl.load(
                    QE
                    + ((first + pair * 2) * 32 + qh[:, None]) * 128
                    + block * 32
                    + qk[None, :]
                )
            )
            qmeta.slot(block * 3 + pair).store(
                tl.load(QS + ((first + pair * 2) * 32 + qh) * 4 + block)
            )
    k0 = tle.gpu.alloc([STAGES, BL, 32], tl.float8e4nv)
    k1 = tle.gpu.alloc([STAGES, BL, 32], tl.float8e4nv)
    k2 = tle.gpu.alloc([STAGES, BL, 32], tl.float8e4nv)
    k3 = tle.gpu.alloc([STAGES, BL, 32], tl.float8e4nv)
    exponents = tle.gpu.alloc([STAGES, BL, 4], tl.int32, nv_mma_shared_layout=False)
    validity = tle.gpu.alloc([STAGES, 1], tl.int32, nv_mma_shared_layout=False)
    if WRITE_LOGITS and not WRITE_CANDIDATES:
        ol, og = tl.arange(0, BL), tl.arange(0, 2)
        for tail in range(
            tl.cdiv(tl.maximum(limit, 0), BL) + tl.program_id(1),
            tl.cdiv(L, BL),
            tl.num_programs(1),
        ):
            lp = tail * BL + ol
            for pair in tl.static_range(3):
                qb = first + pair * 2 + og
                tl.store(
                    OUT + qb[:, None] * SO + lp[None, :],
                    -float("inf"),
                    lp[None, :] < SO,
                )
    if USE_WS:
        pipe = tle.pipe(
            capacity=STAGES,
            name="decoded_keys",
            k0=k0,
            k1=k1,
            k2=k2,
            k3=k3,
            exponents=exponents,
            validity=validity,
        )
        tle.gpu.warp_specialize(
            [
                (
                    _consumer,
                    (
                        pipe.reader(),
                        qsm,
                        qmeta,
                        LENS,
                        OUT,
                        CS,
                        FLAGS,
                        NEED,
                        first,
                        limit,
                        L,
                        SO,
                        SCS,
                        BL,
                        CB,
                        WRITE_LOGITS,
                        WRITE_CANDIDATES,
                    ),
                ),
                (
                    _producer,
                    (
                        pipe.writer(),
                        RT,
                        TABLE,
                        FLAGS,
                        NEED,
                        req,
                        eligible,
                        limit,
                        L,
                        PAGE,
                        ROW,
                        SR,
                        RATIO,
                        BL,
                    ),
                ),
            ],
            [PRODUCER_WARPS],
            [PRODUCER_REGS],
        )
    else:
        for tile in range(
            tl.program_id(1), tl.cdiv(tl.maximum(limit, 0), BL), tl.num_programs(1)
        ):
            _load_tile(
                RT,
                TABLE,
                req,
                eligible,
                limit,
                tile,
                k0.slot(0),
                k1.slot(0),
                k2.slot(0),
                k3.slot(0),
                exponents.slot(0),
                validity.slot(0),
                PAGE,
                ROW,
                SR,
                RATIO,
                BL,
            )
            _compute_tile(
                k0.slot(0),
                k1.slot(0),
                k2.slot(0),
                k3.slot(0),
                exponents.slot(0),
                validity.slot(0),
                qsm,
                qmeta,
                LENS,
                OUT,
                CS,
                first,
                tile,
                L,
                SO,
                SCS,
                BL,
                CB,
                WRITE_LOGITS,
                WRITE_CANDIDATES,
            )
            fast = tl.sum(validity.slot(0).load(), 0) != 0
            tl.store(
                FLAGS + tl.program_id(0) * tl.cdiv(L, BL) + tile, fast.to(tl.int32)
            )
            if not fast:
                tl.atomic_or(NEED + tl.program_id(0), 1, sem="relaxed")


class _NativeKernel:
    def __init__(self, *, ws=False, stages=1):
        self.ws = ws
        self.stages = stages
        self.capture = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            compiled = native_tle[grid](
                *args,
                **kwargs,
                STAGES=self.stages,
                USE_WS=self.ws,
                PRODUCER_WARPS=4,
                PRODUCER_REGS=112,
            )
            if self.capture is not None:
                self.capture.append((args, compiled, grid))
            return compiled

        return launch


# Frozen serial schedule. No environment switch enables the experimental WS.
_native = _native_r16 = _NativeKernel()


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
