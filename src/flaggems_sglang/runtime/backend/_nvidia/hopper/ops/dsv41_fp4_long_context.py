# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""H100 long-context packed index scoring for SGLang request layouts."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# The request table width includes SGLang's one-megatoken cache geometry.
RT_WIDTH = 1048843
DENSE_WIDTH = 524421
BATCHES = (6, 12, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 108, 120, 144, 192)
DENSE_BATCHES = (108, 120, 144, 192)
WORKERS = {
    6: 44,
    12: 44,
    18: 59,
    24: 22,
    30: 36,
    36: 30,
    42: 26,
    48: 11,
    60: 18,
    72: 15,
    84: 13,
    96: 11,
    108: 10,
    120: 9,
    144: 8,
    192: 6,
}
_candidate_keys = None
fp4_weighted_head_score = None


def _load_candidate_math():
    global _candidate_keys, fp4_weighted_head_score
    if _candidate_keys is None:
        from . import _dsv41_fp4_candidate_keys as compact

        compact._load_fp4_math()
        fp4_weighted_head_score = compact.fp4_weighted_head_score
        _candidate_keys = compact._candidate_keys


def _abi(q, weights, rt, req, table):
    """Check static tensor and hardware contracts before dispatch."""
    try:
        b = q.shape[0]
        if not (
            q.ndim == 3
            and q.shape == (b, 32, 128)
            and b in BATCHES
            and q.dtype == torch.bfloat16
            and q.is_cuda
            and weights.shape == (b, 32)
            and weights.dtype == torch.bfloat16
            and req.shape == (b,)
            and req.dtype == torch.int64
            and rt.ndim == 2
            and rt.shape[1] == RT_WIDTH
            and rt.dtype == torch.int32
            and rt.stride(1) == 1
            and table.ndim == 2
            and table.shape[1] == 4352
            and table.dtype == torch.uint8
            and table.stride(1) == 1
            and table.stride(0) % 2 == 0
            and table.data_ptr() % 2 == 0
            and all(
                t.is_cuda and t.device == q.device
                for t in (weights, rt, req, table)
            )
        ):
            return False
        props = torch.cuda.get_device_properties(q.device)
        return (props.major, props.minor, props.multi_processor_count) == (
            9,
            0,
            132,
        )
    except (AttributeError, IndexError, TypeError):
        return False


def producer_family(
    q,
    weights,
    rt,
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
    if not _abi(q, weights, rt, req, table):
        return None
    if (
        page_size != 64
        or query_group_size != 6
        or lens.shape != (q.shape[0],)
        or lens.dtype not in (torch.int32, torch.int64)
        or not lens.is_cuda
        or lens.device != q.device
    ):
        return None
    if (
        q.shape[0] in BATCHES
        and ratio == 1
        and width == RT_WIDTH
        and candidate_block_size == 8
        and not write_logits
    ):
        return "source"
    if (
        q.shape[0] in DENSE_BATCHES
        and ratio == 2
        and width == DENSE_WIDTH
        and candidate_block_size == 0
        and write_logits
    ):
        return "dense"
    return None


def dsv41_long_context_producer(
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
    """Return None before execution when the decode tensor contract is unsupported."""
    family = producer_family(
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
    if family is None:
        return None
    if family == "source":
        from . import dsv41_fp4_indexer as bf16

        try:
            from flag_gems.fused.dsv41.fp4_fp8 import FP4_FP8_SCORE_ABI
        except ImportError:
            return None
        if FP4_FP8_SCORE_ABI != 2:
            return None
        source_impl = bf16._tle_native_fp8()
        if source_impl is None:
            return None
        return source_impl(
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
    from . import dsv41_fp4_indexer as bf16

    bf16._load_fp4_math()
    b = q.shape[0]
    q = q.contiguous()
    weights = weights.contiguous()
    req = req.contiguous()
    lens = lens.to(torch.int64).contiguous()
    stride = triton.cdiv(width, 4) * 4
    storage = torch.empty((b, stride), device=q.device, dtype=torch.float32)
    out = storage[:, :width]
    bf16._fp4_persistent_initialize[(triton.cdiv(b * stride, 1024),)](
        out,
        out,
        lens,
        lens,
        b,
        stride,
        stride,
        True,
        False,
        1,
        1024,
        num_warps=4,
    )
    bf16._fp4_index_logits_grouped_persistent_kernel[(b // 6, 512)](
        q,
        weights,
        req_to_token,
        req,
        lens,
        table,
        out,
        out,
        lens,
        b,
        width,
        64,
        table.stride(0),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        req_to_token.stride(0),
        stride,
        stride,
        GROUP=6,
        PGROUP=8,
        RATIO=2,
        BL=128,
        CB=1,
        WRITE_LOGITS=True,
        WRITE_CANDIDATES=False,
        num_warps=4,
        maxnreg=160,
    )
    return out


def compact_supported(
    q,
    weights,
    rt,
    req,
    blocks,
    counts,
    table,
    page_size,
    ratio,
    candidate_block_size,
):
    return (
        _abi(q, weights, rt, req, table)
        and q.shape[0] in BATCHES
        and page_size == 64
        and ratio == 1
        and candidate_block_size == 8
        and blocks.shape == (q.shape[0], 2048)
        and blocks.dtype == torch.int32
        and counts.shape == (q.shape[0],)
        and counts.dtype == torch.int32
        and blocks.is_cuda
        and blocks.device == q.device
        and counts.is_cuda
        and counts.device == q.device
    )


def dsv41_long_context_compact(
    q,
    weights,
    req_to_token,
    req,
    candidate_blocks,
    candidate_lens,
    table,
    page_size,
    ratio,
    candidate_block_size,
):
    """Score compact decode candidates, or return None before execution."""
    if not compact_supported(
        q,
        weights,
        req_to_token,
        req,
        candidate_blocks,
        candidate_lens,
        table,
        page_size,
        ratio,
        candidate_block_size,
    ):
        return None
    q = q.contiguous()
    weights = weights.contiguous()
    req = req.contiguous()
    candidate_blocks = candidate_blocks.contiguous()
    candidate_lens = candidate_lens.contiguous()
    b = q.shape[0]
    out = torch.empty((b, 16384), device=q.device, dtype=torch.float32)
    workers = WORKERS[b]
    _load_candidate_math()
    _long_context_candidate_kernel[(b, workers)](
        q,
        weights,
        req_to_token,
        req,
        candidate_blocks,
        candidate_lens,
        table,
        out,
        N_REQUESTS=req_to_token.shape[0],
        N_SLOTS=table.shape[0] * 64,
        TABLE_STRIDE=req_to_token.stride(0),
        CACHE_STRIDE=table.stride(0),
        WORKERS=workers,
        RT_WIDTH=RT_WIDTH,
        num_warps=4,
    )
    return out


# The request-table width is explicit to support the long-context layout.
@triton.jit
def _long_context_candidate_kernel(
    Q,
    W,
    ReqTable,
    Requests,
    Blocks,
    Counts,
    Table,
    Out,
    N_REQUESTS: tl.constexpr,
    N_SLOTS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    WORKERS: tl.constexpr,
    RT_WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    req = tl.load(Requests + row).to(tl.int64)
    count = tl.minimum(tl.maximum(tl.load(Counts + row), 0), 16384)
    h = tl.arange(0, 32)
    i = tl.arange(0, 64)
    qe = tl.load(Q + row * 4096 + h[:, None] * 128 + 2 * i[None, :])
    qo = tl.load(Q + row * 4096 + h[:, None] * 128 + 2 * i[None, :] + 1)
    weights = tl.load(W + row * 32 + h).to(tl.float32)
    for tile in range(tl.program_id(1), 256, WORKERS):
        compact = tile * 64 + tl.arange(0, 64)
        if tile * 64 >= count:
            tl.store(Out + row * 16384 + compact, -float("inf"))
        else:
            valid = compact < count
            block = tl.load(
                Blocks + row * 2048 + compact // 8, valid, other=-1
            ).to(tl.int64)
            logical = block * 8 + compact % 8
            valid = (
                valid
                & (block >= 0)
                & (logical >= 0)
                & (logical < RT_WIDTH)
                & (req >= 0)
                & (req < N_REQUESTS)
            )
            slot = tl.load(
                ReqTable + req * TABLE_STRIDE + logical, valid, other=0
            ).to(tl.int64)
            valid = valid & (slot >= 0) & (slot < N_SLOTS)
            kl, kh = _candidate_keys(Table, slot, valid, 64, CACHE_STRIDE)
            score = fp4_weighted_head_score(qe, qo, kl, kh, weights)
            tl.store(
                Out + row * 16384 + compact,
                tl.where(valid, score, -float("inf")),
            )


__all__ = ["dsv41_long_context_producer", "dsv41_long_context_compact"]
