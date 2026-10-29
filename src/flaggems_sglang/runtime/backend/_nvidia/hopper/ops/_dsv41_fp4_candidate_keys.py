# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Packed key decoding for H100 compact-candidate scoring."""

import triton
import triton.language as tl

_e2m1_decode = None
fp4_weighted_head_score = None


def _load_fp4_math():
    global _e2m1_decode, fp4_weighted_head_score
    if _e2m1_decode is None:
        from flag_gems.fused.dsv41 import fp4_indexer

        if fp4_indexer.FP4_INDEX_SCORE_ABI != 1:
            raise RuntimeError("incompatible FlagGems FP4 index-score ABI")
        fp4_weighted_head_score = fp4_indexer.fp4_weighted_head_score
        _e2m1_decode = fp4_indexer._e2m1_decode


@triton.jit
def _candidate_keys(
    Table,
    slots,
    valid,
    PAGE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
):
    i = tl.arange(0, 64)
    page = slots // PAGE
    off = slots % PAGE
    base = page * CACHE_STRIDE
    payload = tl.load(
        Table + base[:, None] + off[:, None] * 64 + i[None, :],
        valid[:, None],
        other=0,
    )
    low = _e2m1_decode(payload & 15)
    high = _e2m1_decode((payload >> 4) & 15)
    exps = tl.load(
        Table
        + base[:, None]
        + PAGE * 64
        + off[:, None] * 4
        + (i // 16)[None, :],
        valid[:, None],
        other=127,
    )
    scale = tl.exp2(exps.to(tl.float32) - 127.0)
    return (low * scale).to(tl.bfloat16), (high * scale).to(tl.bfloat16)
