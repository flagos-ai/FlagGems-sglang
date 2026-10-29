# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SGLang compressed-candidate packet addressing for prefill index scores.

Independent Q/K score math is in flag_gems.fused.dsv41.fp8_prefill_score.
This fused path retains candidate-ID gathering and the consumer's output
column layout, avoiding an intermediate gathered key tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _loop_score(
    Q,
    W,
    K,
    Lens,
    Candidates,
    Out,
    M: tl.constexpr,
    L: tl.constexpr,
    OUT_L: tl.constexpr,
    CAND_STRIDE: tl.constexpr,
    BN: tl.constexpr,
    NTILES: tl.constexpr,
    GROUP: tl.constexpr,
    USE_CANDIDATES: tl.constexpr,
    CBS: tl.constexpr,
    PIPE: tl.constexpr,
    SKIP_TILE: tl.constexpr,
    STORE_CACHE: tl.constexpr,
    GATHER_MODE: tl.constexpr,
    KEY_CACHE: tl.constexpr,
    KEY_EVICT: tl.constexpr,
):
    packets = tl.cdiv(OUT_L, BN * NTILES)
    if GROUP == 0:
        row = tl.program_id(0)
        packet = tl.program_id(1)
    else:
        pid = tl.program_id(0)
        first = pid // (GROUP * packets) * GROUP
        gm = tl.minimum(M - first, GROUP)
        row = first + pid % gm
        packet = (pid % (GROUP * packets)) // gm
    visible = tl.minimum(tl.load(Lens + row), L)
    packet_first = packet * BN * NTILES
    if USE_CANDIDATES:
        # Inspect one ID per candidate block before loading q. Multiplication
        # remains int64, matching the source's index and sentinel arithmetic.
        cb = packet_first // CBS + tl.arange(0, BN * NTILES // CBS)
        bid = tl.load(
            Candidates + row * CAND_STRIDE + cb, cb * CBS < OUT_L, other=-1
        ).to(tl.int64)
        logical_first = bid * CBS
        packet_has_values = (
            tl.sum(
                (
                    (logical_first >= 0)
                    & (logical_first < visible)
                    & (cb * CBS < OUT_L)
                ).to(tl.int32),
                0,
            )
            > 0
        )
    else:
        packet_has_values = packet_first < visible
    if packet_has_values:
        h = tl.arange(0, 32)
        d = tl.arange(0, 128)
        n = tl.arange(0, BN)
        # Q and weights are reused across independent N tiles. Each individual
        # dot and its head reduction retain the source H32/K128 contract.
        q = tl.load(Q + row * 4096 + h[:, None] * 128 + d[None, :])
        weights = tl.load(W + row * 32 + h).to(tl.float32)
        for tile in tl.range(0, NTILES, num_stages=PIPE, loop_unroll_factor=1):
            cols = packet_first + tile * BN + n
            if USE_CANDIDATES:
                if GATHER_MODE == 1:
                    tile_cb = (
                        packet_first // CBS
                        + tile * (BN // CBS)
                        + tl.arange(0, BN // CBS)
                    )
                    tile_bid = tl.load(
                        Candidates + row * CAND_STRIDE + tile_cb,
                        tile_cb * CBS < OUT_L,
                        other=-1,
                    ).to(tl.int64)
                    logical = tl.reshape(
                        tile_bid[:, None] * CBS + tl.arange(0, CBS)[None, :],
                        (BN,),
                    )
                else:
                    blocks = tl.load(
                        Candidates + row * CAND_STRIDE + cols // CBS,
                        cols < OUT_L,
                        other=-1,
                    ).to(tl.int64)
                    logical = blocks * CBS + cols % CBS
            else:
                logical = cols.to(tl.int64)
            valid = (logical >= 0) & (logical < visible) & (cols < OUT_L)
            if (not SKIP_TILE) or tl.sum(valid.to(tl.int32), 0) > 0:
                keys = tl.load(
                    K + logical[:, None] * 128 + d[None, :],
                    valid[:, None],
                    other=0.0,
                    cache_modifier=KEY_CACHE,
                    eviction_policy=KEY_EVICT,
                )
                acc = tl.dot(
                    q,
                    tl.trans(keys),
                    out_dtype=tl.float32,
                    max_num_imprecise_acc=0,
                )
                s = acc.to(tl.bfloat16).to(tl.float32)
                s = tl.maximum(s, 0.0)
                s = (s * weights[:, None]).to(tl.bfloat16).to(tl.float32)
                logits = tl.sum(s, 0).to(tl.bfloat16).to(tl.float32)
                logits = tl.where(valid, logits, float("-inf"))
            else:
                logits = tl.full((BN,), float("-inf"), tl.float32)
            tl.store(
                Out + row * OUT_L + cols,
                logits,
                cols < OUT_L,
                cache_modifier=STORE_CACHE,
            )
    else:
        cols = packet_first + tl.arange(0, BN * NTILES)
        tl.store(
            Out + row * OUT_L + cols,
            float("-inf"),
            cols < OUT_L,
            cache_modifier=STORE_CACHE,
        )


def supports_candidate_prefill(q, weights, keys, lens, candidate_blocks):
    """Whether inputs match the measured H100 packet-scoring domain."""
    tensors = (q, weights, keys, lens, candidate_blocks)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    return (
        q.ndim == 3
        and q.shape[1:] == (32, 128)
        and keys.ndim == 2
        and keys.shape[1] == 128
        and 2048 <= q.shape[0] <= 16384
        and 16384 <= keys.shape[0] <= 614400
        and weights.shape == (q.shape[0], 32)
        and lens.shape == (q.shape[0],)
        and candidate_blocks.shape == (q.shape[0], 2048)
        and q.dtype == keys.dtype == torch.float8_e4m3fn
        and weights.dtype == torch.bfloat16
        and lens.dtype in (torch.int32, torch.int64)
        and candidate_blocks.dtype == torch.int32
        and q.is_cuda
        and all(t.device == q.device and t.is_contiguous() for t in tensors)
        and triton.__version__.split("+")[0] == "3.6.0"
        and torch.cuda.get_device_capability(q.device) == (9, 0)
    )


def candidate_prefill_score(q, weights, keys, lens, candidate_blocks):
    """Score SGLang's 2,048 candidate blocks of eight logical keys each.

    Candidate -1 is an invalid block; invisible output positions are -inf.
    The caller keeps the original entry point outside this qualified domain.
    """
    if not supports_candidate_prefill(
        q, weights, keys, lens, candidate_blocks
    ):
        raise ValueError(
            "unsupported SGLang compressed-candidate score layout"
        )
    m, length, width = q.shape[0], keys.shape[0], 2048 * 8
    out = torch.empty((m, width), device=q.device, dtype=torch.float32)
    store_cache = ".cs" if length > 131072 else ""
    _loop_score[(m, triton.cdiv(width, 128 * 32))](
        q,
        weights,
        keys,
        lens,
        candidate_blocks,
        out,
        m,
        length,
        width,
        2048,
        128,
        32,
        0,
        True,
        8,
        1,
        True,
        store_cache,
        1,
        "",
        "",
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return out


__all__ = ["candidate_prefill_score"]
