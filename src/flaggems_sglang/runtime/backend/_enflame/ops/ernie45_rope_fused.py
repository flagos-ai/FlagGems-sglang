# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

"""Triton implementation of rope/ernie45_rope_fused (GCU-adapted).

Semantics (mirrors the reference): for each pair index ``i < rotary_dim // 2``
the rotation angles come from one of three position rows of
``cos_sin_cache`` -- ``positions[0]`` (temporal) for ``i >= section_h +
section_w``, otherwise ``positions[1]`` (height) on even ``i`` and
``positions[2]`` (width) on odd ``i``; ``x1 = x[..., :half]``,
``x2 = x[..., half:rotary_dim]`` are rotated out-of-place in float32 and the
``head_size - rotary_dim`` tail is copied through unchanged.

GCU adaptation notes (2026-11-07 20:22 toolchain):
* The previous 2D-tile kernel (one program per token, ``(heads, pairs)``
  masked tiles) no longer fresh-compiles: same-base-pointer offset-pair 2D
  masked tiles lower through the broken ``memref_ext.slice_pad_start``
  path (see compiler_issues/sglang_ernie45_rope_fused.md). Only 1D accesses
  survive -- affine or computed-index (gather) with shift/AND index
  arithmetic (div/mod gather indices silently miscompile per
  compiler_issues/sglang_rotary_embedding.md).
* Fast path: each program owns ``TPP`` consecutive tokens and all heads.
  The 1D index space ``m`` enumerates (token, pair) with ``t = m >>
  log2(half)`` and ``p = m & (half - 1)``; positions/cos/sin/x are 1D
  gathers of ``TPP * half`` lanes, so the rotation math needs no value-level
  broadcast. The tail block is anchored (``tok0 = min(pid * TPP, N - TPP)``)
  so every access is mask-free (idempotent overwrite instead of predicates).
  Same-base offset pairs (x1/x2, out1/out2) are defeated by passing each
  tensor twice as independent pointer arguments; ``sin`` uses an offset
  view of the cache (``cos_sin_cache[:, half:]``) for the same reason.
* ``num_tokens`` is a runtime argument and the tail block is anchored, so
  one compiled kernel serves every token count per (TPP bucket, shape
  constants); the bucket targets ~48-96 programs (L600 dual-program
  residency) and is clamped to ``[4, 64]``.
* Second-round tuning (2026-11-13):
  - Positions are read through a zero-copy int32 word view of the int64
    tensor (``positions.view(torch.int32)``, low word = value for ids
    < 2**31) with a ``POS_WS`` word stride (2 under
    ``TORCH_GCU_ENABLE_INT64_AND_UINT64`` where int64 slots are 8B,
    1 with the compact int32 storage the env gate keeps when off). This
    drops the per-call ATen int64->int32 cast kernel, which costs a
    fixed +2.5~4.8us per bench call (19-32% of small-token latencies)
    -- the same scheme as set_mla_kv_buffer / concat_and_cache_mla. The
    view is only taken for ``num_tokens <= 2048``: at large token
    counts the stride-2 gather of 3*N 4B words is slower than the cast
    (measured +12.6us kernel time at t=8192 vs a ~0.8us cast), so the
    flat path keeps the host cast above the threshold.
  - ``num_tokens <= 4`` dispatches to a head-split kernel (grid
    ``(ceil(N/TPP), nq+nk)``, one head per program): a single tiny
    token block leaves the whole rotation serial in one program (~40
    dependent 128B segments); splitting by head cuts kernel time
    7.98 -> 4.49us at t=1 (measured, nw=2 sweet spot). Past ~32 tokens
    the duplicated cos/sin gathers outweigh the parallelism, so the
    flat kernel stays.
  Both kernels are bit-exact against the flat kernel on every bench
  shape (7 shapes x {flat, head-split} A/B).
* Cold-compile retry (2026-11-13, gate round 1): the
  ``gcu-compiler-compile`` launcher resolves its own directory with
  ``cd "$(dirname $0)"`` (failure swallowed) and under transient
  user-mode storms the path mis-resolves, failing fresh JIT compiles
  with ``gcu-compiler-compile-args-parsing.sh: No such file or
  directory`` (P02 toolchain flake family; same signature as the
  2026-11-10 sinh gate failure). Every launch goes through
  ``_launch_with_retry`` (3 attempts) -- the flake passes on retry,
  deterministic failures still surface on the final attempt. Retrying
  is safe here: all three kernels are out-of-place (write-only
  outputs), so a re-launch after a failed attempt is idempotent.
* Everything else falls back to a per-token strided kernel (1D masked
  lanes): non-float or mixed dtypes and non-power-of-two half. Rare
  non-contiguous q/k/cache layouts are normalized with host-side
  ``contiguous()`` copies first -- ``empty_like()`` on a non-dense view
  allocates a contiguous output, so reusing the view's row stride for
  output addressing would write out of bounds (and crash the device with
  a Sip/RAS error); after normalization both kernels see contiguous
  row-major tensors and outputs.
"""

import os

import torch
import triton
import triton.language as tl

# Word-view threshold: above this the stride-2 gather of the int64 word
# view costs more kernel time than the host cast it replaces.
_WORD_VIEW_MAX_TOKENS = 2048
# Head-split arm: one tiny token block leaves the rotation serial in a
# single program; splitting by head only pays off for very few tokens.
_HEAD_SPLIT_MAX_TOKENS = 4
_HEAD_SPLIT_MAX_HEADS = 255
# Cold-compile flake retry count (P02 toolchain family; see module doc).
_COMPILE_RETRY_ATTEMPTS = 3


def _launch_with_retry(kernel, grid, args, kwargs):
    """Launch a kernel, retrying transient cold-compile flakes.

    ``gcu-compiler-compile`` resolves its own directory via
    ``cd "$(dirname $0)"`` (failure swallowed) and can mis-resolve under
    transient user-mode storms, failing a fresh JIT compile with
    ``gcu-compiler-compile-args-parsing.sh: No such file or directory``
    (P02 flake family; retry passes, deterministic failures re-raise on
    the final attempt). Idempotent here: all kernels write only to
    freshly allocated outputs.
    """
    for attempt in range(_COMPILE_RETRY_ATTEMPTS):
        try:
            kernel[grid](*args, **kwargs)
            return
        except Exception:
            if attempt == _COMPILE_RETRY_ATTEMPTS - 1:
                raise


@triton.jit
def _ernie45_rope_qk_flat_kernel(
    q_lo_ptr,
    q_hi_ptr,
    k_lo_ptr,
    k_hi_ptr,
    q_out_lo_ptr,
    q_out_hi_ptr,
    k_out_lo_ptr,
    k_out_hi_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    num_tokens,
    RS_Q: tl.constexpr,
    RS_K: tl.constexpr,
    LOG_HALF: tl.constexpr,
    HALF: tl.constexpr,
    SECTION_HW: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    TPP: tl.constexpr,
    N_M: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    LOG_TAIL: tl.constexpr,
    TAIL_B: tl.constexpr,
    TAIL_LEN: tl.constexpr,
    N_T: tl.constexpr,
    POS_WS: tl.constexpr,
):
    pid = tl.program_id(0)
    tok0 = tl.minimum(pid * TPP, num_tokens - TPP)
    m = tl.arange(0, N_M)
    t = m >> LOG_HALF
    p = m & (HALF - 1)
    tok = tok0 + t
    pr = tok * POS_WS
    pos_row = num_tokens * POS_WS
    pt = tl.load(positions_ptr + pr)
    ph = tl.load(positions_ptr + pos_row + pr)
    pw = tl.load(positions_ptr + 2 * pos_row + pr)
    pos_sel = tl.where(p < SECTION_HW, tl.where((p & 1) == 0, ph, pw), pt)
    cos = tl.load(cos_ptr + pos_sel * ROTARY_DIM + p).to(tl.float32)
    sin = tl.load(sin_ptr + pos_sel * ROTARY_DIM + p).to(tl.float32)
    qbase = tok * RS_Q + p
    kbase = tok * RS_K + p
    for h in range(0, NUM_Q_HEADS):
        off = qbase + h * HEAD_SIZE
        x1 = tl.load(q_lo_ptr + off).to(tl.float32)
        x2 = tl.load(q_hi_ptr + off + HALF).to(tl.float32)
        tl.store(q_out_lo_ptr + off, x1 * cos - x2 * sin)
        tl.store(q_out_hi_ptr + off + HALF, x2 * cos + x1 * sin)
    for h in range(0, NUM_K_HEADS):
        off = kbase + h * HEAD_SIZE
        x1 = tl.load(k_lo_ptr + off).to(tl.float32)
        x2 = tl.load(k_hi_ptr + off + HALF).to(tl.float32)
        tl.store(k_out_lo_ptr + off, x1 * cos - x2 * sin)
        tl.store(k_out_hi_ptr + off + HALF, x2 * cos + x1 * sin)
    if HAS_TAIL:
        m2 = tl.arange(0, N_T)
        t2 = m2 >> LOG_TAIL
        c2 = m2 & (TAIL_B - 1)
        tlm = c2 < TAIL_LEN
        tok2 = tok0 + t2
        for h in range(0, NUM_Q_HEADS):
            off2 = tok2 * RS_Q + h * HEAD_SIZE + ROTARY_DIM + c2
            tl.store(
                q_out_lo_ptr + off2,
                tl.load(q_lo_ptr + off2, mask=tlm),
                mask=tlm,
            )
        for h in range(0, NUM_K_HEADS):
            off2 = tok2 * RS_K + h * HEAD_SIZE + ROTARY_DIM + c2
            tl.store(
                k_out_lo_ptr + off2,
                tl.load(k_lo_ptr + off2, mask=tlm),
                mask=tlm,
            )


@triton.jit
def _ernie45_rope_qk_headsplit_kernel(
    q_lo_ptr,
    q_hi_ptr,
    k_lo_ptr,
    k_hi_ptr,
    q_out_lo_ptr,
    q_out_hi_ptr,
    k_out_lo_ptr,
    k_out_hi_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    num_tokens,
    RS_Q: tl.constexpr,
    RS_K: tl.constexpr,
    LOG_HALF: tl.constexpr,
    HALF: tl.constexpr,
    SECTION_HW: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    TPP: tl.constexpr,
    N_M: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    LOG_TAIL: tl.constexpr,
    TAIL_B: tl.constexpr,
    TAIL_LEN: tl.constexpr,
    N_T: tl.constexpr,
    POS_WS: tl.constexpr,
):
    pid = tl.program_id(0)
    hid = tl.program_id(1)
    tok0 = tl.minimum(pid * TPP, num_tokens - TPP)
    m = tl.arange(0, N_M)
    t = m >> LOG_HALF
    p = m & (HALF - 1)
    tok = tok0 + t
    pr = tok * POS_WS
    pos_row = num_tokens * POS_WS
    pt = tl.load(positions_ptr + pr)
    ph = tl.load(positions_ptr + pos_row + pr)
    pw = tl.load(positions_ptr + 2 * pos_row + pr)
    pos_sel = tl.where(p < SECTION_HW, tl.where((p & 1) == 0, ph, pw), pt)
    cos = tl.load(cos_ptr + pos_sel * ROTARY_DIM + p).to(tl.float32)
    sin = tl.load(sin_ptr + pos_sel * ROTARY_DIM + p).to(tl.float32)
    if hid < NUM_Q_HEADS:
        off = tok * RS_Q + hid * HEAD_SIZE + p
        x1 = tl.load(q_lo_ptr + off).to(tl.float32)
        x2 = tl.load(q_hi_ptr + off + HALF).to(tl.float32)
        tl.store(q_out_lo_ptr + off, x1 * cos - x2 * sin)
        tl.store(q_out_hi_ptr + off + HALF, x2 * cos + x1 * sin)
        if HAS_TAIL:
            m2 = tl.arange(0, N_T)
            t2 = m2 >> LOG_TAIL
            c2 = m2 & (TAIL_B - 1)
            tlm = c2 < TAIL_LEN
            tok2 = tok0 + t2
            off2 = tok2 * RS_Q + hid * HEAD_SIZE + ROTARY_DIM + c2
            tl.store(
                q_out_lo_ptr + off2,
                tl.load(q_lo_ptr + off2, mask=tlm),
                mask=tlm,
            )
    else:
        khead = hid - NUM_Q_HEADS
        off = tok * RS_K + khead * HEAD_SIZE + p
        x1 = tl.load(k_lo_ptr + off).to(tl.float32)
        x2 = tl.load(k_hi_ptr + off + HALF).to(tl.float32)
        tl.store(k_out_lo_ptr + off, x1 * cos - x2 * sin)
        tl.store(k_out_hi_ptr + off + HALF, x2 * cos + x1 * sin)
        if HAS_TAIL:
            m2 = tl.arange(0, N_T)
            t2 = m2 >> LOG_TAIL
            c2 = m2 & (TAIL_B - 1)
            tlm = c2 < TAIL_LEN
            tok2 = tok0 + t2
            off2 = tok2 * RS_K + khead * HEAD_SIZE + ROTARY_DIM + c2
            tl.store(
                k_out_lo_ptr + off2,
                tl.load(k_lo_ptr + off2, mask=tlm),
                mask=tlm,
            )


@triton.jit
def _ernie45_rope_qk_strided_kernel(
    q_lo_ptr,
    q_hi_ptr,
    k_lo_ptr,
    k_hi_ptr,
    q_out_lo_ptr,
    q_out_hi_ptr,
    k_out_lo_ptr,
    k_out_hi_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    num_tokens,
    RS_Q: tl.constexpr,
    RS_K: tl.constexpr,
    HALF: tl.constexpr,
    HALF_B: tl.constexpr,
    SECTION_HW: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    TAIL_B: tl.constexpr,
    TAIL_LEN: tl.constexpr,
    POS_WS: tl.constexpr,
):
    token = tl.program_id(0)
    pair = tl.arange(0, HALF_B)
    lm = pair < HALF
    for tok in range(token, num_tokens, 65535):
        pr = tok * POS_WS
        pos_row = num_tokens * POS_WS
        pos_t = tl.load(positions_ptr + pr)
        pos_h = tl.load(positions_ptr + pos_row + pr)
        pos_w = tl.load(positions_ptr + 2 * pos_row + pr)
        pos_sel = tl.where(
            pair < SECTION_HW, tl.where((pair & 1) == 0, pos_h, pos_w), pos_t
        )
        cos = tl.load(
            cos_ptr + pos_sel * ROTARY_DIM + pair, mask=lm, other=0.0
        ).to(tl.float32)
        sin = tl.load(
            sin_ptr + pos_sel * ROTARY_DIM + pair, mask=lm, other=0.0
        ).to(tl.float32)
        q_row = tok * RS_Q
        k_row = tok * RS_K
        for h in range(0, NUM_Q_HEADS):
            base = q_row + h * HEAD_SIZE
            x1 = tl.load(q_lo_ptr + base + pair, mask=lm, other=0.0).to(
                tl.float32
            )
            x2 = tl.load(q_hi_ptr + base + HALF + pair, mask=lm, other=0.0).to(
                tl.float32
            )
            tl.store(q_out_lo_ptr + base + pair, x1 * cos - x2 * sin, mask=lm)
            tl.store(
                q_out_hi_ptr + base + HALF + pair, x2 * cos + x1 * sin, mask=lm
            )
        for h in range(0, NUM_K_HEADS):
            base = k_row + h * HEAD_SIZE
            x1 = tl.load(k_lo_ptr + base + pair, mask=lm, other=0.0).to(
                tl.float32
            )
            x2 = tl.load(k_hi_ptr + base + HALF + pair, mask=lm, other=0.0).to(
                tl.float32
            )
            tl.store(k_out_lo_ptr + base + pair, x1 * cos - x2 * sin, mask=lm)
            tl.store(
                k_out_hi_ptr + base + HALF + pair, x2 * cos + x1 * sin, mask=lm
            )
        if HEAD_SIZE > ROTARY_DIM:
            tail = tl.arange(0, TAIL_B)
            tm = tail < TAIL_LEN
            for h in range(0, NUM_Q_HEADS):
                off = q_row + h * HEAD_SIZE + ROTARY_DIM + tail
                tl.store(
                    q_out_lo_ptr + off,
                    tl.load(q_lo_ptr + off, mask=tm),
                    mask=tm,
                )
            for h in range(0, NUM_K_HEADS):
                off = k_row + h * HEAD_SIZE + ROTARY_DIM + tail
                tl.store(
                    k_out_lo_ptr + off,
                    tl.load(k_lo_ptr + off, mask=tm),
                    mask=tm,
                )


def _pow2_floor(n):
    n = max(int(n), 1)
    return 1 << (n.bit_length() - 1)


def _positions_words(positions):
    """Zero-copy (int32 tensor, word stride) view of the (3, N) positions.

    torch_gcu without TORCH_GCU_ENABLE_INT64_AND_UINT64 keeps device
    int64 values as compact int32 words (element j at word j), while with
    the gate on they sit in the low word of each 8-byte slot (word 2*j).
    Position ids stay far below 2**31, so the low word is value-exact.
    This avoids the per-call ATen int64->int32 cast kernel (measured
    +2.5~4.8us per bench call at small token counts).
    """
    if positions.dtype == torch.int64:
        words = positions.contiguous().view(torch.int32)
        word_stride = (
            2
            if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") == "1"
            else 1
        )
        return words, word_stride
    if positions.dtype == torch.int32:
        return positions.contiguous(), 1
    return positions.to(torch.int32).contiguous(), 1


def _flat_launch_config(num_tokens):
    """(TPP, num_warps) for the flat kernel: ~48-96 programs on the card."""
    tpp = 64
    while tpp > 4 and tpp * 48 > num_tokens:
        tpp //= 2
    tpp = min(tpp, _pow2_floor(num_tokens))
    return tpp, 2


def ernie45_rope_fused(
    q,
    k,
    cos_sin_cache,
    positions,
    mrope_section,
    head_size,
    rotary_dim,
):
    num_tokens = q.shape[0]
    num_q_heads = q.shape[1] // head_size
    num_k_heads = k.shape[1] // head_size
    # Non-contiguous inputs are normalized up front: empty_like() on a
    # non-dense view allocates a *contiguous* output, so mixing the view's
    # row stride into output addressing would write out of bounds. Copies
    # only happen for out-of-contract layouts.
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not cos_sin_cache.is_contiguous():
        cos_sin_cache = cos_sin_cache.contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    if num_tokens == 0:
        return q_out, k_out
    half = rotary_dim // 2
    section_hw = int(mrope_section[0]) + int(mrope_section[1])
    use_flat = (
        q.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and k.dtype == q.dtype
        and half >= 1
        and (half & (half - 1)) == 0
    )
    if use_flat:
        if (
            num_tokens <= _HEAD_SPLIT_MAX_TOKENS
            and num_q_heads + num_k_heads <= _HEAD_SPLIT_MAX_HEADS
        ):
            # Tiny token counts: one head per program beats the flat
            # kernel's serial per-head loop (7.98 -> 4.49us at t=1).
            positions_arg, pos_ws = _positions_words(positions)
            tpp = min(_HEAD_SPLIT_MAX_TOKENS, _pow2_floor(num_tokens))
            tail_len = head_size - rotary_dim
            tail_b = triton.next_power_of_2(max(1, tail_len))
            sin_view = cos_sin_cache[:, half:]
            grid = (
                (num_tokens + tpp - 1) // tpp,
                num_q_heads + num_k_heads,
            )
            _launch_with_retry(
                _ernie45_rope_qk_headsplit_kernel,
                grid,
                (
                    q,
                    q,
                    k,
                    k,
                    q_out,
                    q_out,
                    k_out,
                    k_out,
                    cos_sin_cache,
                    sin_view,
                    positions_arg,
                    num_tokens,
                ),
                dict(
                    RS_Q=num_q_heads * head_size,
                    RS_K=num_k_heads * head_size,
                    LOG_HALF=half.bit_length() - 1,
                    HALF=half,
                    SECTION_HW=section_hw,
                    ROTARY_DIM=rotary_dim,
                    NUM_Q_HEADS=num_q_heads,
                    NUM_K_HEADS=num_k_heads,
                    HEAD_SIZE=head_size,
                    TPP=tpp,
                    N_M=tpp * half,
                    HAS_TAIL=tail_len > 0,
                    LOG_TAIL=tail_b.bit_length() - 1,
                    TAIL_B=tail_b,
                    TAIL_LEN=max(1, tail_len),
                    N_T=tpp * tail_b,
                    POS_WS=pos_ws,
                    num_warps=2,
                ),
            )
            return q_out, k_out
        tpp, num_warps = _flat_launch_config(num_tokens)
        if (num_tokens + tpp - 1) // tpp <= 65535:
            if num_tokens <= _WORD_VIEW_MAX_TOKENS:
                positions_arg, pos_ws = _positions_words(positions)
            else:
                # The stride-2 word-view gather costs more kernel time
                # than the cast at large token counts (+12.6us at t=8192
                # vs ~0.8us for the cast), so keep the host cast here.
                positions_arg = positions.to(torch.int32).contiguous()
                pos_ws = 1
            tail_len = head_size - rotary_dim
            has_tail = tail_len > 0
            tail_b = triton.next_power_of_2(max(1, tail_len))
            sin_view = cos_sin_cache[:, half:]
            grid = ((num_tokens + tpp - 1) // tpp,)
            _launch_with_retry(
                _ernie45_rope_qk_flat_kernel,
                grid,
                (
                    q,
                    q,
                    k,
                    k,
                    q_out,
                    q_out,
                    k_out,
                    k_out,
                    cos_sin_cache,
                    sin_view,
                    positions_arg,
                    num_tokens,
                ),
                dict(
                    RS_Q=num_q_heads * head_size,
                    RS_K=num_k_heads * head_size,
                    LOG_HALF=half.bit_length() - 1,
                    HALF=half,
                    SECTION_HW=section_hw,
                    ROTARY_DIM=rotary_dim,
                    NUM_Q_HEADS=num_q_heads,
                    NUM_K_HEADS=num_k_heads,
                    HEAD_SIZE=head_size,
                    TPP=tpp,
                    N_M=tpp * half,
                    HAS_TAIL=has_tail,
                    LOG_TAIL=tail_b.bit_length() - 1,
                    TAIL_B=tail_b,
                    TAIL_LEN=max(1, tail_len),
                    N_T=tpp * tail_b,
                    POS_WS=pos_ws,
                    num_warps=num_warps,
                ),
            )
            return q_out, k_out
    positions_arg, pos_ws = _positions_words(positions)
    half_b = max(triton.next_power_of_2(max(half, 1)), 2)
    tail_len = max(head_size - rotary_dim, 0)
    tail_b = triton.next_power_of_2(max(1, tail_len))
    sin_view = cos_sin_cache[:, half:] if half >= 1 else cos_sin_cache
    grid = (min(num_tokens, 65535),)
    _launch_with_retry(
        _ernie45_rope_qk_strided_kernel,
        grid,
        (
            q,
            q,
            k,
            k,
            q_out,
            q_out,
            k_out,
            k_out,
            cos_sin_cache,
            sin_view,
            positions_arg,
            num_tokens,
        ),
        dict(
            RS_Q=num_q_heads * head_size,
            RS_K=num_k_heads * head_size,
            HALF=half,
            HALF_B=half_b,
            SECTION_HW=section_hw,
            ROTARY_DIM=rotary_dim,
            NUM_Q_HEADS=num_q_heads,
            NUM_K_HEADS=num_k_heads,
            HEAD_SIZE=head_size,
            TAIL_B=tail_b,
            TAIL_LEN=max(1, tail_len),
            POS_WS=pos_ws,
            num_warps=4,
        ),
    )
    return q_out, k_out


__all__ = ["ernie45_rope_fused"]
