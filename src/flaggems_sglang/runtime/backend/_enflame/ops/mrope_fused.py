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

"""Triton implementation of rope/mrope_fused (GCU-adapted).

Semantics (mirrors the reference): for each pair index ``i < rotary_dim // 2``
the rotation angle row of ``cos_sin_cache`` is selected by ``mrope_section``
-- ``positions[0]`` (temporal) for ``i < section_t``, ``positions[1]``
(height) for ``section_t <= i < section_t + section_h``, else ``positions[2]``
(width); ``x1 = x[..., :half]``, ``x2 = x[..., half:rotary_dim]`` are rotated
out-of-place in float32 and the ``head_size - rotary_dim`` tail is copied
through unchanged. q and k share one kernel launch.

GCU adaptation notes (2026-11-07 20:22 toolchain, follows the
``ernie45_rope_fused`` template in knowledge/flaggems_cases.md; second
round 2026-11-12):
* The previous one-token-per-program kernel (2D ``(heads, pairs)`` masked
  tiles with same-base-pointer offset pairs) spends ~230ns/token on
  per-program dispatch/prologue tax: at num_tokens=8192 the whole op runs at
  ~18GB/s, two orders of magnitude off roofline. 2D tiles of that shape are
  also on the broken ``memref_ext.slice_pad_start`` path
  (compiler_issues/sglang_ernie45_rope_fused.md).
* Fast path: each program owns ``TPP`` consecutive tokens and all heads.
  The 1D index space ``m`` enumerates (token, pair) with ``t = m >>
  log2(half)`` and ``p = m & (half - 1)``; the rotation math needs no
  value-level broadcast. The tail block is anchored (``tok0 = min(pid *
  TPP, N - TPP)``) so every rotation access is mask-free (idempotent
  overwrite instead of predicates). Same-base offset pairs (x1/x2,
  out1/out2) are defeated by passing each tensor twice as independent
  pointer arguments; ``sin`` uses an offset view of the cache
  (``cos_sin_cache[:, half:]``) for the same reason.
* Second round (2026-11-12): the section row ``0/1/2`` that selects
  t/h/w positions is derived from the arange-constant ``p`` alone, so the
  position gather collapses from three ``[TPP*half]`` loads plus a
  two-level ``where`` chain on loaded values into a **single**
  ``positions[row * num_tokens + tok]`` gather (the row predicate folds
  at compile time, same mechanism as the interleaved_rope constant-column
  arm). Probe ladder at num_tokens=8192: index chain 82.5us -> ~32us
  (cos/sin gather only), rotation arithmetic stays fully hidden under the
  loads. A kernel-side direct int64 read measured equal to int32 within
  noise, but the committed version keeps the host-side
  ``positions.to(torch.int32)`` normalization (fused_pack_qkv /
  gather_spec_extras house style): the kernels address positions with
  4-byte elements, so correctness must not depend on the
  ``TORCH_GCU_ENABLE_INT64_AND_UINT64`` storage mode.
* ``num_tokens`` is a runtime argument and the tail block is anchored, so
  one compiled kernel serves every token count per (TPP bucket, shape
  constants); the bucket targets ~48-96 programs (L600 dual-program
  residency) and is clamped to ``[4, 64]``.
* Everything else falls back to a per-token strided kernel (1D masked
  lanes): non-float or mixed dtypes and non-power-of-two half. Rare
  non-contiguous q/k/cache/positions layouts are normalized with
  host-side ``contiguous()`` copies first -- ``empty_like()`` on a
  non-dense view allocates a contiguous output, so reusing the view's row
  stride for output addressing would write out of bounds.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mrope_qk_flat_kernel(
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
    SECTION_T: tl.constexpr,
    SECTION_H: tl.constexpr,
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
):
    pid = tl.program_id(0)
    tok0 = tl.minimum(pid * TPP, num_tokens - TPP)
    m = tl.arange(0, N_M)
    t = m >> LOG_HALF
    p = m & (HALF - 1)
    tok = tok0 + t
    row = tl.where(
        p < SECTION_T, 0, tl.where(p < SECTION_T + SECTION_H, 1, 2)
    )
    pos_sel = tl.load(positions_ptr + row * num_tokens + tok)
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
            tl.store(q_out_lo_ptr + off2, tl.load(q_lo_ptr + off2, mask=tlm), mask=tlm)
        for h in range(0, NUM_K_HEADS):
            off2 = tok2 * RS_K + h * HEAD_SIZE + ROTARY_DIM + c2
            tl.store(k_out_lo_ptr + off2, tl.load(k_lo_ptr + off2, mask=tlm), mask=tlm)


@triton.jit
def _mrope_qk_strided_kernel(
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
    SECTION_T: tl.constexpr,
    SECTION_H: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    TAIL_B: tl.constexpr,
    TAIL_LEN: tl.constexpr,
):
    token = tl.program_id(0)
    pair = tl.arange(0, HALF_B)
    lm = pair < HALF
    for tok in range(token, num_tokens, 65535):
        pos_t = tl.load(positions_ptr + tok)
        pos_h = tl.load(positions_ptr + num_tokens + tok)
        pos_w = tl.load(positions_ptr + 2 * num_tokens + tok)
        pos_sel = tl.where(
            pair < SECTION_T,
            pos_t,
            tl.where(pair < SECTION_T + SECTION_H, pos_h, pos_w),
        )
        cos = tl.load(cos_ptr + pos_sel * ROTARY_DIM + pair, mask=lm, other=0.0).to(
            tl.float32
        )
        sin = tl.load(
            sin_ptr + pos_sel * ROTARY_DIM + pair, mask=lm, other=0.0
        ).to(tl.float32)
        q_row = tok * RS_Q
        k_row = tok * RS_K
        for h in range(0, NUM_Q_HEADS):
            base = q_row + h * HEAD_SIZE
            x1 = tl.load(q_lo_ptr + base + pair, mask=lm, other=0.0).to(tl.float32)
            x2 = tl.load(q_hi_ptr + base + HALF + pair, mask=lm, other=0.0).to(
                tl.float32
            )
            tl.store(q_out_lo_ptr + base + pair, x1 * cos - x2 * sin, mask=lm)
            tl.store(
                q_out_hi_ptr + base + HALF + pair, x2 * cos + x1 * sin, mask=lm
            )
        for h in range(0, NUM_K_HEADS):
            base = k_row + h * HEAD_SIZE
            x1 = tl.load(k_lo_ptr + base + pair, mask=lm, other=0.0).to(tl.float32)
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
                    q_out_lo_ptr + off, tl.load(q_lo_ptr + off, mask=tm), mask=tm
                )
            for h in range(0, NUM_K_HEADS):
                off = k_row + h * HEAD_SIZE + ROTARY_DIM + tail
                tl.store(
                    k_out_lo_ptr + off, tl.load(k_lo_ptr + off, mask=tm), mask=tm
                )


def _pow2_floor(n):
    n = max(int(n), 1)
    return 1 << (n.bit_length() - 1)


def _flat_launch_config(num_tokens):
    """(TPP, num_warps) for the flat kernel: ~48-96 programs on the card."""
    tpp = 64
    while tpp > 4 and tpp * 48 > num_tokens:
        tpp //= 2
    tpp = min(tpp, _pow2_floor(num_tokens))
    return tpp, 2


def mrope_fused(
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
    if not positions.is_contiguous():
        positions = positions.contiguous()
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    if num_tokens == 0:
        return q_out, k_out
    # int32 normalization (fused_pack_qkv / gather_spec_extras house style):
    # the kernels address positions with 4-byte elements, so the layout must
    # not depend on the TORCH_GCU_ENABLE_INT64_AND_UINT64 storage mode.
    positions_i32 = positions.to(torch.int32).contiguous()
    half = rotary_dim // 2
    section_t = int(mrope_section[0])
    section_h = int(mrope_section[1])
    use_flat = (
        q.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and k.dtype == q.dtype
        and half >= 1
        and (half & (half - 1)) == 0
    )
    if use_flat:
        tpp, num_warps = _flat_launch_config(num_tokens)
        if (num_tokens + tpp - 1) // tpp <= 65535:
            tail_len = head_size - rotary_dim
            has_tail = tail_len > 0
            tail_b = triton.next_power_of_2(max(1, tail_len))
            sin_view = cos_sin_cache[:, half:]
            grid = ((num_tokens + tpp - 1) // tpp,)
            _mrope_qk_flat_kernel[grid](
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
                positions_i32,
                num_tokens,
                RS_Q=num_q_heads * head_size,
                RS_K=num_k_heads * head_size,
                LOG_HALF=half.bit_length() - 1,
                HALF=half,
                SECTION_T=section_t,
                SECTION_H=section_h,
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
                num_warps=num_warps,
            )
            return q_out, k_out
    half_b = max(triton.next_power_of_2(max(half, 1)), 2)
    tail_len = max(head_size - rotary_dim, 0)
    tail_b = triton.next_power_of_2(max(1, tail_len))
    sin_view = cos_sin_cache[:, half:] if half >= 1 else cos_sin_cache
    grid = (min(num_tokens, 65535),)
    _mrope_qk_strided_kernel[grid](
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
        positions_i32,
        num_tokens,
        RS_Q=num_q_heads * head_size,
        RS_K=num_k_heads * head_size,
        HALF=half,
        HALF_B=half_b,
        SECTION_T=section_t,
        SECTION_H=section_h,
        ROTARY_DIM=rotary_dim,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        HEAD_SIZE=head_size,
        TAIL_B=tail_b,
        TAIL_LEN=max(1, tail_len),
        num_warps=4,
    )
    return q_out, k_out


__all__ = ["mrope_fused"]
