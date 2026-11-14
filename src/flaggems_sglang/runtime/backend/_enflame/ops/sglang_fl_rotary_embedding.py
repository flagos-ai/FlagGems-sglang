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

"""Triton implementation of SGLang's fused in-place RoPE (GCU-adapted).

run(positions, query, key, head_size, cos_sin_cache, is_neox=True) -> None

Semantics (mirrors sgl_kernel.apply_rope_with_cos_sin_cache_inplace):
  positions      [nnz] int64, indices into cos_sin_cache
  query          [nnz, num_q_heads * head_size] fp16/bf16
  key            [nnz, num_kv_heads * head_size] fp16/bf16
  cos_sin_cache  [max_position, rotary_dim] float32, cos in [:, :rd/2], sin in [:, rd/2:]
  head_size      int; rotary_dim = cos_sin_cache.shape[-1]
  is_neox        True  -> rotate lanes [0, rd/2) against [rd/2, rd)
                 False -> rotate even/odd lane pairs
  lanes [rd, head_size) of every head are passed through untouched.
Arithmetic is float32; results are rounded back into the original dtype and
written in place.  One launch per call.

GCU adaptation (2026-11-08 toolchain): the generic 2D-token x head-tile form
with pointer tiles ``cos_ptr + p[:, None] * stride + hoff[None, :]`` and 3D
``[TOK, HEADS, LANE]`` addressing fails to compile (make_gcuir dies with
``Pipeline run failed``; stripped probes show the ``memref_ext.slice_*``
attribute-conversion failures for multi-dimensional tiles with runtime or
data-dependent offsets -- see compiler_issues/sglang_fl_rotary_embedding.md).
Two GCU-proven structures ship here:

* Fast path (neox, pow2 half, contiguous, fp16/bf16/fp32): the
  ernie45_rope_fused / rotary_embedding flat form.  Each program owns TPP
  consecutive tokens; the 1D index space ``m`` enumerates (token, head-batch,
  lane) with shift/AND math only (div/mod index forms silently miscompile),
  positions are gathered as a 1D data-dependent vector and cos/sin rows as 1D
  gathers -- no 2D/3D pointer tiles and no value-level broadcast anywhere.
  Same-base offset pairs (x1/x2 loads, o1/o2 stores) are kept on independent
  pointer arguments and ``sin`` is an offset view of the cache
  (``cos_sin_cache[:, half:]``), mirroring the ernie45_rope_fused workaround.
  Because this op rotates q/k *in place*, the ernie45 tail-anchor trick
  (idempotent overwrite) is NOT sound here: the tail block would re-rotate
  already-rotated tokens.  A token mask (``tok < num_tokens``) is used
  instead -- measured cost-free at streaming widths on this family
  (rotary_embedding ships the same mask at 94% of copy floor).
  ``num_tokens`` is a runtime argument, so one compiled variant serves every
  token count per (TPP bucket, shape constants); the (TPP, HB_Q, HB_K,
  num_warps) dispatch buckets were scanned on L600 (do_bench median,
  2026-11-08): nnz 1/8/64/512 -> 6.4/6.4/10.7/32.9 us vs 20.1/20.5/39.5/229.3
  us for the per-token fallback (3.1x/3.2x/3.7x/7.0x); nnz=4096 -> 161.8 us.
  The half-split lane pairing keeps x1/x2 accesses in 128B segments
  (structural floor of the family, see knowledge/flaggems_cases.md
  "half-split RoPE/mrope" section), so the large-shape ceiling is
  segment-issue-rate bound, not copy-bandwidth bound.
* Fallback (everything else): one program per token (grid = (nnz,); nnz must
  stay below the 65535 grid.x limit -- same envelope as the vllm twin op),
  scalar ``pos`` load and scalar row advance, 1D lane tiles only, and a
  compile-time-unrolled head loop.

Both paths keep address math in the int32 safe domain (nnz * row_stride and
max_position * cache_stride < 2**31) and re-expose int64 positions as their
int32 payload view on the host (torch_gcu "Long use Int replace"); with
TORCH_GCU_ENABLE_INT64_AND_UINT64=1 the true int64 tensor is passed to the
flat kernel as well (1D i64 gather validated on GCU400/L600: bit-exact vs
the int32 payload variant, A/B 32.2 vs 33.8 us @nnz=512; offsets stay < 2**31
so the i64 arithmetic is only a wider load -- see
compiler_issues/int64_scalar_load_addressing.md for the env-off storage trap).
"""

import os

import torch
import triton
import triton.language as tl

_NUM_WARPS = 1


def _positions_as_int32_payload(positions):
    # torch_gcu (without TORCH_GCU_ENABLE_INT64_AND_UINT64=1) stores int64
    # tensors as an int32 payload ("GCU not support Long use Int replace"):
    # the value of element i lives in int32 word i.  A scalar i64 tl.load is
    # lowered to an 8-byte access at byte offset 8*i, which reads
    # value[2i] | (value[2i+1] << 32) and drives the cos/sin row far out of
    # bounds (compiler_issues/int64_scalar_load_addressing.md).  Re-expose
    # the payload as int32 words so the kernel loads the real values.
    # With the env enabled, int64 storage is true 8-byte: the tensor is
    # passed through as-is (only normalized to unit stride so both dispatch
    # paths can index it with ``pos_ptr + tok``).
    if positions.dtype != torch.int64:
        return positions
    if not positions.is_contiguous():
        positions = positions.contiguous()
    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64", "") in ("1", "true"):
        return positions
    return positions.view(torch.int32)


@triton.jit
def _rope_flat_kernel(
    q_ld1_ptr,
    q_ld2_ptr,
    q_st1_ptr,
    q_st2_ptr,
    k_ld1_ptr,
    k_ld2_ptr,
    k_st1_ptr,
    k_st2_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    num_tokens,
    RS_Q: tl.constexpr,
    RS_K: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    HALF: tl.constexpr,
    LOG_HALF: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    H_Q: tl.constexpr,
    H_K: tl.constexpr,
    HBQ: tl.constexpr,
    HBK: tl.constexpr,
    LOG_TH_Q: tl.constexpr,
    LOG_TH_K: tl.constexpr,
    HBH_Q: tl.constexpr,
    HBH_K: tl.constexpr,
    TPP: tl.constexpr,
):
    pid = tl.program_id(0)
    tok0 = pid * TPP
    # ---- query pass: m enumerates (token, head-batch, lane) ----
    mq = tl.arange(0, TPP * HBQ * HALF)
    tq = mq >> LOG_TH_Q
    hq = (mq & (HBH_Q - 1)) >> LOG_HALF
    jq = mq & (HALF - 1)
    tokq = tok0 + tq
    tqm = tokq < num_tokens
    posq = tl.load(pos_ptr + tokq, mask=tqm, other=0)
    cosq = tl.load(cos_ptr + posq * ROTARY_DIM + jq, mask=tqm, other=0.0).to(
        tl.float32
    )
    sinq = tl.load(sin_ptr + posq * ROTARY_DIM + jq, mask=tqm, other=0.0).to(
        tl.float32
    )
    qoff = tokq * RS_Q + hq * HEAD_SIZE + jq
    for hb in range(0, H_Q, HBQ):
        off = qoff + hb * HEAD_SIZE
        x1 = tl.load(q_ld1_ptr + off, mask=tqm, other=0.0).to(tl.float32)
        x2 = tl.load(q_ld2_ptr + off + HALF, mask=tqm, other=0.0).to(tl.float32)
        tl.store(
            q_st1_ptr + off,
            (x1 * cosq - x2 * sinq).to(q_st1_ptr.dtype.element_ty),
            mask=tqm,
        )
        tl.store(
            q_st2_ptr + off + HALF,
            (x2 * cosq + x1 * sinq).to(q_st2_ptr.dtype.element_ty),
            mask=tqm,
        )
    # ---- key pass ----
    mk = tl.arange(0, TPP * HBK * HALF)
    tk = mk >> LOG_TH_K
    hkk = (mk & (HBH_K - 1)) >> LOG_HALF
    jk = mk & (HALF - 1)
    tokk = tok0 + tk
    tkm = tokk < num_tokens
    posk = tl.load(pos_ptr + tokk, mask=tkm, other=0)
    cosk = tl.load(cos_ptr + posk * ROTARY_DIM + jk, mask=tkm, other=0.0).to(
        tl.float32
    )
    sink = tl.load(sin_ptr + posk * ROTARY_DIM + jk, mask=tkm, other=0.0).to(
        tl.float32
    )
    koff = tokk * RS_K + hkk * HEAD_SIZE + jk
    for hb in range(0, H_K, HBK):
        off = koff + hb * HEAD_SIZE
        x1 = tl.load(k_ld1_ptr + off, mask=tkm, other=0.0).to(tl.float32)
        x2 = tl.load(k_ld2_ptr + off + HALF, mask=tkm, other=0.0).to(tl.float32)
        tl.store(
            k_st1_ptr + off,
            (x1 * cosk - x2 * sink).to(k_st1_ptr.dtype.element_ty),
            mask=tkm,
        )
        tl.store(
            k_st2_ptr + off + HALF,
            (x2 * cosk + x1 * sink).to(k_st2_ptr.dtype.element_ty),
            mask=tkm,
        )


@triton.jit
def _rope_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    cos_ptr,
    q_stride,
    k_stride,
    cos_stride,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HALF: tl.constexpr,
    HALF_BLOCK: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    tok = tl.program_id(0)
    pos_id = tl.load(pos_ptr + tok)
    cos_row = cos_ptr + pos_id * cos_stride

    hoff = tl.arange(0, HALF_BLOCK)
    lane_ok = hoff < HALF
    c = tl.load(cos_row + hoff, mask=lane_ok, other=0.0).to(tl.float32)
    s = tl.load(cos_row + HALF + hoff, mask=lane_ok, other=0.0).to(tl.float32)

    q_ptr += tok * q_stride
    for off_h in range(0, NUM_Q_HEADS):
        base = off_h * HEAD_SIZE
        if IS_NEOX:
            x1 = tl.load(q_ptr + base + hoff, mask=lane_ok, other=0.0).to(
                tl.float32
            )
            x2 = tl.load(q_ptr + base + HALF + hoff, mask=lane_ok, other=0.0).to(
                tl.float32
            )
            tl.store(
                q_ptr + base + hoff,
                (x1 * c - x2 * s).to(q_ptr.dtype.element_ty),
                mask=lane_ok,
            )
            tl.store(
                q_ptr + base + HALF + hoff,
                (x2 * c + x1 * s).to(q_ptr.dtype.element_ty),
                mask=lane_ok,
            )
        else:
            even = base + 2 * hoff
            odd = even + 1
            x1 = tl.load(q_ptr + even, mask=lane_ok, other=0.0).to(tl.float32)
            x2 = tl.load(q_ptr + odd, mask=lane_ok, other=0.0).to(tl.float32)
            tl.store(
                q_ptr + even,
                (x1 * c - x2 * s).to(q_ptr.dtype.element_ty),
                mask=lane_ok,
            )
            tl.store(
                q_ptr + odd,
                (x2 * c + x1 * s).to(q_ptr.dtype.element_ty),
                mask=lane_ok,
            )

    k_ptr += tok * k_stride
    for off_h in range(0, NUM_K_HEADS):
        base = off_h * HEAD_SIZE
        if IS_NEOX:
            x1 = tl.load(k_ptr + base + hoff, mask=lane_ok, other=0.0).to(
                tl.float32
            )
            x2 = tl.load(k_ptr + base + HALF + hoff, mask=lane_ok, other=0.0).to(
                tl.float32
            )
            tl.store(
                k_ptr + base + hoff,
                (x1 * c - x2 * s).to(k_ptr.dtype.element_ty),
                mask=lane_ok,
            )
            tl.store(
                k_ptr + base + HALF + hoff,
                (x2 * c + x1 * s).to(k_ptr.dtype.element_ty),
                mask=lane_ok,
            )
        else:
            even = base + 2 * hoff
            odd = even + 1
            x1 = tl.load(k_ptr + even, mask=lane_ok, other=0.0).to(tl.float32)
            x2 = tl.load(k_ptr + odd, mask=lane_ok, other=0.0).to(tl.float32)
            tl.store(
                k_ptr + even,
                (x1 * c - x2 * s).to(k_ptr.dtype.element_ty),
                mask=lane_ok,
            )
            tl.store(
                k_ptr + odd,
                (x2 * c + x1 * s).to(k_ptr.dtype.element_ty),
                mask=lane_ok,
            )


def _pow2_divisor(n, cap):
    p = 1
    while p * 2 <= n and p * 2 <= cap and n % (p * 2) == 0:
        p *= 2
    return p


def _flat_config(nnz, hq, hk, half):
    """(TPP, HBQ, HBK, num_warps) buckets, scanned on L600 (do_bench median).

    Small nnz favours head-batch flattening (op count is king in the latency
    regime); large nnz favours big TPP / small HB (less redundant cos/sin
    gather traffic and wider streaming ops).
    """
    if nnz <= 16:
        tpp, hbq, hbk, num_warps = 1, 16, 8, 4
    elif nnz <= 96:
        tpp, hbq, hbk, num_warps = 2, 8, 8, 2
    elif nnz <= 320:
        tpp, hbq, hbk, num_warps = 8, 2, 2, 4
    elif nnz <= 1280:
        tpp, hbq, hbk, num_warps = 16, 2, 2, 2
    else:
        tpp, hbq, hbk, num_warps = 32, 1, 1, 2
    hbq = _pow2_divisor(hq, hbq)
    hbk = _pow2_divisor(hk, hbk)
    tpp = min(tpp, 1 << max(nnz.bit_length() - 1, 0))
    while hbq > 1 and tpp * hbq * half > 8192:
        hbq //= 2
    while hbk > 1 and tpp * hbk * half > 8192:
        hbk //= 2
    while tpp > 1 and (tpp * hbq * half > 8192 or tpp * hbk * half > 8192):
        tpp //= 2
    if tpp * hbq * half > 8192 or tpp * hbk * half > 8192:
        return None
    return tpp, hbq, hbk, num_warps


def sglang_fl_rotary_embedding(
    positions, query, key, head_size, cos_sin_cache, is_neox=True
):
    hs = int(head_size)
    rd = int(cos_sin_cache.shape[-1])
    nnz = positions.numel()
    if nnz == 0:
        return query, key
    hq = int(query.shape[1]) // hs
    hk = int(key.shape[1]) // hs
    if hq == 0 and hk == 0:
        return query, key

    half = rd // 2
    half_block = max(triton.next_power_of_2(half), 2)
    positions = _positions_as_int32_payload(positions)

    if (
        bool(is_neox)
        and rd % 2 == 0
        and half >= 1
        and half & (half - 1) == 0
        and query.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and key.dtype == query.dtype
        and positions.dtype in (torch.int32, torch.int64)
        and query.is_contiguous()
        and key.is_contiguous()
        and cos_sin_cache.is_contiguous()
    ):
        cfg = _flat_config(nnz, hq, hk, half)
        if cfg is not None:
            tpp, hbq, hbk, num_warps = cfg
            grid_n = (nnz + tpp - 1) // tpp
            if (
                grid_n <= 65535
                and nnz * max(hq * hs, hk * hs) < 2**31
                and int(cos_sin_cache.shape[0]) * rd < 2**31
            ):
                sin_view = cos_sin_cache[:, half:]
                _rope_flat_kernel[(grid_n,)](
                    query,
                    query,
                    query,
                    query,
                    key,
                    key,
                    key,
                    key,
                    cos_sin_cache,
                    sin_view,
                    positions,
                    nnz,
                    RS_Q=hq * hs,
                    RS_K=hk * hs,
                    ROTARY_DIM=rd,
                    HALF=half,
                    LOG_HALF=half.bit_length() - 1,
                    HEAD_SIZE=hs,
                    H_Q=hq,
                    H_K=hk,
                    HBQ=hbq,
                    HBK=hbk,
                    LOG_TH_Q=(hbq * half).bit_length() - 1,
                    LOG_TH_K=(hbk * half).bit_length() - 1,
                    HBH_Q=hbq * half,
                    HBH_K=hbk * half,
                    TPP=tpp,
                    num_warps=num_warps,
                )
                return query, key

    grid = (nnz,)
    _rope_kernel[grid](
        query,
        key,
        positions,
        cos_sin_cache,
        query.stride(0),
        key.stride(0),
        cos_sin_cache.stride(0),
        NUM_Q_HEADS=hq,
        NUM_K_HEADS=hk,
        HEAD_SIZE=hs,
        HALF=half,
        HALF_BLOCK=half_block,
        IS_NEOX=bool(is_neox),
        num_warps=_NUM_WARPS,
    )
    return query, key


__all__ = ["sglang_fl_rotary_embedding"]
