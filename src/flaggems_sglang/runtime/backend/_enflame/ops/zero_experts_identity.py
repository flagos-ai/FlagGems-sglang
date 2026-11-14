# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""GCU rewrite of zero_experts_identity.

The generic kernel launches one CTA per (token, 1024-wide hidden chunk):
grid = num_tokens * cdiv(hidden, 1024). GCU400 dispatches CTAs at a ~52 ns
floor (compiler_issues/vllm_moe_sum.md), so t=512/hidden=7168 pays
3584 * 52 ns ~= 186 us of pure dispatch on top of a ~14 us memory floor
(measured 200 us, kernel mode do_bench median). Small token counts are
instead pinned by the per-CTA serial chain.

The rewrite follows the proven flat 1D "mt" form (moe_topk_sum /
scatter_to_padded precedent): the (token, col-chunk) space is walked by at
most 48 persistent programs (one per SPC) with a grid stride, and each work
item is a masked [BD] 1D load/store with BD capped at 8192 (16 KB bf16 tile,
4 KB/warp at num_warps=4 -- inside the <=8 KB/warp, <=32 KB/tile safe zone
of the memref_ext.slice_(pad_)start compile boundary).

Two dispatch arms, selected on the host by token count only:
- _kernel (item = one (token, chunk)): best when fewer than ~4*48 tokens,
  where per-CTA parallelism beats batch overhead. Also serves hidden > 8192
  (multiple chunks per row) and any non-pow2 top_k via folded k-masks.
- _batched_kernel (item = four consecutive tokens, text-unrolled): for
  larger token counts it overlaps the four per-token scalar chains
  (idx/scale loads + where/sum) with the four wide load/store pairs.
  The four wide h loads are issued after the metadata loads but BEFORE
  the four where/sum reductions: with the loads placed after the sums
  (2026-11-08 form) the wide DMA issue waits on the reduction tree, which
  cost a measured -3.9% @ t=512 (15.96 -> 15.33 us) and -0.9% @ t=4096
  (84.7 -> 83.9 us, kernel-mode do_bench median, 5x interleaved A/B).
  Tail rows clamp to the last token; the store is a pure function of the
  row, so clamped rewrites are idempotent. Brings t=4096 to ~1.06x the
  pure-copy floor.

Indexing is int32 like the moe_topk_sum rewrite: the safe domain is
num_tokens * hidden < 2^31 (same exposure as the generic kernel).

Measured, official bench shapes (t, top_k=8, hidden=7168, num_experts=32,
bf16), kernel mode do_bench median, old -> new:
t=1: 4.14 -> 4.1 us; t=8: 6.4 -> 4.1 us; t=64: 27.4 -> 5.8 us;
t=512: 200.0 -> 15.4 us (13.0x); t=4096: 1576 -> 83.1 us (19.0x).
2026-11-12 second pass: batched-arm load hoist trims t=512 to ~15.4 us
(official) and t=4096 to ~83.9 us; the wide arm is unchanged (its variants
measured within the bimodal noise band of the tiny-shape regime).
"""

import torch
import triton
import triton.language as tl

# One program per SPC; extra programs only queue behind the ~52 ns/CTA
# dispatch floor (compiler_issues/vllm_moe_sum.md).
_GRID_CAP = 48
# Column-chunk width cap. 8192 bf16 lanes = 16 KB/tile, 4 KB/warp at
# num_warps=4 (safe zone; 8192@nw1 = 16 KB/warp fails to compile, see
# vllm_silu_flat_slice_start.md compile matrix).
_BLOCK_H_MAX = 8192
# U=4 row batch arm kicks in once the batched grid would still fill all
# SPCs (>= 4 * _GRID_CAP items). Below that, single-token items win by
# per-CTA parallelism.
_BATCH_MIN_TOKENS = 4 * _GRID_CAP
_U = 4
_NUM_WARPS = 4


@triton.jit
def _zero_experts_identity_kernel(
    idx_ptr,  # [num_tokens, top_k] int32, contiguous
    scale_ptr,  # [num_tokens, top_k] fp32, contiguous
    hidden_ptr,  # [num_tokens, hidden_dim], contiguous
    out_ptr,  # [num_tokens, hidden_dim], contiguous
    num_tokens,
    num_experts,
    TOPK: tl.constexpr,  # actual top_k (row stride of idx/scale)
    KPAD: tl.constexpr,  # next_pow2(top_k), arange width
    HIDDEN: tl.constexpr,
    NC: tl.constexpr,  # cdiv(hidden_dim, BD)
    BD: tl.constexpr,
):
    pid = tl.program_id(0)
    total = num_tokens * NC
    offs_k = tl.arange(0, KPAD)
    kmask = offs_k < TOPK  # folds away when top_k is a power of two
    for flat in range(pid, total, tl.num_programs(0)):
        token = flat // NC
        cols = (flat % NC) * BD + tl.arange(0, BD)
        mask = cols < HIDDEN
        # Fold this token's routing slots into one fp32 scale total:
        # identity slots (idx >= num_experts) keep their scale, others 0.
        idx = tl.load(idx_ptr + token * TOPK + offs_k, mask=kmask, other=-1)
        scale = tl.load(scale_ptr + token * TOPK + offs_k, mask=kmask,
                        other=0.0)
        total_scale = tl.sum(tl.where(idx >= num_experts, scale, 0.0), axis=0)
        h = tl.load(hidden_ptr + token * HIDDEN + cols, mask=mask, other=0.0)
        tl.store(
            out_ptr + token * HIDDEN + cols,
            (h.to(tl.float32) * total_scale).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit
def _zero_experts_identity_batched_kernel(
    idx_ptr,
    scale_ptr,
    hidden_ptr,
    out_ptr,
    num_tokens,
    num_experts,
    TOPK: tl.constexpr,
    KPAD: tl.constexpr,
    HIDDEN: tl.constexpr,  # must satisfy NC == 1 (HIDDEN <= BD)
    BD: tl.constexpr,
):
    pid = tl.program_id(0)
    nitems = (num_tokens + 3) // 4
    offs_k = tl.arange(0, KPAD)
    kmask = offs_k < TOPK
    cols = tl.arange(0, BD)
    mask = cols < HIDDEN
    for item in range(pid, nitems, tl.num_programs(0)):
        r0 = item * 4
        # Clamp tail rows to the last token: the store is a pure function
        # of the row, so duplicated writes are idempotent.
        t0 = min(r0 + 0, num_tokens - 1)
        t1 = min(r0 + 1, num_tokens - 1)
        t2 = min(r0 + 2, num_tokens - 1)
        t3 = min(r0 + 3, num_tokens - 1)
        i0 = tl.load(idx_ptr + t0 * TOPK + offs_k, mask=kmask, other=-1)
        s0 = tl.load(scale_ptr + t0 * TOPK + offs_k, mask=kmask, other=0.0)
        i1 = tl.load(idx_ptr + t1 * TOPK + offs_k, mask=kmask, other=-1)
        s1 = tl.load(scale_ptr + t1 * TOPK + offs_k, mask=kmask, other=0.0)
        i2 = tl.load(idx_ptr + t2 * TOPK + offs_k, mask=kmask, other=-1)
        s2 = tl.load(scale_ptr + t2 * TOPK + offs_k, mask=kmask, other=0.0)
        i3 = tl.load(idx_ptr + t3 * TOPK + offs_k, mask=kmask, other=-1)
        s3 = tl.load(scale_ptr + t3 * TOPK + offs_k, mask=kmask, other=0.0)
        h0 = tl.load(hidden_ptr + t0 * HIDDEN + cols, mask=mask, other=0.0)
        h1 = tl.load(hidden_ptr + t1 * HIDDEN + cols, mask=mask, other=0.0)
        h2 = tl.load(hidden_ptr + t2 * HIDDEN + cols, mask=mask, other=0.0)
        h3 = tl.load(hidden_ptr + t3 * HIDDEN + cols, mask=mask, other=0.0)
        w0 = tl.sum(tl.where(i0 >= num_experts, s0, 0.0), axis=0)
        w1 = tl.sum(tl.where(i1 >= num_experts, s1, 0.0), axis=0)
        w2 = tl.sum(tl.where(i2 >= num_experts, s2, 0.0), axis=0)
        w3 = tl.sum(tl.where(i3 >= num_experts, s3, 0.0), axis=0)
        tl.store(out_ptr + t0 * HIDDEN + cols,
                 (h0.to(tl.float32) * w0).to(out_ptr.dtype.element_ty),
                 mask=mask)
        tl.store(out_ptr + t1 * HIDDEN + cols,
                 (h1.to(tl.float32) * w1).to(out_ptr.dtype.element_ty),
                 mask=mask)
        tl.store(out_ptr + t2 * HIDDEN + cols,
                 (h2.to(tl.float32) * w2).to(out_ptr.dtype.element_ty),
                 mask=mask)
        tl.store(out_ptr + t3 * HIDDEN + cols,
                 (h3.to(tl.float32) * w3).to(out_ptr.dtype.element_ty),
                 mask=mask)


def zero_experts_identity(
    expert_indices, expert_scales, num_experts, zero_expert_type, hidden_states
):
    num_tokens, top_k = expert_indices.shape
    hidden_dim = hidden_states.shape[1]

    if not (expert_indices.is_contiguous() and expert_scales.is_contiguous()):
        expert_indices = expert_indices.contiguous()
        expert_scales = expert_scales.contiguous()
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

    out = torch.empty_like(hidden_states)

    if num_tokens == 0 or hidden_dim == 0:
        return out

    bd = min(triton.next_power_of_2(hidden_dim), _BLOCK_H_MAX)
    nc = (hidden_dim + bd - 1) // bd
    kpad = triton.next_power_of_2(top_k)

    if nc == 1 and num_tokens >= _BATCH_MIN_TOKENS:
        grid = (min((num_tokens + _U - 1) // _U, _GRID_CAP),)
        _zero_experts_identity_batched_kernel[grid](
            expert_indices,
            expert_scales,
            hidden_states,
            out,
            num_tokens,
            num_experts,
            TOPK=top_k,
            KPAD=kpad,
            HIDDEN=hidden_dim,
            BD=bd,
            num_warps=_NUM_WARPS,
        )
    else:
        grid = (min(num_tokens * nc, _GRID_CAP),)
        _zero_experts_identity_kernel[grid](
            expert_indices,
            expert_scales,
            hidden_states,
            out,
            num_tokens,
            num_experts,
            TOPK=top_k,
            KPAD=kpad,
            HIDDEN=hidden_dim,
            NC=nc,
            BD=bd,
            num_warps=_NUM_WARPS,
        )
    return out


__all__ = ["zero_experts_identity"]
