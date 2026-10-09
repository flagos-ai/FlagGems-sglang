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

"""attention/seqlens_expand: expand per-request (qo_len, kv_len) pairs into
per-query-token causal KV lengths.

The reference loops over batch entries in Python, launching several tiny kernels
(arange / add / clamp / copy) per request — pure launch-overhead bound. The
Triton version replaces that with exactly one launch:

- N <= 1024: a single-CTA fused kernel, grid ``(1, q-tiles)``. It loads the
  whole ``(qo, kv)`` pair vector at once, derives the exclusive cumsum of
  ``qo`` in registers via ``tl.cumsum``, and fills a 2D
  ``(request, q-token)`` output tile — the per-row ranges are disjoint so the
  masked store is conflict free. ``BLOCK_Q`` is capped so very large
  ``max_q_len`` still splits across the second grid dimension instead of
  blowing up the register tile. Measured on Iluvatar, widening to 16-32 warps
  and extending this path up to N=1024 beats a multi-CTA grid (the cumsum
  recompute per program costs more than the wider tile store).
- N > 1024: a multi-CTA kernel with grid ``(ceil(N / CHUNK), q-tiles)``. Each
  program recomputes the global exclusive prefix of its chunk by scanning
  ``qo[0 : chunk_start)`` itself (a few KB of int32 that every program hits in
  L2 — far cheaper than a separate serial-cumsum launch plus workspace), then
  does a local ``tl.cumsum`` inside its chunk and fills a 2D
  ``(chunk-request, q-token)`` output tile. ``CHUNK=512`` with 32 warps
  measured best across N in [2048, 4096] on Iluvatar (16 SMs): 16 CTAs fill
  the machine exactly once, and the shallower per-program prefix rescan beats
  both narrower chunks (more CTAs than SMs, longer serial scan per program)
  and wider ones (less parallelism inside each program).

``clamp(..., min=0)`` is computed in-kernel with ``tl.maximum``: DP-padded /
idle rows can have ``kv_len < qo_len`` and downstream consumers read these
lengths as uint32, so negatives must never leak through.

Loads and stores use ``evict_first`` for streamed data (each program's chunk
is touched once); the prefix-scan region is re-read by every program, so it
uses ``evict_last`` to stay resident in L2.
"""

import torch
import triton
import triton.language as tl

import flaggems_sglang  # noqa: F401


@triton.jit
def _seqlens_expand_fused_kernel(
    qo_ptr,
    kv_ptr,
    out_ptr,
    n_req,
    BLOCK_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    # Single-CTA path: the whole batch fits in one block, offsets never
    # touch memory. Grid dim 1 tiles the q-token range so a huge max_q_len
    # cannot inflate the register tile.
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_req
    qo = tl.load(
        qo_ptr + offs_n, mask=mask_n, other=0, eviction_policy="evict_first"
    )
    kv = tl.load(
        kv_ptr + offs_n, mask=mask_n, other=0, eviction_policy="evict_first"
    )
    offsets = tl.cumsum(qo, axis=0) - qo  # exclusive cumsum
    offs_q = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    # vals[r, c] = clamp(kv[r] - qo[r] + 1 + c, min=0); rows are disjoint in
    # the output so the flattened masked store below is conflict free.
    vals = tl.maximum(kv[:, None] - qo[:, None] + 1 + offs_q[None, :], 0)
    mask = mask_n[:, None] & (offs_q[None, :] < qo[:, None])
    out_offs = offsets[:, None] + offs_q[None, :]
    tl.store(
        out_ptr + out_offs, vals, mask=mask, eviction_policy="evict_first"
    )


@triton.jit
def _seqlens_expand_multi_kernel(
    qo_ptr,
    kv_ptr,
    out_ptr,
    n_req,
    CHUNK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_SCAN: tl.constexpr,
):
    # Multi-CTA path: program (b, t) owns requests [b*CHUNK, (b+1)*CHUNK) and
    # q-token tile t. The global exclusive prefix of the chunk is recomputed
    # from qo directly — the prefix region is a few KB of int32 that stays hot
    # in L2 across programs, so this beats a separate cumsum launch + workspace
    # round-trip.
    pid_n = tl.program_id(0)
    my_start = pid_n * CHUNK

    base = tl.zeros((), dtype=tl.int32)
    for start in range(0, my_start, BLOCK_SCAN):
        offs_s = start + tl.arange(0, BLOCK_SCAN)
        qo_s = tl.load(
            qo_ptr + offs_s,
            mask=offs_s < my_start,
            other=0,
            eviction_policy="evict_last",
        )
        base += tl.sum(qo_s, axis=0)

    offs_c = my_start + tl.arange(0, CHUNK)
    mask_c = offs_c < n_req
    qo = tl.load(
        qo_ptr + offs_c, mask=mask_c, other=0, eviction_policy="evict_first"
    )
    kv = tl.load(
        kv_ptr + offs_c, mask=mask_c, other=0, eviction_policy="evict_first"
    )
    offsets = base + tl.cumsum(qo, axis=0) - qo  # global exclusive cumsum

    offs_q = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    vals = tl.maximum(kv[:, None] - qo[:, None] + 1 + offs_q[None, :], 0)
    mask = mask_c[:, None] & (offs_q[None, :] < qo[:, None])
    out_offs = offsets[:, None] + offs_q[None, :]
    tl.store(
        out_ptr + out_offs, vals, mask=mask, eviction_policy="evict_first"
    )


def _next_pow2(x):
    return 1 << max(0, int(x) - 1).bit_length()


def seqlens_expand(extend_seq_lens, seq_lens, total_len, max_q_len):
    device = extend_seq_lens.device
    n = extend_seq_lens.numel()
    out = torch.empty(total_len, dtype=torch.int32, device=device)
    if n == 0 or total_len == 0:
        return out

    qo = (
        extend_seq_lens
        if extend_seq_lens.is_contiguous()
        else extend_seq_lens.contiguous()
    )
    kv = seq_lens if seq_lens.is_contiguous() else seq_lens.contiguous()
    max_q = max(1, int(max_q_len))

    if n <= 1024:
        # Small/medium batches: one fused single-CTA launch (see module
        # docstring). BLOCK_Q is capped so the register tile stays bounded and
        # large max_q_len splits over grid dim 1 instead. num_warps is graded
        # by register-tile size: tiny tiles (N<=8, <=256 elems) launch fastest
        # with a single warp, while the 64x16 tile wants 16 warps and the
        # 512x16 tile saturates best at 32 warps (measured on Iluvatar: 16/32
        # warps shave the tail of the masked 2D store for mid-size tiles).
        block_n = max(16, _next_pow2(n))
        block_q = min(64, max(16, _next_pow2(max_q)))
        elems = block_n * block_q
        if elems <= 256:
            num_warps = 1
        elif elems <= 1024:
            # 64x16 tile: 16 warps beat 4 (one warp per output row-group;
            # the masked store tail is the bottleneck, not launch width).
            num_warps = 16
        elif elems <= 2048:
            num_warps = 8
        elif elems <= 4096:
            num_warps = 16
        else:
            num_warps = 32
        grid = (1, triton.cdiv(max_q, block_q))
        _seqlens_expand_fused_kernel[grid](
            qo,
            kv,
            out,
            n,
            BLOCK_N=block_n,
            BLOCK_Q=block_q,
            num_warps=num_warps,
        )
    else:
        # Large batches: one multi-CTA launch, parallel over both batch
        # chunks and q-token tiles (see module docstring).
        block_q = min(64, max(16, _next_pow2(max_q)))
        grid = (triton.cdiv(n, 512), triton.cdiv(max_q, block_q))
        _seqlens_expand_multi_kernel[grid](
            qo,
            kv,
            out,
            n,
            CHUNK=512,
            BLOCK_Q=block_q,
            BLOCK_SCAN=4096,
            num_warps=32,
        )
    return out


__all__ = ["seqlens_expand"]
