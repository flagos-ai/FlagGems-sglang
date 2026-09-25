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

"""Triton kernel for attention/seqlens_expand.

Expand per-request ``(qo_len, kv_len)`` pairs into a per-query-token vector of
visible causal KV lengths for an extend batch:

    out[offset[i] + t] = clamp(kv_len[i] - qo_len[i] + 1 + t, min=0)
    offset = exclusive_cumsum(extend_seq_lens),  t in [0, qo_len[i])

The ``clamp(min=0)`` is load-bearing: DP-padded / idle rows can carry
``kv_len < qo_len`` and downstream consumers read these lengths as uint32,
so a negative value would wrap to ~4e9 tokens and cause illegal access.

v3 — fused row-loop kernel: one launch, one program per 256 rows
----------------------------------------------------------------

The reference is a Python loop over the batch with several kernel launches
*per request*, plus a host sync per request — O(N) launches and O(N) device
round-trips. v1 collapsed that to two launches; v2 fused the exclusive prefix
sum into the expand kernel for small batches (each program scanned ``[0, i)``
with one BLOCK_N-wide load).

The dominant cost left at large batch was **program count**, not arithmetic:
on the target device each program costs roughly 60 ns of scheduling, so the
old one-program-per-request expand (grid = n) alone accounted for ~250 us at
n=4096 — about two thirds of the whole op. Fusing the cumsum into that grid
could not fix it, because the fused path still launched n programs.

v3 instead gives each program **many rows**: the kernel runs a grid of
``cdiv(n, ROWS)`` programs (R = 256), and every program

  * computes its base offset with a chunked exclusive prefix scan of the
    preceding rows (``BLOCK_N``-wide masked chunks, ``tl.sum`` per chunk —
    O(N^2/ROWS) redundant cache-resident int32 loads, cheap relative to the
    launch it removes), and
  * walks its own rows sequentially, keeping the running offset in a scalar,
    so each row is a couple of scalar loads, one vector ramp, and one store.

The row loop replaces both the per-program scan (one program per row) and the
``offsets`` buffer, so the op stays at a **single launch, single allocation**
for every batch size up to ``_FUSED_MAX_N``; only batches so large that
O(N^2/ROWS) would genuinely outgrow one launch keep the v1 two-kernel path.

``max_q_len`` (host int, part of the op contract) sizes the ramp; the kernel
re-checks every lane against the device-side ``qo_len[i]``, so correctness
never depends on the host value beyond grid/ramp coverage.

Both kernels are portable pure Triton: 1D masked integer math, no vendor
intrinsics, no torch op in the core compute path. The only torch usage is
the ``torch.empty`` output allocation the kernels then fill. Device is
taken from the input tensors themselves — never hardcoded.

Correctness: exact integer arithmetic (int32), bit-identical to the
reference including the ``clamp`` saturation for idle rows.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Rows handled by one program in the fused row-loop path.
_FUSED_ROWS = 256
# Prefix-scan chunk width: bounds the register footprint of the O(N^2/ROWS)
# redundant cache-resident scan regardless of batch size.
_FUSED_BLOCK_N = 256
# Batch-size ceiling for the fused single-launch path.
_FUSED_MAX_N = 32768
# Ramp-width ceiling for the fused path (one store per row).
_FUSED_MAX_Q = 1024


@triton.jit
def _seqlens_expand_fused_kernel(
    qo_ptr,  # [n] int32, query lengths per request
    kv_ptr,  # [n] int32, kv lengths per request
    out_ptr,  # [total_len] int32
    n,  # runtime batch size
    BLOCK_N: tl.constexpr,  # prefix-scan chunk width
    R: tl.constexpr,  # rows per program
    BLOCK_Q: tl.constexpr,  # next_pow2(max_q_len), covers the ramp
):
    p = tl.program_id(0)
    hi = p * R

    # Exclusive base offset without a separate cumsum pass: re-sum the (tiny,
    # cache-resident) query lengths of all preceding rows in BLOCK_N-wide
    # chunks, so register pressure stays bounded no matter how large n grows.
    base = tl.zeros((), dtype=tl.int32)
    r = tl.arange(0, BLOCK_N)
    for s in tl.range(0, hi, BLOCK_N):
        idx = s + r
        base += tl.sum(tl.load(qo_ptr + idx, mask=idx < hi, other=0), axis=0)

    # Walk this program's rows, carrying the running offset in a scalar.
    t = tl.arange(0, BLOCK_Q)[None, :]
    for i in tl.range(hi, tl.minimum(hi + R, n)):
        qo = tl.load(qo_ptr + i)
        kv = tl.load(kv_ptr + i)
        # Local causal length ramp, saturated at 0 for idle / DP-padded rows.
        val = tl.maximum(kv - qo + 1 + t, 0)
        tl.store(out_ptr + base + t, val, mask=t < qo)
        base += qo


@triton.jit
def _exclusive_cumsum_kernel(
    x_ptr,  # [n] int32, per-request query lengths
    out_ptr,  # [n] int32, exclusive prefix sum
    N,  # runtime batch size
    BLOCK: tl.constexpr,
):
    run = tl.zeros((), dtype=tl.int32)
    for start in tl.range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < N
        v = tl.load(x_ptr + offs, mask=m, other=0)
        c = tl.cumsum(v, axis=0)
        # Exclusive prefix: inclusive cumsum minus the current element,
        # plus the carry from all previous chunks.
        tl.store(out_ptr + offs, run + c - v, mask=m)
        run += tl.sum(v, axis=0)


@triton.jit
def _seqlens_expand_kernel(
    qo_ptr,  # [n] int32, query lengths per request
    kv_ptr,  # [n] int32, kv lengths per request
    off_ptr,  # [n] int32, exclusive prefix sum of qo
    out_ptr,  # [total_len] int32
    BLOCK_Q: tl.constexpr,
):
    i = tl.program_id(0)  # request
    c = tl.program_id(1)  # qo chunk within the request
    qo = tl.load(qo_ptr + i)
    kv = tl.load(kv_ptr + i)
    off = tl.load(off_ptr + i)

    t = c * BLOCK_Q + tl.arange(0, BLOCK_Q)
    m = t < qo
    # Local causal length ramp, saturated at 0 for idle / DP-padded rows.
    val = tl.maximum(kv - qo + 1 + t, 0)
    tl.store(out_ptr + off + t, val, mask=m)


def seqlens_expand(extend_seq_lens, seq_lens, total_len, max_q_len):
    out = torch.empty(
        total_len, dtype=torch.int32, device=extend_seq_lens.device
    )
    n = extend_seq_lens.numel()
    if n == 0 or total_len == 0:
        return out

    if n <= _FUSED_MAX_N and max_q_len <= _FUSED_MAX_Q:
        # Single launch: fold the exclusive cumsum into the expand kernel.
        block_q = max(triton.next_power_of_2(int(max_q_len)), 1)
        _seqlens_expand_fused_kernel[(triton.cdiv(n, _FUSED_ROWS),)](
            extend_seq_lens,
            seq_lens,
            out,
            n,
            BLOCK_N=_FUSED_BLOCK_N,
            R=_FUSED_ROWS,
            BLOCK_Q=block_q,
            num_warps=4,
            num_stages=1,
        )
        return out

    offsets = torch.empty(n, dtype=torch.int32, device=extend_seq_lens.device)
    _exclusive_cumsum_kernel[(1,)](
        extend_seq_lens,
        offsets,
        n,
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )

    block_q = 16
    num_q = triton.cdiv(max(int(max_q_len), 1), block_q)
    _seqlens_expand_kernel[(n, num_q)](
        extend_seq_lens,
        seq_lens,
        offsets,
        out,
        BLOCK_Q=block_q,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["seqlens_expand"]
