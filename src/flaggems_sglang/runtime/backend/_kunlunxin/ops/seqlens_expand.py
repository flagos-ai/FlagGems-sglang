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

"""seqlens_expand: expand per-request ``(qo_len, kv_len)`` pairs into a
per-query-token causal KV-length vector of shape ``[total_len]``:

    out[offset[i] : offset[i] + qo_len[i]]
        = clamp(kv_len[i] - qo_len[i] + 1 + arange(qo_len[i]), min=0)
    offset = exclusive_cumsum(extend_seq_lens)

The reference implementation is a Python loop over the batch with one
``int(tensor[i])`` device->host sync per request plus a tiny ``torch.arange``
launch per request — cost grows linearly with N.  Here the whole thing is two
kernel launches regardless of N:

- ``_offsets_kernel``: a single program computes the exclusive prefix sum of
  ``extend_seq_lens`` with ``tl.cumsum`` over BLOCK-sized tiles, carrying the
  running total across tiles.  N is at most a few thousand in practice, so a
  one-CTA scan is far cheaper than a multi-CTA two-pass scan, and it removes
  every host sync from the hot path.
- ``_expand_kernel``: one program per request streams ``BLOCK``-sized chunks
  of that request's query range, computing
  ``max(kv - qo + 1 + offs, 0)`` directly — no ``arange`` tensor, no clamp
  pass.  Writes are contiguous int32, so they are fully coalesced.  The
  chunk-loop bound is dynamic (``cdiv(qo, BLOCK)``), so any request length is
  handled by one compiled kernel; ``qo == 0`` requests simply write nothing.

Host-side launch parameters are a pure function of the batch size.  No
caching, no module-level state.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _offsets_kernel(
    lens_ptr,  # [N] int32
    offsets_ptr,  # [N] int32, exclusive prefix sum output
    N,
    BLOCK: tl.constexpr,
):
    carry = 0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        lens = tl.load(lens_ptr + offs, mask=mask, other=0)
        inc = tl.cumsum(lens, axis=0)
        # exclusive offset for element i is carry + sum of lens[<i]
        tl.store(offsets_ptr + offs, carry + inc - lens, mask=mask)
        carry += tl.sum(lens, axis=0)


@triton.jit
def _expand_kernel(
    qo_ptr,  # [N] int32, per-request query lengths
    kv_ptr,  # [N] int32, per-request kv lengths
    offsets_ptr,  # [N] int32, exclusive prefix sum of qo
    out_ptr,  # [total_len] int32
    SPLIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0)
    qo = tl.load(qo_ptr + i)
    kv = tl.load(kv_ptr + i)
    base = tl.load(offsets_ptr + i)
    # first token of request i sees kv - qo + 1 previous kv positions
    start_val = kv - qo + 1
    dst = out_ptr + base
    # round-robin chunk assignment: part p takes chunks p, p+SPLIT, ...
    part = tl.program_id(1)
    for lb in range(part, tl.cdiv(qo, BLOCK), SPLIT):
        offs = lb * BLOCK + tl.arange(0, BLOCK)
        mask = offs < qo
        vals = tl.maximum(start_val + offs, 0)
        tl.store(dst + offs, vals, mask=mask)


def seqlens_expand(extend_seq_lens, seq_lens, total_len, max_q_len):
    device = extend_seq_lens.device
    n = extend_seq_lens.shape[0]
    out = torch.empty(total_len, dtype=torch.int32, device=device)
    if n == 0 or total_len == 0:
        return out

    offsets = torch.empty(n, dtype=torch.int32, device=device)
    N_BLOCK = 1024
    _offsets_kernel[(1,)](
        extend_seq_lens,
        offsets,
        n,
        BLOCK=N_BLOCK,
        num_warps=4,
        num_stages=1,
    )

    # Small batches launch few programs; fan the chunk axis out so the tail
    # cases (n == 1) still have parallel work.  Large batches get enough
    # programs from one-per-request alone.
    if n <= 8:
        SPLIT = 4
    else:
        SPLIT = 1
    grid = (n, SPLIT)
    _expand_kernel[grid](
        extend_seq_lens,
        seq_lens,
        offsets,
        out,
        SPLIT=SPLIT,
        BLOCK=64,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["seqlens_expand"]
