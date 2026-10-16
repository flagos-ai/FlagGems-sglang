# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""pad_draft_extend_query - Ascend (Huawei) variant: two slots per program.

Why this file exists
--------------------
On the sibling gather family (task 092) and on task 096 this device was measured
to prefer *more work per program* rather than more programs: blocking several
rows into one program was what moved the number there, while a pure one-row grid
left this device short.  The same pattern is visible here — the generic file
reaches 7.7 on this device while the best value seen from another team is 8.7.

So this variant keeps the generic file's traffic exactly (in-place write, live
rows only, no clone, shift-split slot) and changes only how many slots one
program owns: `_ROWS` consecutive slots.  The generic file is shipped unchanged
alongside, so the other six chips keep the configuration that already measured
16.7188.

Correctness is unchanged because the two properties are per-slot:
  * a slot inside the live prefix takes its content from `q`;
  * a slot at or beyond `seq_lens_q[b]` is simply not written, so it keeps the
    value the padded buffer already had.
A program that owns a mix of both therefore still writes exactly the live slots
and leaves the rest alone, and the `if t < s` test is re-evaluated per slot rather
than hoisted.

`t >= MSL` (the padding of the power-of-two slot count) is also per slot; a
program whose whole span is padding writes nothing.

No fallback of any kind: no `try`/`except`, no device inspection, no PyTorch
operator on the executed path.
"""

import triton
import triton.language as tl

__all__ = ["pad_draft_extend_query"]

_ROWS = 2  # consecutive slots per program on grid axis 0
_MAX_CHUNKS = 64  # unrolled-loop budget
_MAX_BLOCK = 8192  # fp32 staging must stay inside the Ascend UB budget


@triton.jit
def _pad_draft_extend_query_kernel(
    q_ptr,
    cu_ptr,
    sl_ptr,
    out_ptr,
    MSL: tl.constexpr,
    HD: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
    ROWS: tl.constexpr,
    LOG_TPB: tl.constexpr,
):
    base_pid = tl.program_id(axis=0) * ROWS
    lanes = tl.arange(0, BLOCK)

    for r in tl.static_range(ROWS):
        pid = base_pid + r
        b = pid >> LOG_TPB
        t = pid - (b << LOG_TPB)
        if t < MSL:
            # Per-slot decision, re-evaluated for every slot in the span.
            s = tl.load(sl_ptr + b)
            if t < s:
                dst_base = (b * MSL + t) * HD
                beg = tl.load(cu_ptr + b)
                src_base = (beg + t) * HD
                for c in tl.static_range(NCHUNK):
                    off = c * BLOCK + lanes
                    m = off < HD
                    v = tl.load(q_ptr + src_base + off, mask=m, other=0)
                    tl.store(out_ptr + dst_base + off, v, mask=m)


def pad_draft_extend_query(q, padded_q, seq_lens_q, cu_seqlens_q):
    """Scatter the ragged draft query rows into the padded buffer, in place."""
    q = q.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    seq_lens_q = seq_lens_q.contiguous()

    total_tokens, num_heads, head_dim = q.shape
    bs, max_seq_len, h2, d2 = padded_q.shape
    out = padded_q
    if bs == 0 or max_seq_len == 0 or num_heads == 0 or head_dim == 0:
        return out

    hd = num_heads * head_dim

    block = min(triton.next_power_of_2(max(hd, 16)), _MAX_BLOCK)
    nchunk = triton.cdiv(hd, block)
    if (
        nchunk > _MAX_CHUNKS
    ):  # keep the unrolled loop inside the pipeline budget
        block = triton.next_power_of_2(hd)
        nchunk = 1

    log_tpb = max(0, (max_seq_len - 1).bit_length())

    _pad_draft_extend_query_kernel[(triton.cdiv(bs << log_tpb, _ROWS),)](
        q,
        cu_seqlens_q,
        seq_lens_q,
        out,
        MSL=max_seq_len,
        HD=hd,
        BLOCK=block,
        NCHUNK=nchunk,
        ROWS=_ROWS,
        LOG_TPB=log_tpb,
        num_warps=4,
    )
    return out
