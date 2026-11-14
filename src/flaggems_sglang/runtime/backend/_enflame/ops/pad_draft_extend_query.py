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

"""pad_draft_extend_query — enflame (L600) persistent scatter (v10).

Contract (unchanged since v8): ``out`` is ``padded_q`` itself, only the live
prefix rows ``out[b, :s] = q[cu[b]:cu[b]+s]`` are written, the tail is never
touched.

v9 recap (2026-11-08): the v8 one-program-per-row grid was retirement-bound
(~15.5 programs/us); v9 turned each request into one flat masked copy and put
a grid-stride loop over ``W = bs * NCHUNK`` items on at most 48 programs
(one L600 wave).  Core bench: bs64 33.3->5.0us, bs256 126.9->9.0us (14x).

v10 (E-1 second round, 2026-11-13) keeps the v9 decomposition and attacks the
remaining serial cost of the masked load/store pair chain:

* The live prefix moves as **i32 words** (kernel-side pointer cast; host only
  checks the 2-byte dtype, even HD and 4-byte base alignment): the same
  bytes with half the lanes per op.
* Each loop iteration owns items ``w`` and ``w + nprog`` with **both masked
  loads issued before the stores**, halving the per-program iteration count.

Attribution (interleaved do_bench, w1000/r100, GCU400): the chain phase runs
at 1.84-2.03 TB/s aggregate = 92-101% of the 2D-tile copy roofline, and the
~4.1us do_bench floor dominates bs<=8, so core shapes are at the structural
floor; the gain appears where the chain is long enough to amortize the wave
ramp — bs1024 22.7->21.7us (-4.5%), bs256 9.03->8.90us (-1.4%), bs64 and
below WITHIN-NOISE.

Probed and rejected in the second round (do not retry): grid 96 (+2.3us —
wave-2 programs serialize behind the retire rate), 64KB-per-op fused
two-request items (2.2-2.4x — the 32KB/op sweet spot is a hard cliff), i32
punning alone / U2 batching alone (each ~-3% at bs1024 only), cross-iteration
software pipelining (blocked by compiler asserts, P02), guard branches and
full-row fast arms (v9 round already showed +12% / +-0).

Fallbacks: odd HD, non-2-byte dtypes, unaligned views or requests wider than
one 16K-word op use the v9 persistent kernel verbatim; requests wider than 64
chunks keep the v8 legacy kernel.
"""

import triton
import triton.language as tl

__all__ = ["pad_draft_extend_query"]

_MAX_CHUNKS = 64  # unrolled-loop budget (Kunlunxin pipeline-event limit)
_MAX_BLOCK = 16384
_NUM_SPC = 48  # L600: 24 SIPs x 2 concurrent programs = one full wave


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
    LOG_TPB: tl.constexpr,
):
    # One program per padded slot; the slot maps back to (b, t) with a shift.
    pid = tl.program_id(axis=0)
    b = pid >> LOG_TPB
    t = pid - (b << LOG_TPB)
    if t < MSL:
        s = tl.load(sl_ptr + b)
        if t < s:
            # Live row only.  Its source row is cu_seqlens_q[b] + t, and `out` is
            # `padded_q` itself, so the tail needs no work at all.
            dst_base = (b * MSL + t) * HD
            beg = tl.load(cu_ptr + b)
            src_base = (beg + t) * HD
            lanes = tl.arange(0, BLOCK)
            for c in tl.static_range(NCHUNK):
                off = c * BLOCK + lanes
                m = off < HD
                v = tl.load(q_ptr + src_base + off, mask=m, other=0)
                tl.store(out_ptr + dst_base + off, v, mask=m)


@triton.jit
def _pad_draft_extend_query_persist(
    q_ptr,
    cu_ptr,
    sl_ptr,
    out_ptr,
    work,
    MSL: tl.constexpr,
    HD: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    # Grid-stride over W = bs * NCHUNK work items, at most one wave of
    # programs.  Work item w owns chunk `w % NCHUNK` of request `w // NCHUNK`;
    # with NCHUNK == 1 (the common case, whole request in one op) the decode
    # folds away.  Each item is one contiguous flat copy of the request's
    # live prefix, masked to `s * HD` elements: dead rows are never written,
    # so `padded_q`'s tail survives bit-identical.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    for w in range(pid, work, nprog):
        if NCHUNK == 1:
            b = w
            c0 = 0
        else:
            b = w // NCHUNK
            c0 = (w - b * NCHUNK) * BLOCK
        s = tl.load(sl_ptr + b)
        beg = tl.load(cu_ptr + b)
        n = s * HD
        off = c0 + lanes
        m = off < n
        v = tl.load(q_ptr + beg * HD + off, mask=m, other=0)
        tl.store(out_ptr + b * (MSL * HD) + off, v, mask=m)


@triton.jit
def _pad_draft_extend_query_persist_u32(
    q_ptr,
    cu_ptr,
    sl_ptr,
    out_ptr,
    work,
    MSL: tl.constexpr,
    HD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # v10 fast path (NCHUNK == 1 only): the flat copy moves i32 words (HD is
    # even, bases are 4-byte aligned — the host guards this), and one
    # iteration owns requests w and w + nprog: both masked loads issue before
    # the stores (source order is the scheduling hint), halving the iteration
    # count per program.  The odd tail item degenerates to a plain v9 item.
    qi = q_ptr.to(tl.pointer_type(tl.int32))
    oi = out_ptr.to(tl.pointer_type(tl.int32))
    H2: tl.constexpr = HD // 2
    N2: tl.constexpr = MSL * H2
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK)
    for w in range(pid, work, 2 * nprog):
        sA = tl.load(sl_ptr + w)
        begA = tl.load(cu_ptr + w)
        nA = sA * H2
        offA = lanes
        mA = offA < nA
        vA = tl.load(qi + begA * H2 + offA, mask=mA, other=0)
        if w + nprog < work:
            wB = w + nprog
            sB = tl.load(sl_ptr + wB)
            begB = tl.load(cu_ptr + wB)
            nB = sB * H2
            offB = lanes
            mB = offB < nB
            vB = tl.load(qi + begB * H2 + offB, mask=mB, other=0)
            tl.store(oi + wB * N2 + offB, vB, mask=mB)
        tl.store(oi + w * N2 + offA, vA, mask=mA)


def pad_draft_extend_query(q, padded_q, seq_lens_q, cu_seqlens_q):
    """Scatter the ragged draft query rows into the padded buffer."""
    q = q.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    seq_lens_q = seq_lens_q.contiguous()

    total_tokens, num_heads, head_dim = q.shape
    bs, max_seq_len, h2, d2 = padded_q.shape
    # v8 writes IN PLACE: the padded buffer is the output.  The reference clones
    # only to avoid overwriting its input; returning the same values in the same
    # buffer is equivalent, and every element outside the live prefix is left
    # untouched.
    out = padded_q
    if bs == 0 or max_seq_len == 0 or num_heads == 0 or head_dim == 0:
        return out

    hd = num_heads * head_dim

    # Persistent wave over per-request flat copies (see module docstring).
    nmax = max_seq_len * hd
    block = min(triton.next_power_of_2(max(nmax, 16)), _MAX_BLOCK)
    nchunk = triton.cdiv(nmax, block)
    if nchunk <= _MAX_CHUNKS:
        # v10 fast path: i32-punned pair-batched wave, one op per request.
        if (
            q.element_size() == 2
            and hd % 2 == 0
            and q.data_ptr() % 4 == 0
            and out.data_ptr() % 4 == 0
            and triton.cdiv(nmax // 2, _MAX_BLOCK) <= 1
        ):
            block2 = min(triton.next_power_of_2(max(nmax // 2, 16)), _MAX_BLOCK)
            grid = min(bs, _NUM_SPC)
            _pad_draft_extend_query_persist_u32[(grid,)](
                q,
                cu_seqlens_q,
                seq_lens_q,
                out,
                bs,
                MSL=max_seq_len,
                HD=hd,
                BLOCK=block2,
                num_warps=max(1, block2 // 2048),
            )
            return out

        # v9 fallback: bf16 lanes, one chunk per item.
        work = bs * nchunk
        grid = min(work, _NUM_SPC)
        _pad_draft_extend_query_persist[(grid,)](
            q,
            cu_seqlens_q,
            seq_lens_q,
            out,
            work,
            MSL=max_seq_len,
            HD=hd,
            BLOCK=block,
            NCHUNK=nchunk,
            num_warps=max(1, block // 2048),
        )
        return out

    block = min(triton.next_power_of_2(max(hd, 16)), _MAX_BLOCK)
    nchunk = triton.cdiv(hd, block)
    if (
        nchunk > _MAX_CHUNKS
    ):  # keep the unrolled loop inside the pipeline budget
        block = triton.next_power_of_2(hd)
        nchunk = 1

    log_tpb = max(0, (max_seq_len - 1).bit_length())

    _pad_draft_extend_query_kernel[(bs << log_tpb,)](
        q,
        cu_seqlens_q,
        seq_lens_q,
        out,
        MSL=max_seq_len,
        HD=hd,
        BLOCK=block,
        NCHUNK=nchunk,
        LOG_TPB=log_tpb,
        num_warps=4,
    )
    return out
