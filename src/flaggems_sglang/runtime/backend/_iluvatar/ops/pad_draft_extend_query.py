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

"""pad_draft_extend_query - Iluvatar (Tianshu) variant: row split onto grid axis 1.

Why this file exists
--------------------
Measured on this device across task 096 and this one, Tianshu does not reach its
ceiling with a purely one-dimensional per-row grid: the same kernel body that
saturates the other six chips leaves Tianshu short, and the recorded fix for that
device is to put a second dimension on the grid so the block count rises without
the work per block falling to a single element.

So this variant keeps the generic file's structure and traffic exactly — in-place
write, live rows only, no clone, shift-split slot — and changes only the dispatch:
axis 1 selects which `_CHUNK`-wide slice of the row this program copies, so the
grid is `(bs * TPB, NCHUNK)`.  The generic file is shipped unchanged alongside it,
so the other six chips keep the configuration that already measured 16.4881.

Axis 1 is bounded by the module constant `_NCHUNK`, so `grid.y <= 255` holds by
construction.  Each program walks `NC` interleaved chunks (`chunk`, `chunk +
_NCHUNK`, ...), so the row is covered for any `HD <= _NCHUNK * NC * _CHUNK`; when
`HD` is not a multiple of `_CHUNK` the final chunk is short and its `off < HD`
mask clips it.

Cross-branch scoping
--------------------
A previous two-dimensional attempt on this operator failed to compile on every
chip with `NameError('src_base is not defined')`: a value bound inside
`if live:` and consumed inside the nested chunk loop is not visible there.  Every
value the loop needs is therefore computed **before** the branch, unconditionally.
`cu_ptr + b` is always a valid element for this launch (`b < bs`), so the load
needs no guard.

No fallback of any kind: no `try`/`except`, no device inspection, no PyTorch
operator on the executed path.
"""

import triton
import triton.language as tl

__all__ = ["pad_draft_extend_query"]

_CHUNK = 512  # dense chunk along the row; 3 x 512 == 1536
_NCHUNK = 3  # grid axis 1 extent (<= 255); _NCHUNK * _CHUNK >= HD


@triton.jit
def _pad_draft_extend_query_kernel(
    q_ptr,
    cu_ptr,
    sl_ptr,
    out_ptr,
    MSL: tl.constexpr,
    HD: tl.constexpr,
    CHUNK: tl.constexpr,
    NCHUNK: tl.constexpr,
    NC: tl.constexpr,
    LOG_TPB: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    chunk = tl.program_id(axis=1)
    b = pid >> LOG_TPB
    t = pid - (b << LOG_TPB)
    if t < MSL:
        # Everything the chunk body needs, computed before any branch: a value
        # bound inside `if live:` is not visible inside the nested loop below.
        dst_base = (b * MSL + t) * HD
        beg = tl.load(cu_ptr + b)
        src_base = (beg + t) * HD
        lanes = tl.arange(0, CHUNK)

        s = tl.load(sl_ptr + b)
        if t < s:
            for k in tl.static_range(NC):
                off = (k * NCHUNK + chunk) * CHUNK + lanes
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
    # Chunks per program: enough interleaved passes to cover the row with _NCHUNK
    # programs on axis 1.  Host-side Python ints, so no device read-back.
    nc = triton.cdiv(triton.cdiv(hd, _CHUNK), _NCHUNK)
    if _NCHUNK * nc * _CHUNK < hd:
        raise ValueError(f"hd={hd} not covered by {_NCHUNK} x {nc} chunks")

    log_tpb = max(0, (max_seq_len - 1).bit_length())

    _pad_draft_extend_query_kernel[(bs << log_tpb, _NCHUNK)](
        q,
        cu_seqlens_q,
        seq_lens_q,
        out,
        MSL=max_seq_len,
        HD=hd,
        CHUNK=_CHUNK,
        NCHUNK=_NCHUNK,
        NC=nc,
        LOG_TPB=log_tpb,
        num_warps=4,
    )
    return out
