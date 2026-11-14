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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GCU rewrite of moe_topk_sum.

Round 1 (2026-11-08) replaced the original 2D ``[BM, BD]`` tile kernel with
the vllm moe_sum "mt" form: the (token, col-chunk) tile space flattened to a
1D affine grid-stride loop with ``tl.static_range`` TOPK unrolling and the
launch capped at 48 programs (one per SPC). That form reached ~1.7 TB/s.

Round 2 (2026-11-12) searched the remaining dimensions and found the win in
indexing and predication instead of shape: the flat form recomputed
``flat // NC``/``flat % NC`` and a (mostly all-true) predicate on every
chunk of every item. The kernel now maps the chunk index to ``program_id(1)``
so it is uniform per CTA, drops the div/mod entirely, and takes a
compile-time-specialized maskless arm whenever the chunk is full
(``(chunk + 1) * BD <= HIDDEN`` — always true for the leading chunk, and for
every chunk when HIDDEN divides BD, e.g. hidden 1024/2048/4096/8192).
The token loop keeps the grid-stride persistent form, so the grid still
launches at most 48 CTAs total (``p0 * nc``) and compiles once per
(topk, hidden) pair.

Measured on the official bench core shapes (kernel mode, do_bench
warmup=1000/rep=100 median), round-1 flat form -> this kernel:
(512,4) 23.4->22.2us, (512,8) 38.3->37.1us, (64,8) 9.0->8.9us,
(4096,8) 269->263us; decode-sized shapes (m<=8) unchanged (~4.1us launch
floor). Rejected with numbers this round: EXACT-only-width (BD=1024, NC=7)
+37..113% (narrow loads lose more than predicates save), adjacent-item
pairing U=2 at BD 4096/2048/1024 +27..45% (register pressure), grid caps
above 48 +3..27% (dispatch tax ~52ns/CTA, no bandwidth gain — the SPC
memory port saturates near 46 GB/s/CTA), num_stages 2/3 (no effect),
balanced/slab/flat-chunk-branch variants (no gain).

The host also guards the row-offset arithmetic: ``token * (topk * hidden)``
overflows int32 once the input crosses 2**31 elements, so the kernel takes
an int64-offset arm then (same fix as moe_sum_reduce's walk kernel; the
i32 and i64 arms are bitwise identical below the threshold).
"""

import triton
import triton.language as tl

# Total CTAs across both grid dims stay at one per SPC; extra programs just
# queue behind the ~52 ns/CTA dispatch floor (compiler_issues/vllm_moe_sum.md).
_GRID_CAP = 48
# Column-chunk width. 4096 elements (8 KB bf16 rows) measured fastest for
# topk in {4, 8}; smaller hidden sizes fall back to their pow2 width.
_BLOCK_H_MAX = 4096


@triton.jit
def _sum(
    X,
    Out,
    M,
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    BD: tl.constexpr,
    I64: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BD + tl.arange(0, BD)
    if (chunk + 1) * BD <= HIDDEN:
        # Full chunk: no predicate at all on loads or the store.
        for token in range(pid, M, tl.num_programs(0)):
            if I64:
                # token * (TOPK * HIDDEN) overflows int32 once the input
                # exceeds 2**31 elements; widen the row offsets only then.
                row = token.to(tl.int64) * (TOPK * HIDDEN)
                orow = token.to(tl.int64) * HIDDEN
            else:
                row = token * (TOPK * HIDDEN)
                orow = token * HIDDEN
            base = X + row + cols
            acc = tl.zeros((BD,), tl.float32)
            for expert in tl.static_range(0, TOPK):
                acc += tl.load(base + expert * HIDDEN).to(tl.float32)
            tl.store(Out + orow + cols, acc.to(Out.dtype.element_ty))
    else:
        # Tail chunk only: predicated on its own uniform lane mask.
        mask = cols < HIDDEN
        for token in range(pid, M, tl.num_programs(0)):
            if I64:
                row = token.to(tl.int64) * (TOPK * HIDDEN)
                orow = token.to(tl.int64) * HIDDEN
            else:
                row = token * (TOPK * HIDDEN)
                orow = token * HIDDEN
            base = X + row + cols
            acc = tl.zeros((BD,), tl.float32)
            for expert in tl.static_range(0, TOPK):
                acc += tl.load(
                    base + expert * HIDDEN, mask=mask, other=0.0
                ).to(tl.float32)
            tl.store(
                Out + orow + cols,
                acc.to(Out.dtype.element_ty),
                mask=mask,
            )


def moe_topk_sum(x, out):
    tokens, topk, hidden = x.shape
    if tokens == 0 or hidden == 0:
        return out
    bd = min(triton.next_power_of_2(hidden), _BLOCK_H_MAX)
    nc = (hidden + bd - 1) // bd
    p0 = max(1, min(tokens, _GRID_CAP // nc))
    # The token row stride (topk * hidden) leaves int32 once the input
    # crosses 2**31 elements; switch that arm to int64 offsets then.
    i64 = x.numel() > 2**31
    _sum[(p0, nc)](x, out, tokens, topk, hidden, bd, i64, num_warps=4)
    return out


__all__ = ["moe_topk_sum"]
