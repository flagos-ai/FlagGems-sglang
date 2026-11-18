# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl

# Pure-1D rewrite (2026-11-08). The previous kunlunxin-recipe grouped-GEMM
# host loop (index_select packing + per-adapter slice GEMMs, five host syncs
# and ~15 launches per call) can no longer compile on the 11-07 20:22
# toolchain: every rank-2 tile access lowers to memref_ext.slice_(pad_)start
# with mismatched operand segments (see
# compiler_issues/sglang_chunked_sgmv_expand.md). Only 1D loads/stores stay
# compilable, so both kernels below are pure 1D.
#
# Measured L600 primitives that shaped the vector kernel (probe logs in the
# job tmp/): strided per-k (BLOCK_N,) W loads are L2-line amplified (128B
# line per lane per load, 2.94 TB/s ceiling, no cross-k line reuse); scalar
# loads have zero MLP (~190 ns each, fully serialized); (R,) tl.sum
# reductions cost ~15 ns; bf16 pairs ride free in u32/u64 words
# (hi-half-mask / lo-shift-16 bitcasts). Therefore:
#   * W is fetched once per work item as RANK//4 strided u64 vectors
#     (4 bf16 per lane, line amplification cut 4x) and unpacked once into
#     RANK named fp32 vectors that live across the item's row loop;
#   * x rows are loaded as u32 word vectors and the per-k scalars are
#     extracted with (R2C,) one-hot where+sum (integer sums are bit-exact);
#   * each row does one contiguous (BLOCK_N,) base load, RANK wide FMAs and
#     one contiguous store.
# Work items are (slice, col_block, segment, row_chunk); a persistent 48-SP
# grid strides over them. Fallback (odd rank / >64 / misaligned strides):
# the row kernel (one program per row, runtime k loop).


@triton.jit(
    do_not_specialize=[
        "col_blocks_per_slice",
        "m_blocks_per_seg",
        "total_row_progs",
        "total_items",
        "chunk",
    ]
)
def _sgmv_expand_vec_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    permutation_ptr,
    slice_offsets_ptr,
    x_wstride0,
    w_wstride0,
    w_wstride1,
    out_stride0,
    col_blocks_per_slice,
    m_blocks_per_seg,
    total_row_progs,
    total_items,
    chunk,
    RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NSPC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    w64 = w_ptr.to(tl.pointer_type(tl.uint64))
    x64 = x_ptr.to(tl.pointer_type(tl.uint64))
    for item in range(pid, total_items, NSPC):
        col_prog = item // total_row_progs
        row_prog = item - col_prog * total_row_progs
        seg = row_prog // m_blocks_per_seg
        m_block = row_prog - seg * m_blocks_per_seg
        slice_id = col_prog // col_blocks_per_slice
        col_block = col_prog - slice_id * col_blocks_per_slice
        seg_start = tl.load(seg_indptr_ptr + seg)
        seg_end = tl.load(seg_indptr_ptr + seg + 1)
        p_lo = seg_start + m_block * chunk
        if p_lo < seg_end:
            o_start = tl.load(slice_offsets_ptr + slice_id)
            o_end = tl.load(slice_offsets_ptr + slice_id + 1)
            slice_len = o_end - o_start
            col_lo = col_block * BLOCK_N
            if col_lo < slice_len:
                w_idx = tl.load(weight_indices_ptr + seg)
                if tl.load(lora_ranks_ptr + w_idx) != 0:
                    scaling = tl.load(scalings_ptr + w_idx).to(tl.float32)
                    col_mask = (col_lo + offs_n) < slice_len
                    w_lane = (
                        w64
                        + w_idx * w_wstride0
                        + (o_start + col_lo) * w_wstride1
                        + offs_n * (RANK // 4)
                    )
                    if RANK > 3:
                        q0 = tl.load(w_lane + 0, mask=col_mask, other=0)
                        lo0 = (q0 & 0xFFFFFFFF).to(tl.uint32)
                        hi0 = (q0 >> 32).to(tl.uint32)
                        wB0 = (lo0 << 16).to(tl.float32, bitcast=True)
                        wA0 = (lo0 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD0 = (hi0 << 16).to(tl.float32, bitcast=True)
                        wC0 = (hi0 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 7:
                        q1 = tl.load(w_lane + 1, mask=col_mask, other=0)
                        lo1 = (q1 & 0xFFFFFFFF).to(tl.uint32)
                        hi1 = (q1 >> 32).to(tl.uint32)
                        wB1 = (lo1 << 16).to(tl.float32, bitcast=True)
                        wA1 = (lo1 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD1 = (hi1 << 16).to(tl.float32, bitcast=True)
                        wC1 = (hi1 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 11:
                        q2 = tl.load(w_lane + 2, mask=col_mask, other=0)
                        lo2 = (q2 & 0xFFFFFFFF).to(tl.uint32)
                        hi2 = (q2 >> 32).to(tl.uint32)
                        wB2 = (lo2 << 16).to(tl.float32, bitcast=True)
                        wA2 = (lo2 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD2 = (hi2 << 16).to(tl.float32, bitcast=True)
                        wC2 = (hi2 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 15:
                        q3 = tl.load(w_lane + 3, mask=col_mask, other=0)
                        lo3 = (q3 & 0xFFFFFFFF).to(tl.uint32)
                        hi3 = (q3 >> 32).to(tl.uint32)
                        wB3 = (lo3 << 16).to(tl.float32, bitcast=True)
                        wA3 = (lo3 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD3 = (hi3 << 16).to(tl.float32, bitcast=True)
                        wC3 = (hi3 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 19:
                        q4 = tl.load(w_lane + 4, mask=col_mask, other=0)
                        lo4 = (q4 & 0xFFFFFFFF).to(tl.uint32)
                        hi4 = (q4 >> 32).to(tl.uint32)
                        wB4 = (lo4 << 16).to(tl.float32, bitcast=True)
                        wA4 = (lo4 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD4 = (hi4 << 16).to(tl.float32, bitcast=True)
                        wC4 = (hi4 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 23:
                        q5 = tl.load(w_lane + 5, mask=col_mask, other=0)
                        lo5 = (q5 & 0xFFFFFFFF).to(tl.uint32)
                        hi5 = (q5 >> 32).to(tl.uint32)
                        wB5 = (lo5 << 16).to(tl.float32, bitcast=True)
                        wA5 = (lo5 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD5 = (hi5 << 16).to(tl.float32, bitcast=True)
                        wC5 = (hi5 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 27:
                        q6 = tl.load(w_lane + 6, mask=col_mask, other=0)
                        lo6 = (q6 & 0xFFFFFFFF).to(tl.uint32)
                        hi6 = (q6 >> 32).to(tl.uint32)
                        wB6 = (lo6 << 16).to(tl.float32, bitcast=True)
                        wA6 = (lo6 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD6 = (hi6 << 16).to(tl.float32, bitcast=True)
                        wC6 = (hi6 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 31:
                        q7 = tl.load(w_lane + 7, mask=col_mask, other=0)
                        lo7 = (q7 & 0xFFFFFFFF).to(tl.uint32)
                        hi7 = (q7 >> 32).to(tl.uint32)
                        wB7 = (lo7 << 16).to(tl.float32, bitcast=True)
                        wA7 = (lo7 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD7 = (hi7 << 16).to(tl.float32, bitcast=True)
                        wC7 = (hi7 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 35:
                        q8 = tl.load(w_lane + 8, mask=col_mask, other=0)
                        lo8 = (q8 & 0xFFFFFFFF).to(tl.uint32)
                        hi8 = (q8 >> 32).to(tl.uint32)
                        wB8 = (lo8 << 16).to(tl.float32, bitcast=True)
                        wA8 = (lo8 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD8 = (hi8 << 16).to(tl.float32, bitcast=True)
                        wC8 = (hi8 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 39:
                        q9 = tl.load(w_lane + 9, mask=col_mask, other=0)
                        lo9 = (q9 & 0xFFFFFFFF).to(tl.uint32)
                        hi9 = (q9 >> 32).to(tl.uint32)
                        wB9 = (lo9 << 16).to(tl.float32, bitcast=True)
                        wA9 = (lo9 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD9 = (hi9 << 16).to(tl.float32, bitcast=True)
                        wC9 = (hi9 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 43:
                        q10 = tl.load(w_lane + 10, mask=col_mask, other=0)
                        lo10 = (q10 & 0xFFFFFFFF).to(tl.uint32)
                        hi10 = (q10 >> 32).to(tl.uint32)
                        wB10 = (lo10 << 16).to(tl.float32, bitcast=True)
                        wA10 = (lo10 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD10 = (hi10 << 16).to(tl.float32, bitcast=True)
                        wC10 = (hi10 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 47:
                        q11 = tl.load(w_lane + 11, mask=col_mask, other=0)
                        lo11 = (q11 & 0xFFFFFFFF).to(tl.uint32)
                        hi11 = (q11 >> 32).to(tl.uint32)
                        wB11 = (lo11 << 16).to(tl.float32, bitcast=True)
                        wA11 = (lo11 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD11 = (hi11 << 16).to(tl.float32, bitcast=True)
                        wC11 = (hi11 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 51:
                        q12 = tl.load(w_lane + 12, mask=col_mask, other=0)
                        lo12 = (q12 & 0xFFFFFFFF).to(tl.uint32)
                        hi12 = (q12 >> 32).to(tl.uint32)
                        wB12 = (lo12 << 16).to(tl.float32, bitcast=True)
                        wA12 = (lo12 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD12 = (hi12 << 16).to(tl.float32, bitcast=True)
                        wC12 = (hi12 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 55:
                        q13 = tl.load(w_lane + 13, mask=col_mask, other=0)
                        lo13 = (q13 & 0xFFFFFFFF).to(tl.uint32)
                        hi13 = (q13 >> 32).to(tl.uint32)
                        wB13 = (lo13 << 16).to(tl.float32, bitcast=True)
                        wA13 = (lo13 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD13 = (hi13 << 16).to(tl.float32, bitcast=True)
                        wC13 = (hi13 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 59:
                        q14 = tl.load(w_lane + 14, mask=col_mask, other=0)
                        lo14 = (q14 & 0xFFFFFFFF).to(tl.uint32)
                        hi14 = (q14 >> 32).to(tl.uint32)
                        wB14 = (lo14 << 16).to(tl.float32, bitcast=True)
                        wA14 = (lo14 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD14 = (hi14 << 16).to(tl.float32, bitcast=True)
                        wC14 = (hi14 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    if RANK > 63:
                        q15 = tl.load(w_lane + 15, mask=col_mask, other=0)
                        lo15 = (q15 & 0xFFFFFFFF).to(tl.uint32)
                        hi15 = (q15 >> 32).to(tl.uint32)
                        wB15 = (lo15 << 16).to(tl.float32, bitcast=True)
                        wA15 = (lo15 & 0xFFFF0000).to(tl.float32, bitcast=True)
                        wD15 = (hi15 << 16).to(tl.float32, bitcast=True)
                        wC15 = (hi15 & 0xFFFF0000).to(tl.float32, bitcast=True)
                    p_hi = tl.minimum(p_lo + chunk, seg_end)
                    for rp in range(p_lo, p_hi):
                        row = tl.load(permutation_ptr + rp)
                        xb = x64 + row * x_wstride0 + slice_id * (RANK // 4)
                        out_row = (
                            out_ptr
                            + row * out_stride0
                            + (o_start + col_lo + offs_n)
                        )
                        acc = tl.load(out_row, mask=col_mask, other=0.0).to(
                            tl.float32
                        )
                        if RANK > 3:
                            xq0 = tl.load(xb + 0)
                            xlo0 = (xq0 & 0xFFFFFFFF).to(tl.uint32)
                            xhi0 = (xq0 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo0 << 16).to(tl.float32, bitcast=True)
                            ) * wB0
                            acc += (
                                scaling
                                * (xlo0 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA0
                            acc += (
                                scaling
                                * (xhi0 << 16).to(tl.float32, bitcast=True)
                            ) * wD0
                            acc += (
                                scaling
                                * (xhi0 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC0
                        if RANK > 7:
                            xq1 = tl.load(xb + 1)
                            xlo1 = (xq1 & 0xFFFFFFFF).to(tl.uint32)
                            xhi1 = (xq1 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo1 << 16).to(tl.float32, bitcast=True)
                            ) * wB1
                            acc += (
                                scaling
                                * (xlo1 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA1
                            acc += (
                                scaling
                                * (xhi1 << 16).to(tl.float32, bitcast=True)
                            ) * wD1
                            acc += (
                                scaling
                                * (xhi1 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC1
                        if RANK > 11:
                            xq2 = tl.load(xb + 2)
                            xlo2 = (xq2 & 0xFFFFFFFF).to(tl.uint32)
                            xhi2 = (xq2 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo2 << 16).to(tl.float32, bitcast=True)
                            ) * wB2
                            acc += (
                                scaling
                                * (xlo2 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA2
                            acc += (
                                scaling
                                * (xhi2 << 16).to(tl.float32, bitcast=True)
                            ) * wD2
                            acc += (
                                scaling
                                * (xhi2 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC2
                        if RANK > 15:
                            xq3 = tl.load(xb + 3)
                            xlo3 = (xq3 & 0xFFFFFFFF).to(tl.uint32)
                            xhi3 = (xq3 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo3 << 16).to(tl.float32, bitcast=True)
                            ) * wB3
                            acc += (
                                scaling
                                * (xlo3 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA3
                            acc += (
                                scaling
                                * (xhi3 << 16).to(tl.float32, bitcast=True)
                            ) * wD3
                            acc += (
                                scaling
                                * (xhi3 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC3
                        if RANK > 19:
                            xq4 = tl.load(xb + 4)
                            xlo4 = (xq4 & 0xFFFFFFFF).to(tl.uint32)
                            xhi4 = (xq4 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo4 << 16).to(tl.float32, bitcast=True)
                            ) * wB4
                            acc += (
                                scaling
                                * (xlo4 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA4
                            acc += (
                                scaling
                                * (xhi4 << 16).to(tl.float32, bitcast=True)
                            ) * wD4
                            acc += (
                                scaling
                                * (xhi4 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC4
                        if RANK > 23:
                            xq5 = tl.load(xb + 5)
                            xlo5 = (xq5 & 0xFFFFFFFF).to(tl.uint32)
                            xhi5 = (xq5 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo5 << 16).to(tl.float32, bitcast=True)
                            ) * wB5
                            acc += (
                                scaling
                                * (xlo5 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA5
                            acc += (
                                scaling
                                * (xhi5 << 16).to(tl.float32, bitcast=True)
                            ) * wD5
                            acc += (
                                scaling
                                * (xhi5 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC5
                        if RANK > 27:
                            xq6 = tl.load(xb + 6)
                            xlo6 = (xq6 & 0xFFFFFFFF).to(tl.uint32)
                            xhi6 = (xq6 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo6 << 16).to(tl.float32, bitcast=True)
                            ) * wB6
                            acc += (
                                scaling
                                * (xlo6 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA6
                            acc += (
                                scaling
                                * (xhi6 << 16).to(tl.float32, bitcast=True)
                            ) * wD6
                            acc += (
                                scaling
                                * (xhi6 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC6
                        if RANK > 31:
                            xq7 = tl.load(xb + 7)
                            xlo7 = (xq7 & 0xFFFFFFFF).to(tl.uint32)
                            xhi7 = (xq7 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo7 << 16).to(tl.float32, bitcast=True)
                            ) * wB7
                            acc += (
                                scaling
                                * (xlo7 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA7
                            acc += (
                                scaling
                                * (xhi7 << 16).to(tl.float32, bitcast=True)
                            ) * wD7
                            acc += (
                                scaling
                                * (xhi7 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC7
                        if RANK > 35:
                            xq8 = tl.load(xb + 8)
                            xlo8 = (xq8 & 0xFFFFFFFF).to(tl.uint32)
                            xhi8 = (xq8 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo8 << 16).to(tl.float32, bitcast=True)
                            ) * wB8
                            acc += (
                                scaling
                                * (xlo8 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA8
                            acc += (
                                scaling
                                * (xhi8 << 16).to(tl.float32, bitcast=True)
                            ) * wD8
                            acc += (
                                scaling
                                * (xhi8 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC8
                        if RANK > 39:
                            xq9 = tl.load(xb + 9)
                            xlo9 = (xq9 & 0xFFFFFFFF).to(tl.uint32)
                            xhi9 = (xq9 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo9 << 16).to(tl.float32, bitcast=True)
                            ) * wB9
                            acc += (
                                scaling
                                * (xlo9 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA9
                            acc += (
                                scaling
                                * (xhi9 << 16).to(tl.float32, bitcast=True)
                            ) * wD9
                            acc += (
                                scaling
                                * (xhi9 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC9
                        if RANK > 43:
                            xq10 = tl.load(xb + 10)
                            xlo10 = (xq10 & 0xFFFFFFFF).to(tl.uint32)
                            xhi10 = (xq10 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo10 << 16).to(tl.float32, bitcast=True)
                            ) * wB10
                            acc += (
                                scaling
                                * (xlo10 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA10
                            acc += (
                                scaling
                                * (xhi10 << 16).to(tl.float32, bitcast=True)
                            ) * wD10
                            acc += (
                                scaling
                                * (xhi10 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC10
                        if RANK > 47:
                            xq11 = tl.load(xb + 11)
                            xlo11 = (xq11 & 0xFFFFFFFF).to(tl.uint32)
                            xhi11 = (xq11 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo11 << 16).to(tl.float32, bitcast=True)
                            ) * wB11
                            acc += (
                                scaling
                                * (xlo11 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA11
                            acc += (
                                scaling
                                * (xhi11 << 16).to(tl.float32, bitcast=True)
                            ) * wD11
                            acc += (
                                scaling
                                * (xhi11 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC11
                        if RANK > 51:
                            xq12 = tl.load(xb + 12)
                            xlo12 = (xq12 & 0xFFFFFFFF).to(tl.uint32)
                            xhi12 = (xq12 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo12 << 16).to(tl.float32, bitcast=True)
                            ) * wB12
                            acc += (
                                scaling
                                * (xlo12 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA12
                            acc += (
                                scaling
                                * (xhi12 << 16).to(tl.float32, bitcast=True)
                            ) * wD12
                            acc += (
                                scaling
                                * (xhi12 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC12
                        if RANK > 55:
                            xq13 = tl.load(xb + 13)
                            xlo13 = (xq13 & 0xFFFFFFFF).to(tl.uint32)
                            xhi13 = (xq13 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo13 << 16).to(tl.float32, bitcast=True)
                            ) * wB13
                            acc += (
                                scaling
                                * (xlo13 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA13
                            acc += (
                                scaling
                                * (xhi13 << 16).to(tl.float32, bitcast=True)
                            ) * wD13
                            acc += (
                                scaling
                                * (xhi13 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC13
                        if RANK > 59:
                            xq14 = tl.load(xb + 14)
                            xlo14 = (xq14 & 0xFFFFFFFF).to(tl.uint32)
                            xhi14 = (xq14 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo14 << 16).to(tl.float32, bitcast=True)
                            ) * wB14
                            acc += (
                                scaling
                                * (xlo14 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA14
                            acc += (
                                scaling
                                * (xhi14 << 16).to(tl.float32, bitcast=True)
                            ) * wD14
                            acc += (
                                scaling
                                * (xhi14 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC14
                        if RANK > 63:
                            xq15 = tl.load(xb + 15)
                            xlo15 = (xq15 & 0xFFFFFFFF).to(tl.uint32)
                            xhi15 = (xq15 >> 32).to(tl.uint32)
                            acc += (
                                scaling
                                * (xlo15 << 16).to(tl.float32, bitcast=True)
                            ) * wB15
                            acc += (
                                scaling
                                * (xlo15 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wA15
                            acc += (
                                scaling
                                * (xhi15 << 16).to(tl.float32, bitcast=True)
                            ) * wD15
                            acc += (
                                scaling
                                * (xhi15 & 0xFFFF0000).to(
                                    tl.float32, bitcast=True
                                )
                            ) * wC15
                        tl.store(
                            out_row,
                            acc.to(out_ptr.dtype.element_ty),
                            mask=col_mask,
                        )


@triton.jit(
    do_not_specialize=[
        "col_blocks_per_slice",
        "m_blocks_per_seg",
        "total_row_progs",
    ]
)
def _sgmv_expand_row_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    seg_indptr_ptr,
    weight_indices_ptr,
    lora_ranks_ptr,
    scalings_ptr,
    permutation_ptr,
    slice_offsets_ptr,
    x_stride0,
    x_stride1,
    w_stride0,
    w_stride1,
    w_stride2,
    out_stride0,
    out_stride1,
    col_blocks_per_slice,
    m_blocks_per_seg,
    total_row_progs,
    RANK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    col_prog = pid // total_row_progs
    row_prog = pid - col_prog * total_row_progs
    seg = row_prog // m_blocks_per_seg
    p = row_prog - seg * m_blocks_per_seg
    slice_id = col_prog // col_blocks_per_slice
    col_block = col_prog - slice_id * col_blocks_per_slice

    seg_start = tl.load(seg_indptr_ptr + seg)
    seg_end = tl.load(seg_indptr_ptr + seg + 1)
    p += seg_start
    if p >= seg_end:
        return
    o_start = tl.load(slice_offsets_ptr + slice_id)
    o_end = tl.load(slice_offsets_ptr + slice_id + 1)
    slice_len = o_end - o_start
    col_lo = col_block * BLOCK_N
    if col_lo >= slice_len:
        return
    w_idx = tl.load(weight_indices_ptr + seg)
    if tl.load(lora_ranks_ptr + w_idx) == 0:
        return
    scaling = tl.load(scalings_ptr + w_idx).to(tl.float32)

    offs_c = col_lo + tl.arange(0, BLOCK_N)
    col_mask = offs_c < slice_len
    row = tl.load(permutation_ptr + p)
    w_cols = w_ptr + w_idx * w_stride0 + (o_start + offs_c) * w_stride1
    x_base = x_ptr + row * x_stride0 + slice_id * RANK * x_stride1
    out_row = out_ptr + row * out_stride0 + (o_start + offs_c) * out_stride1

    acc = tl.load(out_row, mask=col_mask, other=0.0).to(tl.float32)
    for k in range(RANK):
        x_k = tl.load(x_base + k * x_stride1).to(tl.float32)
        w_k = tl.load(w_cols + k * w_stride2, mask=col_mask, other=0.0).to(
            tl.float32
        )
        acc += (scaling * x_k) * w_k
    tl.store(out_row, acc.to(out_ptr.dtype.element_ty), mask=col_mask)


_GCU_NUM_GRID = 48  # Libra/L600 (GCU400): 48 SPs, 448KB DSM per program
_BLOCK_N = 1024
_CHUNK = 128


def chunked_sgmv_expand(
    x, weights, batch_info, slice_offsets, max_slice_size, base_output
):
    output = base_output.clone()
    n_slices = slice_offsets.numel() - 1
    rank = weights.shape[-1]
    if x.shape[1] != n_slices * rank:
        raise ValueError("x width must equal n_slices * rank")
    if (
        output.numel() == 0
        or n_slices <= 0
        or batch_info.bs == 0
        or x.shape[0] == 0
    ):
        return output
    if (
        output.numel() >= 2**31
        or weights.numel() >= 2**31
        or x.numel() >= 2**31
    ):
        raise ValueError(
            "chunked_sgmv_expand: tensor too large for int32 indexing"
        )

    seg_indptr = batch_info.seg_indptr
    # batch_info.max_len is a host int when present (sglang LoRABatchInfo);
    # otherwise pay a single host sync, mirroring the generic implementation.
    max_len = getattr(batch_info, "max_len", None)
    if max_len is None:
        max_len = int((seg_indptr[1:] - seg_indptr[:-1]).max().item())
    if max_len <= 0:
        return output

    permutation = batch_info.permutation
    if permutation is None:
        permutation = torch.arange(
            x.shape[0], device=x.device, dtype=torch.int32
        )

    if max_slice_size is None or int(max_slice_size) <= 0:
        # max_slice_size is advisory; fall back to the full output width.
        max_slice_size = output.shape[1]
    block_n = _BLOCK_N
    col_blocks_per_slice = triton.cdiv(int(max_slice_size), block_n)
    m_blocks_per_seg = triton.cdiv(int(max_len), _CHUNK)
    total_row_progs = batch_info.bs * m_blocks_per_seg
    total_items = n_slices * col_blocks_per_slice * total_row_progs
    if total_items <= 0:
        return output

    packed = (
        rank % 4 == 0
        and rank <= 64
        and x.stride(1) == 1
        and weights.stride(2) == 1
        and output.stride(1) == 1
        and x.stride(0) % 4 == 0
        and weights.stride(0) % 4 == 0
        and weights.stride(1) % 4 == 0
        and x.data_ptr() % 8 == 0
        and weights.data_ptr() % 8 == 0
    )
    if packed:
        grid = (min(2 * _GCU_NUM_GRID, total_items),)
        _sgmv_expand_vec_kernel[grid](
            x,
            weights,
            output,
            seg_indptr,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            batch_info.scalings,
            permutation,
            slice_offsets,
            x.stride(0) // 4,
            weights.stride(0) // 4,
            weights.stride(1) // 4,
            output.stride(0),
            col_blocks_per_slice,
            m_blocks_per_seg,
            total_row_progs,
            total_items,
            _CHUNK,
            RANK=rank,
            BLOCK_N=block_n,
            NSPC=grid[0],
            num_warps=4,
        )
    else:
        m_blocks_row = int(max_len)
        total_row = batch_info.bs * m_blocks_row
        grid = (n_slices * col_blocks_per_slice * total_row,)
        _sgmv_expand_row_kernel[grid](
            x,
            weights,
            output,
            seg_indptr,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            batch_info.scalings,
            permutation,
            slice_offsets,
            x.stride(0),
            x.stride(1),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            output.stride(0),
            output.stride(1),
            col_blocks_per_slice,
            m_blocks_row,
            total_row,
            RANK=rank,
            BLOCK_N=block_n,
            num_warps=4,
        )
    return output


__all__ = ["chunked_sgmv_expand"]
