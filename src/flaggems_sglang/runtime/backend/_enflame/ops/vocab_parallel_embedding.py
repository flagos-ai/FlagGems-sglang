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

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _vocab_parallel_embedding_kernel(
    tok_ptr,
    weight_ptr,
    out_ptr,
    m,
    ORG_START: tl.constexpr,
    ORG_END: tl.constexpr,
    ADDED_START: tl.constexpr,
    ADDED_END: tl.constexpr,
    ADDED_OFFSET: tl.constexpr,
    HIDDEN: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_H: tl.constexpr,
    STREAM_OUT: tl.constexpr,
    IDX32: tl.constexpr,
    TOK_STRIDE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    if EVEN_M:
        tok = tl.load(tok_ptr + rows * TOK_STRIDE)
    else:
        tok = tl.load(tok_ptr + rows * TOK_STRIDE, mask=rows < m, other=0)

    in_org = (tok >= ORG_START) & (tok < ORG_END)
    in_added = (tok >= ADDED_START) & (tok < ADDED_END)
    valid = in_org | in_added

    local_id = tl.where(in_org, tok - ORG_START, tok - ADDED_OFFSET)
    local_id = tl.where(valid, local_id, 0)

    if IDX32:
        w_off = local_id[:, None] * WEIGHT_STRIDE + cols[None, :]
        out_off = rows[:, None] * HIDDEN + cols[None, :]
    else:
        w_off = local_id[:, None].to(tl.int64) * WEIGHT_STRIDE + cols[None, :]
        out_off = rows.to(tl.int64)[:, None] * HIDDEN + cols[None, :]

    if EVEN_M and EVEN_H:
        w = tl.load(weight_ptr + w_off)
        w = tl.where(valid[:, None], w, 0.0)
        if STREAM_OUT:
            tl.store(out_ptr + out_off, w, eviction_policy="evict_first")
        else:
            tl.store(out_ptr + out_off, w)
    else:
        mask = (rows < m)[:, None] & (cols < HIDDEN)[None, :]
        w = tl.load(weight_ptr + w_off, mask=mask, other=0.0)
        w = tl.where(valid[:, None], w, 0.0)
        if STREAM_OUT:
            tl.store(
                out_ptr + out_off, w, mask=mask, eviction_policy="evict_first"
            )
        else:
            tl.store(out_ptr + out_off, w, mask=mask)


@triton.jit
def _vocab_parallel_embedding_rows_kernel(
    tok_ptr,
    weight_ptr,
    out_ptr,
    m,
    n_items,
    ORG_START: tl.constexpr,
    ORG_END: tl.constexpr,
    ADDED_START: tl.constexpr,
    ADDED_END: tl.constexpr,
    ADDED_OFFSET: tl.constexpr,
    HIDDEN: tl.constexpr,
    WEIGHT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN_H: tl.constexpr,
    UNROLL: tl.constexpr,
    IDX32: tl.constexpr,
    TOK_STRIDE: tl.constexpr,
):
    # Persistent row-gather kernel for the enflame backend (row-gather copy
    # launch law, see knowledge/perf_playbook.md "gather_spec_extras 行 gather
    # 拷贝" entries): the grid is capped at 48 programs and each program walks
    # row items with a grid-stride loop, so neither small nor large token
    # counts pay a per-program dispatch floor, and every row moves through a
    # single full-width 1-D op (BLOCK = next_pow2(hidden), unmasked when
    # hidden is a power of two) instead of narrow per-column tiles.  For
    # UNROLL > 1 the per-row scalar chain (token load -> window check ->
    # local id) batches over UNROLL consecutive rows: scalar loads first,
    # then the UNROLL row gathers, then the UNROLL stores; tail rows are
    # index-clamped for loads and masked for stores.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    cols = tl.arange(0, BLOCK)
    if UNROLL == 1:
        for item in range(pid, n_items, nprog):
            r0 = item
            t0 = tl.load(tok_ptr + r0 * TOK_STRIDE)
            io0 = (t0 >= ORG_START) & (t0 < ORG_END)
            ia0 = (t0 >= ADDED_START) & (t0 < ADDED_END)
            va0 = io0 | ia0
            l0 = tl.where(io0, t0 - ORG_START, t0 - ADDED_OFFSET)
            l0 = tl.where(va0, l0, 0)
            if IDX32:
                wb0 = weight_ptr + l0 * WEIGHT_STRIDE + cols
                ob0 = out_ptr + r0 * HIDDEN + cols
            else:
                wb0 = weight_ptr + l0.to(tl.int64) * WEIGHT_STRIDE + cols
                ob0 = out_ptr + r0.to(tl.int64) * HIDDEN + cols
            if EVEN_H:
                w0 = tl.load(wb0)
                tl.store(ob0, tl.where(va0, w0, 0.0))
            else:
                cm0 = cols < HIDDEN
                w0 = tl.load(wb0, mask=cm0, other=0.0)
                tl.store(ob0, tl.where(va0, w0, 0.0), mask=cm0)
    elif UNROLL == 4:
        m1 = m - 1
        for item in range(pid, n_items, nprog):
            r0 = item * 4
            r1 = r0 + 1
            r2 = r0 + 2
            r3 = r0 + 3
            rc0 = tl.minimum(r0, m1)
            rc1 = tl.minimum(r1, m1)
            rc2 = tl.minimum(r2, m1)
            rc3 = tl.minimum(r3, m1)
            t0 = tl.load(tok_ptr + rc0 * TOK_STRIDE)
            t1 = tl.load(tok_ptr + rc1 * TOK_STRIDE)
            t2 = tl.load(tok_ptr + rc2 * TOK_STRIDE)
            t3 = tl.load(tok_ptr + rc3 * TOK_STRIDE)
            io0 = (t0 >= ORG_START) & (t0 < ORG_END)
            io1 = (t1 >= ORG_START) & (t1 < ORG_END)
            io2 = (t2 >= ORG_START) & (t2 < ORG_END)
            io3 = (t3 >= ORG_START) & (t3 < ORG_END)
            ia0 = (t0 >= ADDED_START) & (t0 < ADDED_END)
            ia1 = (t1 >= ADDED_START) & (t1 < ADDED_END)
            ia2 = (t2 >= ADDED_START) & (t2 < ADDED_END)
            ia3 = (t3 >= ADDED_START) & (t3 < ADDED_END)
            va0 = io0 | ia0
            va1 = io1 | ia1
            va2 = io2 | ia2
            va3 = io3 | ia3
            l0 = tl.where(io0, t0 - ORG_START, t0 - ADDED_OFFSET)
            l1 = tl.where(io1, t1 - ORG_START, t1 - ADDED_OFFSET)
            l2 = tl.where(io2, t2 - ORG_START, t2 - ADDED_OFFSET)
            l3 = tl.where(io3, t3 - ORG_START, t3 - ADDED_OFFSET)
            l0 = tl.where(va0, l0, 0)
            l1 = tl.where(va1, l1, 0)
            l2 = tl.where(va2, l2, 0)
            l3 = tl.where(va3, l3, 0)
            if IDX32:
                wb0 = weight_ptr + l0 * WEIGHT_STRIDE + cols
                wb1 = weight_ptr + l1 * WEIGHT_STRIDE + cols
                wb2 = weight_ptr + l2 * WEIGHT_STRIDE + cols
                wb3 = weight_ptr + l3 * WEIGHT_STRIDE + cols
                ob0 = out_ptr + rc0 * HIDDEN + cols
                ob1 = out_ptr + rc1 * HIDDEN + cols
                ob2 = out_ptr + rc2 * HIDDEN + cols
                ob3 = out_ptr + rc3 * HIDDEN + cols
            else:
                wb0 = weight_ptr + l0.to(tl.int64) * WEIGHT_STRIDE + cols
                wb1 = weight_ptr + l1.to(tl.int64) * WEIGHT_STRIDE + cols
                wb2 = weight_ptr + l2.to(tl.int64) * WEIGHT_STRIDE + cols
                wb3 = weight_ptr + l3.to(tl.int64) * WEIGHT_STRIDE + cols
                ob0 = out_ptr + rc0.to(tl.int64) * HIDDEN + cols
                ob1 = out_ptr + rc1.to(tl.int64) * HIDDEN + cols
                ob2 = out_ptr + rc2.to(tl.int64) * HIDDEN + cols
                ob3 = out_ptr + rc3.to(tl.int64) * HIDDEN + cols
            sm0 = r0 < m
            sm1 = r1 < m
            sm2 = r2 < m
            sm3 = r3 < m
            if EVEN_H:
                w0 = tl.load(wb0)
                w1 = tl.load(wb1)
                w2 = tl.load(wb2)
                w3 = tl.load(wb3)
                tl.store(ob0, tl.where(va0, w0, 0.0), mask=sm0)
                tl.store(ob1, tl.where(va1, w1, 0.0), mask=sm1)
                tl.store(ob2, tl.where(va2, w2, 0.0), mask=sm2)
                tl.store(ob3, tl.where(va3, w3, 0.0), mask=sm3)
            else:
                cm0 = cols < HIDDEN
                w0 = tl.load(wb0, mask=cm0, other=0.0)
                w1 = tl.load(wb1, mask=cm0, other=0.0)
                w2 = tl.load(wb2, mask=cm0, other=0.0)
                w3 = tl.load(wb3, mask=cm0, other=0.0)
                tl.store(ob0, tl.where(va0, w0, 0.0), mask=cm0 & sm0)
                tl.store(ob1, tl.where(va1, w1, 0.0), mask=cm0 & sm1)
                tl.store(ob2, tl.where(va2, w2, 0.0), mask=cm0 & sm2)
                tl.store(ob3, tl.where(va3, w3, 0.0), mask=cm0 & sm3)
    else:
        m1 = m - 1
        for item in range(pid, n_items, nprog):
            r0 = item * 8
            r1 = r0 + 1
            r2 = r0 + 2
            r3 = r0 + 3
            r4 = r0 + 4
            r5 = r0 + 5
            r6 = r0 + 6
            r7 = r0 + 7
            rc0 = tl.minimum(r0, m1)
            rc1 = tl.minimum(r1, m1)
            rc2 = tl.minimum(r2, m1)
            rc3 = tl.minimum(r3, m1)
            rc4 = tl.minimum(r4, m1)
            rc5 = tl.minimum(r5, m1)
            rc6 = tl.minimum(r6, m1)
            rc7 = tl.minimum(r7, m1)
            t0 = tl.load(tok_ptr + rc0 * TOK_STRIDE)
            t1 = tl.load(tok_ptr + rc1 * TOK_STRIDE)
            t2 = tl.load(tok_ptr + rc2 * TOK_STRIDE)
            t3 = tl.load(tok_ptr + rc3 * TOK_STRIDE)
            t4 = tl.load(tok_ptr + rc4 * TOK_STRIDE)
            t5 = tl.load(tok_ptr + rc5 * TOK_STRIDE)
            t6 = tl.load(tok_ptr + rc6 * TOK_STRIDE)
            t7 = tl.load(tok_ptr + rc7 * TOK_STRIDE)
            io0 = (t0 >= ORG_START) & (t0 < ORG_END)
            io1 = (t1 >= ORG_START) & (t1 < ORG_END)
            io2 = (t2 >= ORG_START) & (t2 < ORG_END)
            io3 = (t3 >= ORG_START) & (t3 < ORG_END)
            io4 = (t4 >= ORG_START) & (t4 < ORG_END)
            io5 = (t5 >= ORG_START) & (t5 < ORG_END)
            io6 = (t6 >= ORG_START) & (t6 < ORG_END)
            io7 = (t7 >= ORG_START) & (t7 < ORG_END)
            ia0 = (t0 >= ADDED_START) & (t0 < ADDED_END)
            ia1 = (t1 >= ADDED_START) & (t1 < ADDED_END)
            ia2 = (t2 >= ADDED_START) & (t2 < ADDED_END)
            ia3 = (t3 >= ADDED_START) & (t3 < ADDED_END)
            ia4 = (t4 >= ADDED_START) & (t4 < ADDED_END)
            ia5 = (t5 >= ADDED_START) & (t5 < ADDED_END)
            ia6 = (t6 >= ADDED_START) & (t6 < ADDED_END)
            ia7 = (t7 >= ADDED_START) & (t7 < ADDED_END)
            va0 = io0 | ia0
            va1 = io1 | ia1
            va2 = io2 | ia2
            va3 = io3 | ia3
            va4 = io4 | ia4
            va5 = io5 | ia5
            va6 = io6 | ia6
            va7 = io7 | ia7
            l0 = tl.where(io0, t0 - ORG_START, t0 - ADDED_OFFSET)
            l1 = tl.where(io1, t1 - ORG_START, t1 - ADDED_OFFSET)
            l2 = tl.where(io2, t2 - ORG_START, t2 - ADDED_OFFSET)
            l3 = tl.where(io3, t3 - ORG_START, t3 - ADDED_OFFSET)
            l4 = tl.where(io4, t4 - ORG_START, t4 - ADDED_OFFSET)
            l5 = tl.where(io5, t5 - ORG_START, t5 - ADDED_OFFSET)
            l6 = tl.where(io6, t6 - ORG_START, t6 - ADDED_OFFSET)
            l7 = tl.where(io7, t7 - ORG_START, t7 - ADDED_OFFSET)
            l0 = tl.where(va0, l0, 0)
            l1 = tl.where(va1, l1, 0)
            l2 = tl.where(va2, l2, 0)
            l3 = tl.where(va3, l3, 0)
            l4 = tl.where(va4, l4, 0)
            l5 = tl.where(va5, l5, 0)
            l6 = tl.where(va6, l6, 0)
            l7 = tl.where(va7, l7, 0)
            if IDX32:
                wb0 = weight_ptr + l0 * WEIGHT_STRIDE + cols
                wb1 = weight_ptr + l1 * WEIGHT_STRIDE + cols
                wb2 = weight_ptr + l2 * WEIGHT_STRIDE + cols
                wb3 = weight_ptr + l3 * WEIGHT_STRIDE + cols
                wb4 = weight_ptr + l4 * WEIGHT_STRIDE + cols
                wb5 = weight_ptr + l5 * WEIGHT_STRIDE + cols
                wb6 = weight_ptr + l6 * WEIGHT_STRIDE + cols
                wb7 = weight_ptr + l7 * WEIGHT_STRIDE + cols
                ob0 = out_ptr + rc0 * HIDDEN + cols
                ob1 = out_ptr + rc1 * HIDDEN + cols
                ob2 = out_ptr + rc2 * HIDDEN + cols
                ob3 = out_ptr + rc3 * HIDDEN + cols
                ob4 = out_ptr + rc4 * HIDDEN + cols
                ob5 = out_ptr + rc5 * HIDDEN + cols
                ob6 = out_ptr + rc6 * HIDDEN + cols
                ob7 = out_ptr + rc7 * HIDDEN + cols
            else:
                wb0 = weight_ptr + l0.to(tl.int64) * WEIGHT_STRIDE + cols
                wb1 = weight_ptr + l1.to(tl.int64) * WEIGHT_STRIDE + cols
                wb2 = weight_ptr + l2.to(tl.int64) * WEIGHT_STRIDE + cols
                wb3 = weight_ptr + l3.to(tl.int64) * WEIGHT_STRIDE + cols
                wb4 = weight_ptr + l4.to(tl.int64) * WEIGHT_STRIDE + cols
                wb5 = weight_ptr + l5.to(tl.int64) * WEIGHT_STRIDE + cols
                wb6 = weight_ptr + l6.to(tl.int64) * WEIGHT_STRIDE + cols
                wb7 = weight_ptr + l7.to(tl.int64) * WEIGHT_STRIDE + cols
                ob0 = out_ptr + rc0.to(tl.int64) * HIDDEN + cols
                ob1 = out_ptr + rc1.to(tl.int64) * HIDDEN + cols
                ob2 = out_ptr + rc2.to(tl.int64) * HIDDEN + cols
                ob3 = out_ptr + rc3.to(tl.int64) * HIDDEN + cols
                ob4 = out_ptr + rc4.to(tl.int64) * HIDDEN + cols
                ob5 = out_ptr + rc5.to(tl.int64) * HIDDEN + cols
                ob6 = out_ptr + rc6.to(tl.int64) * HIDDEN + cols
                ob7 = out_ptr + rc7.to(tl.int64) * HIDDEN + cols
            sm0 = r0 < m
            sm1 = r1 < m
            sm2 = r2 < m
            sm3 = r3 < m
            sm4 = r4 < m
            sm5 = r5 < m
            sm6 = r6 < m
            sm7 = r7 < m
            if EVEN_H:
                w0 = tl.load(wb0)
                w1 = tl.load(wb1)
                w2 = tl.load(wb2)
                w3 = tl.load(wb3)
                w4 = tl.load(wb4)
                w5 = tl.load(wb5)
                w6 = tl.load(wb6)
                w7 = tl.load(wb7)
                tl.store(ob0, tl.where(va0, w0, 0.0), mask=sm0)
                tl.store(ob1, tl.where(va1, w1, 0.0), mask=sm1)
                tl.store(ob2, tl.where(va2, w2, 0.0), mask=sm2)
                tl.store(ob3, tl.where(va3, w3, 0.0), mask=sm3)
                tl.store(ob4, tl.where(va4, w4, 0.0), mask=sm4)
                tl.store(ob5, tl.where(va5, w5, 0.0), mask=sm5)
                tl.store(ob6, tl.where(va6, w6, 0.0), mask=sm6)
                tl.store(ob7, tl.where(va7, w7, 0.0), mask=sm7)
            else:
                cm0 = cols < HIDDEN
                w0 = tl.load(wb0, mask=cm0, other=0.0)
                w1 = tl.load(wb1, mask=cm0, other=0.0)
                w2 = tl.load(wb2, mask=cm0, other=0.0)
                w3 = tl.load(wb3, mask=cm0, other=0.0)
                w4 = tl.load(wb4, mask=cm0, other=0.0)
                w5 = tl.load(wb5, mask=cm0, other=0.0)
                w6 = tl.load(wb6, mask=cm0, other=0.0)
                w7 = tl.load(wb7, mask=cm0, other=0.0)
                tl.store(ob0, tl.where(va0, w0, 0.0), mask=cm0 & sm0)
                tl.store(ob1, tl.where(va1, w1, 0.0), mask=cm0 & sm1)
                tl.store(ob2, tl.where(va2, w2, 0.0), mask=cm0 & sm2)
                tl.store(ob3, tl.where(va3, w3, 0.0), mask=cm0 & sm3)
                tl.store(ob4, tl.where(va4, w4, 0.0), mask=cm0 & sm4)
                tl.store(ob5, tl.where(va5, w5, 0.0), mask=cm0 & sm5)
                tl.store(ob6, tl.where(va6, w6, 0.0), mask=cm0 & sm6)
                tl.store(ob7, tl.where(va7, w7, 0.0), mask=cm0 & sm7)


_I64_TRUE8 = None


def _int64_tokens_true8(device):
    """Detect the on-device int64 storage layout (cached per process).

    torch_gcu packs int64 values 4-byte-in-the-low-half when int64 support
    is off, and uses a true 8-byte layout once
    TORCH_GCU_ENABLE_INT64_AND_UINT64 is enabled.  The raw-int32
    reinterpretation below differs between the two layouts, so probe the
    device once and cache the verdict.
    """
    global _I64_TRUE8
    if _I64_TRUE8 is None:
        try:
            probe = torch.arange(2, dtype=torch.int64, device=device)
            raw = probe.view(torch.int32).tolist()
            # True 8-byte little-endian reads [0, 0, 1, 0]; the packed
            # layout reads [0, 1, ...].
            _I64_TRUE8 = bool(len(raw) >= 4 and raw[1] == 0 and raw[2] == 1)
        except Exception:
            _I64_TRUE8 = os.environ.get(
                "TORCH_GCU_ENABLE_INT64_AND_UINT64", ""
            ).lower() in ("1", "true")
    return _I64_TRUE8


def vocab_parallel_embedding(
    input_,
    weight,
    org_vocab_start_index,
    org_vocab_end_index,
    num_org_vocab_padding,
    added_vocab_start_index,
    added_vocab_end_index,
):
    org_start = (
        0 if org_vocab_start_index is None else int(org_vocab_start_index)
    )
    org_end = 0 if org_vocab_end_index is None else int(org_vocab_end_index)
    padding = (
        0 if num_org_vocab_padding is None else int(num_org_vocab_padding)
    )
    added_start = (
        0 if added_vocab_start_index is None else int(added_vocab_start_index)
    )
    added_end = (
        0 if added_vocab_end_index is None else int(added_vocab_end_index)
    )

    tok_flat = input_.reshape(-1)
    m = tok_flat.numel()
    tok_stride = 1
    if tok_flat.dtype == torch.int64:
        # torch_gcu replaces Long with Int in a 4-byte-packed layout (values
        # in the low half, high half garbage) while int64 support is off, and
        # switches to a true 8-byte layout under
        # TORCH_GCU_ENABLE_INT64_AND_UINT64.  Reinterpret the value halves as
        # int32 in either case: stride 1 recovers the packed values, stride 2
        # picks the low half of every true-8-byte element.
        raw32 = tok_flat.view(torch.int32)
        if _int64_tokens_true8(tok_flat.device):
            tok_stride = 2
            tok_flat = raw32
        else:
            tok_flat = raw32[:m]
    hidden = weight.shape[1]

    out = torch.empty(
        (*input_.shape, hidden), dtype=weight.dtype, device=weight.device
    )
    if m == 0:
        return out

    added_offset = added_start - (org_end - org_start) - padding

    if hidden <= 8192:
        block = triton.next_power_of_2(hidden)
        even_h = block == hidden
        idx32 = weight.shape[0] * weight.stride(0) < 2**31 and m * hidden < 2**31
        narrow = even_h and block <= 2048
        if m <= 48:
            unroll = 1
            num_warps = 4
            n_items = m
        elif narrow and m >= 512:
            unroll = 8
            num_warps = 2
            n_items = triton.cdiv(m, 8)
        else:
            unroll = 4
            num_warps = 2 if narrow else 4
            n_items = triton.cdiv(m, 4)
        grid = (min(n_items, 48),)
        _vocab_parallel_embedding_rows_kernel[grid](
            tok_flat,
            weight,
            out,
            m,
            n_items,
            ORG_START=org_start,
            ORG_END=org_end,
            ADDED_START=added_start,
            ADDED_END=added_end,
            ADDED_OFFSET=added_offset,
            HIDDEN=hidden,
            WEIGHT_STRIDE=weight.stride(0),
            BLOCK=block,
            EVEN_H=even_h,
            UNROLL=unroll,
            IDX32=idx32,
            TOK_STRIDE=tok_stride,
            num_warps=num_warps,
        )
        return out

    if m <= 8:
        block_m, block_h, num_warps = 1, 1024, 4
    elif m < 512:
        if hidden >= 4096:
            block_m, block_h, num_warps = 1, 1024, 1
        else:
            block_m, block_h, num_warps = 8, 1024, 2
    elif m < 4096:
        if hidden >= 4096:
            block_m, block_h, num_warps = 16, 1024, 8
        else:
            block_m, block_h, num_warps = 8, 1024, 2
    elif hidden % 2048 == 0:
        block_m, block_h, num_warps = 8, 2048, 1
    elif hidden % 1024 == 0:
        block_m, block_h, num_warps = 8, 1024, 16
    else:
        block_m, block_h, num_warps = 1, 4096, 8

    stream_out = m * hidden >= 8_000_000

    idx32 = weight.shape[0] * weight.stride(0) < 2**31 and m * hidden < 2**31

    even_m = m % block_m == 0
    even_h = hidden % block_h == 0
    num_h = hidden // block_h if even_h else triton.cdiv(hidden, block_h)
    grid = (triton.cdiv(m, block_m), num_h)

    _vocab_parallel_embedding_kernel[grid](
        tok_flat,
        weight,
        out,
        m,
        ORG_START=org_start,
        ORG_END=org_end,
        ADDED_START=added_start,
        ADDED_END=added_end,
        ADDED_OFFSET=added_offset,
        HIDDEN=hidden,
        WEIGHT_STRIDE=weight.stride(0),
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        EVEN_M=even_m,
        EVEN_H=even_h,
        STREAM_OUT=stream_out,
        IDX32=idx32,
        TOK_STRIDE=tok_stride,
        num_warps=num_warps,
    )
    return out


__all__ = ["vocab_parallel_embedding"]
