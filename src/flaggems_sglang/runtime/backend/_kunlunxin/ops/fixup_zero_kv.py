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

import triton
import triton.language as tl

# Elements per store iteration.  Wide enough that one iteration covers a whole
# production-size request span (max_seq_len x heads x v_head_dim <= 8K elems),
# turning the walk into a single fully-coalesced store.
_BLOCK_OUT = 8192
_BLOCK_LSE = 1024
_NUM_WARPS = 8


@triton.jit
def _fixup_zero_kv_flat_kernel(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_seq_lens_ptr,
    out_row_inner,
    lse_row_inner,
    BLOCK_OUT: tl.constexpr,
    BLOCK_LSE: tl.constexpr,
):
    """Empty requests whose ``out``/``lse`` rows are packed: one flat span."""
    req = tl.program_id(0)
    if tl.load(kv_lens_ptr + req) == 0:
        beg = tl.load(cum_seq_lens_ptr + req)
        n = tl.load(cum_seq_lens_ptr + req + 1) - beg

        # out: ``n`` packed rows of ``out_row_inner`` elements each.
        out_base = beg.to(tl.int64) * out_row_inner
        out_total = n.to(tl.int64) * out_row_inner
        offs = tl.arange(0, BLOCK_OUT)
        zero = tl.zeros((BLOCK_OUT,), dtype=out_ptr.dtype.element_ty)
        for off in range(0, out_total, BLOCK_OUT):
            idx = off + offs
            tl.store(out_ptr + out_base + idx, zero, mask=idx < out_total)

        # lse: ``n`` packed rows of ``lse_row_inner`` elements each.
        lse_base = beg.to(tl.int64) * lse_row_inner
        lse_total = n.to(tl.int64) * lse_row_inner
        lse_offs = tl.arange(0, BLOCK_LSE)
        neg_inf = tl.full((BLOCK_LSE,), float("-inf"), dtype=tl.float32)
        for off in range(0, lse_total, BLOCK_LSE):
            idx = off + lse_offs
            tl.store(lse_ptr + lse_base + idx, neg_inf, mask=idx < lse_total)


@triton.jit
def _fixup_zero_kv_strided_kernel(
    out_ptr,
    lse_ptr,
    kv_lens_ptr,
    cum_seq_lens_ptr,
    out_row_stride,
    out_head_stride,
    num_heads,
    v_head_dim,
    lse_row_stride,
    lse_row_inner,
    BLOCK: tl.constexpr,
):
    """General fallback: flat (token, head, dim) enumeration."""
    req = tl.program_id(0)
    if tl.load(kv_lens_ptr + req) == 0:
        beg = tl.load(cum_seq_lens_ptr + req)
        n = tl.load(cum_seq_lens_ptr + req + 1) - beg
        offs = tl.arange(0, BLOCK)

        # out: ``n`` token rows, each holding ``num_heads`` head blocks of
        # ``v_head_dim`` contiguous elements at stride ``out_head_stride``.
        zero = tl.zeros((BLOCK,), dtype=out_ptr.dtype.element_ty)
        row_elems = num_heads * v_head_dim
        total = n.to(tl.int64) * row_elems
        base = beg.to(tl.int64) * out_row_stride
        for off in range(0, total, BLOCK):
            idx = off + offs
            row = idx // row_elems
            rem = idx % row_elems
            head = rem // v_head_dim
            d = rem % v_head_dim
            addr = base + row * out_row_stride + head * out_head_stride + d
            tl.store(out_ptr + addr, zero, mask=idx < total)

        lse_total = n.to(tl.int64) * lse_row_inner
        lse_base = beg.to(tl.int64) * lse_row_stride
        neg_inf = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
        for off in range(0, lse_total, BLOCK):
            idx = off + offs
            tl.store(lse_ptr + lse_base + idx, neg_inf, mask=idx < lse_total)


def _rows_packed(t):
    """True when consecutive rows of ``t`` are contiguous."""
    if t.dim() < 2:
        return True
    inner = t.shape[-1]
    for dim in range(t.dim() - 2, -1, -1):
        if t.stride(dim) != inner:
            return False
        inner *= t.shape[dim]
    return True


def fixup_zero_kv(out, lse, kv_lens, cum_seq_lens, max_seq_len):
    bs = kv_lens.shape[0]
    if bs == 0 or out.shape[0] == 0:
        return out, lse

    if _rows_packed(out) and lse.stride(-1) == 1 and _rows_packed(lse):
        _fixup_zero_kv_flat_kernel[(bs,)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            out.stride(0),
            lse.stride(0),
            BLOCK_OUT=_BLOCK_OUT,
            BLOCK_LSE=_BLOCK_LSE,
            num_warps=_NUM_WARPS,
        )
    else:
        # ``out`` viewed as (rows, heads, dim) with dim innermost contiguous.
        num_heads = out.shape[1] if out.dim() == 3 else 1
        v_head_dim = out.shape[-1]
        head_stride = out.stride(1) if out.dim() == 3 else out.stride(0)
        lse_row_inner = lse.shape[-1]
        _fixup_zero_kv_strided_kernel[(bs,)](
            out,
            lse,
            kv_lens,
            cum_seq_lens,
            out.stride(0),
            head_stride,
            num_heads,
            v_head_dim,
            lse.stride(0),
            lse_row_inner,
            BLOCK=_BLOCK_LSE,
            num_warps=_NUM_WARPS,
        )
    return out, lse


__all__ = ["fixup_zero_kv"]
