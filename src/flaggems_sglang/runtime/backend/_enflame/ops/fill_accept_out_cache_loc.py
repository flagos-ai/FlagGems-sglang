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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""GCU adaptation of speculative/fill_accept_out_cache_loc.

torch_gcu stores ATen int64 tensors as compact int32 words (element j lives
at word j; with TORCH_GCU_ENABLE_INT64_AND_UINT64=1 it lives in the low word
of each 8-byte slot) while the tensor metadata still reports 8-byte elements.
The generic kernel therefore loads accept_index/out_cache_loc through int64
pointers at the 8-byte dtype stride, reading neighbor-joined garbage, and
stores through an int64 pointer past the compact values - the wild gather
aborts the device (baseline ABORT rc=-6 on the first size=8 case).

Here all three tensors are accessed through zero-copy int32 views with the
matching word stride, so every load/store stays in the int32 word domain.
Values are token positions and cache locations, which are assumed to fit in
the signed int32 safe domain (the same assumption other enflame int64
narrowing fixes rely on).
"""

import os

import torch
import triton
import triton.language as tl


def _int64_word_stride():
    # torch_gcu without TORCH_GCU_ENABLE_INT64_AND_UINT64 keeps ATen int64
    # values as compact int32 words (element j lives at word j); with the gate
    # on, values sit in the low word of each 8-byte slot (word 2*j). Either
    # way a Triton int64 access at the 8-byte dtype stride reads/writes the
    # wrong slots, so int64 tensors are always handed to the kernel as int32
    # words (same pattern as the enflame indexer_k_quant_and_cache fix).
    # torch_gcu only honors the "1"/"true" spellings (calibrated in
    # compiler_issues/triton_i64_narrow_load.md).
    enabled = os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64")
    return 2 if enabled in ("1", "true") else 1


def _input_int32_words(tensor):
    if tensor.dtype == torch.int32:
        return (tensor if tensor.is_contiguous() else tensor.contiguous()), 1
    if tensor.dtype == torch.int64:
        return tensor.contiguous().view(torch.int32), _int64_word_stride()
    raise NotImplementedError(
        "fill_accept_out_cache_loc supports int32/int64 tensors, got "
        f"{tensor.dtype}"
    )


def _output_int32_words(tensor):
    if tensor.dtype not in (torch.int32, torch.int64):
        raise NotImplementedError(
            "fill_accept_out_cache_loc supports int32/int64 tensors, got "
            f"{tensor.dtype}"
        )
    # Stay in-place for the (universal) contiguous case like the generic
    # kernel; a non-contiguous output cannot be word-addressed, so write a
    # contiguous copy and return it (the reference also returns a fresh
    # tensor, and the generic kernel would silently corrupt this case).
    target = tensor if tensor.is_contiguous() else tensor.contiguous()
    if tensor.dtype == torch.int32:
        return target, 1, target
    return target.view(torch.int32), _int64_word_stride(), target


@triton.jit
def _compact_chunk_kernel(
    accept_index_ptr,
    out_cache_loc_ptr,
    out_ptr,
    size,
    CHUNK: tl.constexpr,
    PREFIX_BLOCK: tl.constexpr,
    AI_STRIDE: tl.constexpr,
    LOC_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = pid * CHUNK + tl.arange(0, CHUNK)
    if EVEN:
        ai = tl.load(accept_index_ptr + offs * AI_STRIDE)
    else:
        ai = tl.load(
            accept_index_ptr + offs * AI_STRIDE, mask=offs < size, other=-1
        )
    keep = ai > -1
    ki = keep.to(tl.int32)
    local = tl.cumsum(ki, axis=0) - ki

    poffs = tl.arange(0, PREFIX_BLOCK)
    pmask = poffs < pid * CHUNK
    phi = tl.load(accept_index_ptr + poffs * AI_STRIDE, mask=pmask, other=-1)
    prefix = tl.sum((phi >= 0).to(tl.int32), axis=0)

    dst = local + prefix

    src = tl.where(keep, ai, 0)
    vals = tl.load(out_cache_loc_ptr + src * LOC_STRIDE)

    if EVEN:
        smask = keep
    else:
        smask = keep & (offs < size)
    if OUT_STRIDE == 1:
        tl.store(out_ptr + dst, vals, mask=smask)
    else:
        tl.store(out_ptr + 2 * dst, vals, mask=smask)
        tl.store(out_ptr + 2 * dst + 1, vals >> 31, mask=smask)


@triton.jit
def _fill_accept_out_cache_loc_kernel(
    accept_index_ptr,
    out_cache_loc_ptr,
    out_ptr,
    size,
    limit,
    AI_STRIDE: tl.constexpr,
    LOC_STRIDE: tl.constexpr,
    OUT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    I64_OUT: tl.constexpr,
):
    # NOTE: the prefetch load must stay unconditional and the dead-branch
    # keep-alive must stay AFTER the gather: any control flow placed between
    # the two loads (or guarding the prefetch) breaks their early-issue
    # overlap and costs 1-2.5us on this latency-bound kernel (measured on
    # three structurally "equivalent" rewrites).
    offs = tl.arange(0, BLOCK)
    # Speculative linear prefetch of the gather region: issued concurrently
    # with the accept_index load, so the dependent gather below usually hits
    # cache instead of paying a second serial DRAM round trip. Masked to the
    # in-bounds word count; the value is only consumed by the dead branch
    # below so the load cannot be eliminated while never affecting results.
    pre = tl.load(
        out_cache_loc_ptr + offs * LOC_STRIDE, mask=offs < limit, other=0
    )
    if EVEN:
        ai = tl.load(accept_index_ptr + offs * AI_STRIDE)
    else:
        ai = tl.load(
            accept_index_ptr + offs * AI_STRIDE, mask=offs < size, other=-1
        )
    keep = ai > -1
    ki = keep.to(tl.int32)

    dst = tl.cumsum(ki, axis=0) - ki

    src = tl.where(keep, ai, 0)
    vals = tl.load(out_cache_loc_ptr + src * LOC_STRIDE)

    if size == -12345:
        if I64_OUT:
            tl.store(out_ptr + offs, pre.to(tl.int64), mask=offs < size)
        else:
            tl.store(out_ptr + offs * OUT_STRIDE, pre, mask=offs < size)

    if EVEN:
        smask = keep
    else:
        smask = keep & (offs < size)
    if I64_OUT:
        # env-gated int64 layout: out_ptr is the original int64 tensor and a
        # single sign-extended int64 store writes both words of the 8-byte
        # slot at once (measured -0.05..-0.59us on the 8-element shape,
        # 8/8 paired rounds; neutral at 64/512).
        tl.store(out_ptr + dst, vals.to(tl.int64), mask=smask)
    elif OUT_STRIDE == 1:
        tl.store(out_ptr + dst, vals, mask=smask)
    else:
        tl.store(out_ptr + 2 * dst, vals, mask=smask)
        tl.store(out_ptr + 2 * dst + 1, vals >> 31, mask=smask)


@triton.jit
def _fill_accept_out_cache_loc_kernel_nopre(
    accept_index_ptr,
    out_cache_loc_ptr,
    out_ptr,
    size,
    AI_STRIDE: tl.constexpr,
    LOC_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    # Small env-gated int64 shapes (BLOCK < 64): the prefetch's extra issue
    # slots on a single warp cost more than the DRAM latency it hides once
    # the store side is a single int64 store (measured -0.135us at size=8,
    # 8/8 paired rounds; the prefetch stays essential from BLOCK 64 up).
    offs = tl.arange(0, BLOCK)
    if EVEN:
        ai = tl.load(accept_index_ptr + offs * AI_STRIDE)
    else:
        ai = tl.load(
            accept_index_ptr + offs * AI_STRIDE, mask=offs < size, other=-1
        )
    keep = ai > -1
    ki = keep.to(tl.int32)

    dst = tl.cumsum(ki, axis=0) - ki

    src = tl.where(keep, ai, 0)
    vals = tl.load(out_cache_loc_ptr + src * LOC_STRIDE)

    if EVEN:
        smask = keep
    else:
        smask = keep & (offs < size)
    tl.store(out_ptr + dst, vals.to(tl.int64), mask=smask)


def _pow2_at_least(x):
    return 1 << (x - 1).bit_length()


def fill_accept_out_cache_loc(
    accept_index, out_cache_loc, accept_out_cache_loc, size
):
    size = int(size)
    if size <= 0:
        return accept_out_cache_loc

    ai_words, ai_stride = _input_int32_words(accept_index)
    loc_words, loc_stride = _input_int32_words(out_cache_loc)
    out_words, out_stride, out_tensor = _output_int32_words(
        accept_out_cache_loc
    )

    if size <= 512:
        block = _pow2_at_least(size)
        if block <= 128:
            num_warps = 1
        else:
            num_warps = 8
        # In-bounds word count of out_cache_loc: int64 compact storage holds
        # one word per element (stride 1), the env-gated layout two (stride 2),
        # and an int32 tensor one. The prefetch must never read past it.
        loc_valid_words = out_cache_loc.numel() * loc_stride
        limit = size if size < loc_valid_words else loc_valid_words
        if out_stride == 2:
            # env-gated int64 output: hand the kernel the int64 tensor itself
            # so the store side can use one sign-extended int64 store per
            # element (writes both words of the 8-byte slot) instead of two
            # word stores. Below block 64 (single warp) the prefetch costs
            # more issue pressure than the latency it hides, so a dedicated
            # no-prefetch kernel handles those shapes.
            if block < 64:
                _fill_accept_out_cache_loc_kernel_nopre[(1,)](
                    ai_words,
                    loc_words,
                    out_tensor,
                    size,
                    AI_STRIDE=ai_stride,
                    LOC_STRIDE=loc_stride,
                    BLOCK=block,
                    EVEN=block == size,
                    num_warps=num_warps,
                )
                return out_tensor
            _fill_accept_out_cache_loc_kernel[(1,)](
                ai_words,
                loc_words,
                out_tensor,
                size,
                limit,
                AI_STRIDE=ai_stride,
                LOC_STRIDE=loc_stride,
                OUT_STRIDE=out_stride,
                BLOCK=block,
                EVEN=block == size,
                I64_OUT=True,
                num_warps=num_warps,
            )
            return out_tensor
        _fill_accept_out_cache_loc_kernel[(1,)](
            ai_words,
            loc_words,
            out_words,
            size,
            limit,
            AI_STRIDE=ai_stride,
            LOC_STRIDE=loc_stride,
            OUT_STRIDE=out_stride,
            BLOCK=block,
            EVEN=block == size,
            I64_OUT=False,
            num_warps=num_warps,
        )
        return out_tensor

    chunk = 1024
    grid = (size + chunk - 1) // chunk
    prefix_block = _pow2_at_least((grid - 1) * chunk)
    _compact_chunk_kernel[(grid,)](
        ai_words,
        loc_words,
        out_words,
        size,
        CHUNK=chunk,
        PREFIX_BLOCK=prefix_block,
        AI_STRIDE=ai_stride,
        LOC_STRIDE=loc_stride,
        OUT_STRIDE=out_stride,
        EVEN=size % chunk == 0,
        num_warps=16,
    )
    return out_tensor


__all__ = ["fill_accept_out_cache_loc"]
