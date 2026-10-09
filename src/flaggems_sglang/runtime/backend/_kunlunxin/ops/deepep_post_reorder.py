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

"""DeepEP MoE post-reorder epilogue -- Kunlun XPU specialization.
"""

import triton
import triton.language as tl

# Whole-row chunk for the topk==8 fast path: one program loads its entire
# output row in a single (masked) load batch.  hidden=7168 < 8192 fits.
_CHUNK_FULL_ROW = 8192
# Generic fallback keeps the v4 config: 7 unmasked 1024-chunks for 7168.
_CHUNK_GENERIC = 1024
# Below this offset product the int32 address path is safe.  Rows fit int32
# when (num_tokens*topk) * hidden < 2^31; keep a comfortable margin.
_INT32_LIMIT = (1 << 31) - (1 << 20)


@triton.jit
def _post_reorder_fullrow_i32(
    down_ptr,  # [num_tokens * topk, hidden]  permuted expert outputs
    out_ptr,  # [num_tokens, hidden]         merged output
    src2dst_ptr,  # [num_tokens, topk] int32     permuted row index per slot
    weights_ptr,  # [num_tokens, topk] float32   routing weights
    hidden,  # hidden size (elements)
    SCALE,  # routed_scaling_factor (fp32 scalar)
    CHUNK: tl.constexpr,
    EVEN: tl.constexpr,  # CHUNK == hidden -> compile masks away
):
    pid_t = tl.program_id(0)
    meta = pid_t * 8
    out_base = pid_t * hidden

    # All 16 scalar metadata loads issued once, fully independent, before any
    # row load; per-slot scale folded a single time.
    # NB: scalar tl.where / boolean->float casts crash this backend's unroll
    # pass (uni_sram), so the dst<0 indicator is pure arithmetic --
    # arithmetic right-shift of a negative int32 gives -1, so
    # (dst >> 31) + 1 == 1 for dst >= 0 and 0 for dst < 0.
    dst0 = tl.load(src2dst_ptr + meta + 0)
    dst1 = tl.load(src2dst_ptr + meta + 1)
    dst2 = tl.load(src2dst_ptr + meta + 2)
    dst3 = tl.load(src2dst_ptr + meta + 3)
    dst4 = tl.load(src2dst_ptr + meta + 4)
    dst5 = tl.load(src2dst_ptr + meta + 5)
    dst6 = tl.load(src2dst_ptr + meta + 6)
    dst7 = tl.load(src2dst_ptr + meta + 7)
    w0 = (
        tl.load(weights_ptr + meta + 0).to(tl.float32)
        * SCALE
        * (((dst0 >> 31) + 1).to(tl.float32))
    )
    w1 = (
        tl.load(weights_ptr + meta + 1).to(tl.float32)
        * SCALE
        * (((dst1 >> 31) + 1).to(tl.float32))
    )
    w2 = (
        tl.load(weights_ptr + meta + 2).to(tl.float32)
        * SCALE
        * (((dst2 >> 31) + 1).to(tl.float32))
    )
    w3 = (
        tl.load(weights_ptr + meta + 3).to(tl.float32)
        * SCALE
        * (((dst3 >> 31) + 1).to(tl.float32))
    )
    w4 = (
        tl.load(weights_ptr + meta + 4).to(tl.float32)
        * SCALE
        * (((dst4 >> 31) + 1).to(tl.float32))
    )
    w5 = (
        tl.load(weights_ptr + meta + 5).to(tl.float32)
        * SCALE
        * (((dst5 >> 31) + 1).to(tl.float32))
    )
    w6 = (
        tl.load(weights_ptr + meta + 6).to(tl.float32)
        * SCALE
        * (((dst6 >> 31) + 1).to(tl.float32))
    )
    w7 = (
        tl.load(weights_ptr + meta + 7).to(tl.float32)
        * SCALE
        * (((dst7 >> 31) + 1).to(tl.float32))
    )
    b0 = tl.maximum(dst0, 0) * hidden
    b1 = tl.maximum(dst1, 0) * hidden
    b2 = tl.maximum(dst2, 0) * hidden
    b3 = tl.maximum(dst3, 0) * hidden
    b4 = tl.maximum(dst4, 0) * hidden
    b5 = tl.maximum(dst5, 0) * hidden
    b6 = tl.maximum(dst6, 0) * hidden
    b7 = tl.maximum(dst7, 0) * hidden

    cols = tl.arange(0, CHUNK)
    if EVEN:
        v0 = tl.load(down_ptr + b0 + cols).to(tl.float32)
        v1 = tl.load(down_ptr + b1 + cols).to(tl.float32)
        v2 = tl.load(down_ptr + b2 + cols).to(tl.float32)
        v3 = tl.load(down_ptr + b3 + cols).to(tl.float32)
        v4 = tl.load(down_ptr + b4 + cols).to(tl.float32)
        v5 = tl.load(down_ptr + b5 + cols).to(tl.float32)
        v6 = tl.load(down_ptr + b6 + cols).to(tl.float32)
        v7 = tl.load(down_ptr + b7 + cols).to(tl.float32)
        acc = (
            v0 * w0
            + v1 * w1
            + v2 * w2
            + v3 * w3
            + v4 * w4
            + v5 * w5
            + v6 * w6
            + v7 * w7
        )
        tl.store(out_ptr + out_base + cols, acc.to(out_ptr.dtype.element_ty))
    else:
        m = cols < hidden
        v0 = tl.load(down_ptr + b0 + cols, mask=m, other=0.0).to(tl.float32)
        v1 = tl.load(down_ptr + b1 + cols, mask=m, other=0.0).to(tl.float32)
        v2 = tl.load(down_ptr + b2 + cols, mask=m, other=0.0).to(tl.float32)
        v3 = tl.load(down_ptr + b3 + cols, mask=m, other=0.0).to(tl.float32)
        v4 = tl.load(down_ptr + b4 + cols, mask=m, other=0.0).to(tl.float32)
        v5 = tl.load(down_ptr + b5 + cols, mask=m, other=0.0).to(tl.float32)
        v6 = tl.load(down_ptr + b6 + cols, mask=m, other=0.0).to(tl.float32)
        v7 = tl.load(down_ptr + b7 + cols, mask=m, other=0.0).to(tl.float32)
        acc = (
            v0 * w0
            + v1 * w1
            + v2 * w2
            + v3 * w3
            + v4 * w4
            + v5 * w5
            + v6 * w6
            + v7 * w7
        )
        tl.store(
            out_ptr + out_base + cols, acc.to(out_ptr.dtype.element_ty), mask=m
        )


@triton.jit
def _post_reorder_fullrow_i64(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    weights_ptr,
    hidden,
    SCALE,
    CHUNK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """int64-addressing twin for shapes whose row space exceeds int32."""
    pid_t = tl.program_id(0)
    meta = pid_t * 8
    out_base = pid_t.to(tl.int64) * hidden

    dst0 = tl.load(src2dst_ptr + meta + 0).to(tl.int64)
    dst1 = tl.load(src2dst_ptr + meta + 1).to(tl.int64)
    dst2 = tl.load(src2dst_ptr + meta + 2).to(tl.int64)
    dst3 = tl.load(src2dst_ptr + meta + 3).to(tl.int64)
    dst4 = tl.load(src2dst_ptr + meta + 4).to(tl.int64)
    dst5 = tl.load(src2dst_ptr + meta + 5).to(tl.int64)
    dst6 = tl.load(src2dst_ptr + meta + 6).to(tl.int64)
    dst7 = tl.load(src2dst_ptr + meta + 7).to(tl.int64)
    w0 = (
        tl.load(weights_ptr + meta + 0).to(tl.float32)
        * SCALE
        * (((dst0 >> 31) + 1).to(tl.float32))
    )
    w1 = (
        tl.load(weights_ptr + meta + 1).to(tl.float32)
        * SCALE
        * (((dst1 >> 31) + 1).to(tl.float32))
    )
    w2 = (
        tl.load(weights_ptr + meta + 2).to(tl.float32)
        * SCALE
        * (((dst2 >> 31) + 1).to(tl.float32))
    )
    w3 = (
        tl.load(weights_ptr + meta + 3).to(tl.float32)
        * SCALE
        * (((dst3 >> 31) + 1).to(tl.float32))
    )
    w4 = (
        tl.load(weights_ptr + meta + 4).to(tl.float32)
        * SCALE
        * (((dst4 >> 31) + 1).to(tl.float32))
    )
    w5 = (
        tl.load(weights_ptr + meta + 5).to(tl.float32)
        * SCALE
        * (((dst5 >> 31) + 1).to(tl.float32))
    )
    w6 = (
        tl.load(weights_ptr + meta + 6).to(tl.float32)
        * SCALE
        * (((dst6 >> 31) + 1).to(tl.float32))
    )
    w7 = (
        tl.load(weights_ptr + meta + 7).to(tl.float32)
        * SCALE
        * (((dst7 >> 31) + 1).to(tl.float32))
    )
    b0 = tl.maximum(dst0, 0) * hidden
    b1 = tl.maximum(dst1, 0) * hidden
    b2 = tl.maximum(dst2, 0) * hidden
    b3 = tl.maximum(dst3, 0) * hidden
    b4 = tl.maximum(dst4, 0) * hidden
    b5 = tl.maximum(dst5, 0) * hidden
    b6 = tl.maximum(dst6, 0) * hidden
    b7 = tl.maximum(dst7, 0) * hidden

    cols = tl.arange(0, CHUNK)
    if EVEN:
        v0 = tl.load(down_ptr + b0 + cols).to(tl.float32)
        v1 = tl.load(down_ptr + b1 + cols).to(tl.float32)
        v2 = tl.load(down_ptr + b2 + cols).to(tl.float32)
        v3 = tl.load(down_ptr + b3 + cols).to(tl.float32)
        v4 = tl.load(down_ptr + b4 + cols).to(tl.float32)
        v5 = tl.load(down_ptr + b5 + cols).to(tl.float32)
        v6 = tl.load(down_ptr + b6 + cols).to(tl.float32)
        v7 = tl.load(down_ptr + b7 + cols).to(tl.float32)
        acc = (
            v0 * w0
            + v1 * w1
            + v2 * w2
            + v3 * w3
            + v4 * w4
            + v5 * w5
            + v6 * w6
            + v7 * w7
        )
        tl.store(out_ptr + out_base + cols, acc.to(out_ptr.dtype.element_ty))
    else:
        m = cols < hidden
        v0 = tl.load(down_ptr + b0 + cols, mask=m, other=0.0).to(tl.float32)
        v1 = tl.load(down_ptr + b1 + cols, mask=m, other=0.0).to(tl.float32)
        v2 = tl.load(down_ptr + b2 + cols, mask=m, other=0.0).to(tl.float32)
        v3 = tl.load(down_ptr + b3 + cols, mask=m, other=0.0).to(tl.float32)
        v4 = tl.load(down_ptr + b4 + cols, mask=m, other=0.0).to(tl.float32)
        v5 = tl.load(down_ptr + b5 + cols, mask=m, other=0.0).to(tl.float32)
        v6 = tl.load(down_ptr + b6 + cols, mask=m, other=0.0).to(tl.float32)
        v7 = tl.load(down_ptr + b7 + cols, mask=m, other=0.0).to(tl.float32)
        acc = (
            v0 * w0
            + v1 * w1
            + v2 * w2
            + v3 * w3
            + v4 * w4
            + v5 * w5
            + v6 * w6
            + v7 * w7
        )
        tl.store(
            out_ptr + out_base + cols, acc.to(out_ptr.dtype.element_ty), mask=m
        )


@triton.jit
def _post_reorder_generic_i32(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    weights_ptr,
    hidden,
    SCALE,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    NCHUNK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """Generic-topk fallback: static chunk loop, scalar index per slot."""
    pid_t = tl.program_id(0)
    meta = pid_t * TOPK
    out_base = pid_t * hidden

    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        acc = tl.zeros([CHUNK], dtype=tl.float32)
        for i in tl.static_range(TOPK):
            dst = tl.load(src2dst_ptr + meta + i)
            indicator = ((dst >> 31) + 1).to(tl.float32)
            w = (
                tl.load(weights_ptr + meta + i).to(tl.float32)
                * SCALE
                * indicator
            )
            dstc = tl.maximum(dst, 0)
            if EVEN:
                vec = tl.load(down_ptr + dstc * hidden + cols)
            else:
                m = cols < hidden
                vec = tl.load(
                    down_ptr + dstc * hidden + cols, mask=m, other=0.0
                )
            acc += vec.to(tl.float32) * w
        if EVEN:
            tl.store(
                out_ptr + out_base + cols, acc.to(out_ptr.dtype.element_ty)
            )
        else:
            tl.store(
                out_ptr + out_base + cols,
                acc.to(out_ptr.dtype.element_ty),
                mask=cols < hidden,
            )


@triton.jit
def _post_reorder_generic_i64(
    down_ptr,
    out_ptr,
    src2dst_ptr,
    weights_ptr,
    hidden,
    SCALE,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    NCHUNK: tl.constexpr,
    EVEN: tl.constexpr,
):
    """int64 twin of the generic-topk fallback."""
    pid_t = tl.program_id(0)
    meta = pid_t * TOPK
    out_base = pid_t.to(tl.int64) * hidden

    for c in tl.static_range(NCHUNK):
        cols = c * CHUNK + tl.arange(0, CHUNK)
        acc = tl.zeros([CHUNK], dtype=tl.float32)
        for i in tl.static_range(TOPK):
            dst = tl.load(src2dst_ptr + meta + i).to(tl.int64)
            indicator = ((dst >> 31) + 1).to(tl.float32)
            w = (
                tl.load(weights_ptr + meta + i).to(tl.float32)
                * SCALE
                * indicator
            )
            dstc = tl.maximum(dst, 0)
            if EVEN:
                vec = tl.load(down_ptr + dstc * hidden + cols)
            else:
                m = cols < hidden
                vec = tl.load(
                    down_ptr + dstc * hidden + cols, mask=m, other=0.0
                )
            acc += vec.to(tl.float32) * w
        if EVEN:
            tl.store(
                out_ptr + out_base + cols, acc.to(out_ptr.dtype.element_ty)
            )
        else:
            tl.store(
                out_ptr + out_base + cols,
                acc.to(out_ptr.dtype.element_ty),
                mask=cols < hidden,
            )


def _launch_fullrow(
    down_output,
    output,
    src2dst,
    topk_weights,
    num_tokens,
    hidden,
    scale,
    use_i32,
):
    # One whole-row chunk per program: the full gather+merge+store runs as a
    # single fully-parallel load batch.  8 elements (one 128-bit access) per
    # thread with 8 warps measured best across the sweep.
    chunk = _CHUNK_FULL_ROW
    even = chunk == hidden
    if use_i32:
        _post_reorder_fullrow_i32[(num_tokens,)](
            down_output,
            output,
            src2dst,
            topk_weights,
            hidden,
            float(scale),
            CHUNK=chunk,
            EVEN=even,
            num_warps=8,
            num_stages=2,
        )
    else:
        _post_reorder_fullrow_i64[(num_tokens,)](
            down_output,
            output,
            src2dst,
            topk_weights,
            hidden,
            float(scale),
            CHUNK=chunk,
            EVEN=even,
            num_warps=8,
            num_stages=2,
        )
    return output


def _launch_generic(
    down_output,
    output,
    src2dst,
    topk_weights,
    num_tokens,
    topk,
    hidden,
    scale,
    use_i32,
):
    nchunk = (hidden + _CHUNK_GENERIC - 1) // _CHUNK_GENERIC
    even = hidden % _CHUNK_GENERIC == 0
    if use_i32:
        _post_reorder_generic_i32[(num_tokens,)](
            down_output,
            output,
            src2dst,
            topk_weights,
            hidden,
            float(scale),
            TOPK=topk,
            CHUNK=_CHUNK_GENERIC,
            NCHUNK=nchunk,
            EVEN=even,
            num_warps=4,
            num_stages=2,
        )
    else:
        _post_reorder_generic_i64[(num_tokens,)](
            down_output,
            output,
            src2dst,
            topk_weights,
            hidden,
            float(scale),
            TOPK=topk,
            CHUNK=_CHUNK_GENERIC,
            NCHUNK=nchunk,
            EVEN=even,
            num_warps=4,
            num_stages=2,
        )
    return output


def deepep_post_reorder(
    down_output,
    output,
    src2dst,
    topk_ids,
    topk_weights,
    topk,
    hidden_size,
    routed_scaling_factor,
):
    num_tokens, topk = src2dst.shape
    hidden = down_output.shape[-1]

    # int32 addressing is safe while the largest gathered offset fits;
    # otherwise take the (slower) int64 twin.
    use_i32 = num_tokens * topk * hidden < _INT32_LIMIT

    if topk == 8 and hidden <= _CHUNK_FULL_ROW:
        return _launch_fullrow(
            down_output,
            output,
            src2dst,
            topk_weights,
            num_tokens,
            hidden,
            routed_scaling_factor,
            use_i32,
        )
    return _launch_generic(
        down_output,
        output,
        src2dst,
        topk_weights,
        num_tokens,
        topk,
        hidden,
        routed_scaling_factor,
        use_i32,
    )


__all__ = ["deepep_post_reorder"]
