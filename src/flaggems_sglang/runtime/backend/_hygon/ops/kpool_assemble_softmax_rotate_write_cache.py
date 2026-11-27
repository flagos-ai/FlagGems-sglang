# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Set GET_FLAGGEMS_VERSION before importing FlagGems. The default v0 uses
Triton layouts and Hadamard stages. v1 uses TLE on the supported gfx936 path.
"""

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flaggems_sglang.ops.kpool_assemble_softmax_rotate_write_cache import (
    INDEX_HEAD_DIM,
    _hadamard128,
)
from flaggems_sglang.ops.kpool_assemble_softmax_rotate_write_cache import (
    _kpool_assemble_softmax_rotate_write_cache_kernel as _assemble_online,
)
from flaggems_sglang.ops.kpool_assemble_softmax_rotate_write_cache import (
    _log_kpool_wrapper_call,
    _round_to_nearest_int,
)

# Packed row descriptors change alignment with the inference batch size.
_ROW_POINTERS = (
    "req_pool_idx_ptr",
    "n_from_tail_ptr",
    "chunk_src_start_ptr",
    "tail_logical_base_ptr",
    "loc_ptr",
    "write_mask_ptr",
)
_assemble_online = triton.jit(
    _assemble_online.fn,
    do_not_specialize_on_alignment=_ROW_POINTERS,
)

_selected_version = os.environ.get("GET_FLAGGEMS_VERSION", "v0")
if _selected_version not in ("v0", "v1"):
    raise ValueError(
        "GET_FLAGGEMS_VERSION must be 'v0' or 'v1', "
        f"got {_selected_version!r}"
    )

tle = None
if _selected_version == "v1":
    try:
        from triton.experimental.tle import language as tle
    except ImportError as error:
        raise ImportError(
            "GET_FLAGGEMS_VERSION=v1 requires Triton TLE with explicit-layout "
            "support; use GET_FLAGGEMS_VERSION=v0 for the implementation without TLE."
        ) from error
    if not all(
        hasattr(getattr(tle, "gpu", None), name)
        for name in ("set_layout", "BlockEncoding", "SlicedEncoding")
    ):
        raise ImportError(
            "GET_FLAGGEMS_VERSION=v1 requires Triton TLE explicit-layout support."
        )


@lru_cache(maxsize=16)
def _supports_wave64(device_index):
    if not triton.__version__.startswith("3.6."):
        return False
    props = torch.cuda.get_device_properties(device_index)
    return getattr(props, "gcnArchName", "").split(":")[0] == "gfx936"


_PAIR_BUTTERFLY = tl.constexpr("v_add_f32 $0, $2, $3\nv_sub_f32 $1, $2, $3")
_DPP_BUTTERFLIES = tl.constexpr(
    tuple(
        "v_xor_b32 $0, $2, $4\n"
        "v_xor_b32 $1, $3, $5\n"
        f"v_add_f32_dpp $0, $2, $0 {permute} row_mask:0xf bank_mask:0xf bound_ctrl:0\n"
        f"v_add_f32_dpp $1, $3, $1 {permute} row_mask:0xf bank_mask:0xf bound_ctrl:0"
        for permute in (
            "quad_perm:[1,0,3,2]",
            "quad_perm:[2,3,0,1]",
            "row_ror:8",
        )
    )
)
_TLE_SWIZZLES = tl.constexpr(
    tuple(
        f"ds_swizzle_b32 $0, $2 offset:{31 | ((1 << bit) << 10)}\n"
        f"ds_swizzle_b32 $1, $3 offset:{31 | ((1 << bit) << 10)}\ns_waitcnt lgkmcnt(0)"
        for bit in range(5)
    )
)


@triton.jit
def _load_row_descriptors(row, loc, count, chunk_start, req, tail_base):
    # Active row only. Issue independent scalar reads before waiting. loc,
    # chunk_start and req use the low dword of i64 elements after the i32
    # addressing check. count and tail_base are i32 arrays.
    return tl.inline_asm_elementwise(
        "s_load_dword $0, $5, $10\n"
        "s_load_dword $1, $6, $11\n"
        "s_load_dword $2, $7, $10\n"
        "s_load_dword $3, $8, $10\n"
        "s_load_dword $4, $9, $11\n"
        "s_waitcnt lgkmcnt(0)",
        constraints="=&s,=&s,=&s,=&s,=&s,s,s,s,s,s,s,s",
        args=[
            loc.to(tl.uint64),
            count.to(tl.uint64),
            chunk_start.to(tl.uint64),
            req.to(tl.uint64),
            tail_base.to(tl.uint64),
            row * 8,
            row * 4,
        ],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def _load_mixed(chunk, tail, chunk_rows, tail_rows, from_tail, cols):
    if chunk.dtype == tail.dtype:
        # The entire 128-column slot uses the same source address.
        base = tl.where(
            from_tail,
            (tail + tail_rows).to(tl.uint64),
            (chunk + chunk_rows).to(tl.uint64),
        ).to(chunk.dtype)
        return tl.load(base[:, None] + cols[None, :]).to(tl.float32)
    else:
        # Different score dtypes need separate typed loads.
        tail_values = tl.load(
            tail + tail_rows[:, None] + cols[None, :],
            from_tail[:, None],
            other=0,
        )
        chunk_values = tl.load(
            chunk + chunk_rows[:, None] + cols[None, :],
            ~from_tail[:, None],
            other=0,
        )
        return tl.where(
            from_tail[:, None],
            tail_values.to(tl.float32),
            chunk_values.to(tl.float32),
        )


@triton.jit
def _pool_k4(
    chunk,
    tail,
    ape,
    meta,
    cols,
    layout: tl.constexpr,
    tail_stride_0: tl.constexpr,
    tail_stride_1: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    CHUNK_ONLY: tl.constexpr,
):
    n_tail, start, req, tail_base = meta
    chunk_key, chunk_score = chunk
    tail_key, tail_score = tail
    slots = tl.arange(0, 4)
    if layout is not None:
        slots = tle.gpu.set_layout(slots, tle.gpu.SlicedEncoding(1, layout))
    chunk_rows = (start + slots - n_tail) * 128
    if CHUNK_ONLY:
        offsets = chunk_rows[:, None] + cols[None, :]
        keys = tl.load(chunk_key + offsets).to(tl.float32)
        scores = tl.load(chunk_score + offsets).to(tl.float32)
    else:
        # Reduce the start before adding slots to avoid i32 overflow.
        pos = (tail_base.to(tl.uint32) % TAIL_SIZE).to(tl.int32) + slots
        if TAIL_SIZE >= 4:
            pos = tl.where(pos >= TAIL_SIZE, pos - TAIL_SIZE, pos)
        else:
            pos = pos % TAIL_SIZE
        tail_rows = req * tail_stride_0 + pos * tail_stride_1
        keys = _load_mixed(
            chunk_key,
            tail_key,
            chunk_rows,
            tail_rows,
            slots < n_tail,
            cols,
        )
        scores = _load_mixed(
            chunk_score,
            tail_score,
            chunk_rows,
            tail_rows,
            slots < n_tail,
            cols,
        )
    logits = scores + tl.load(ape + slots[:, None] * 128 + cols[None, :]).to(
        tl.float32
    )
    weights = tl.exp(logits - tl.max(logits, axis=0, keep_dims=True))
    numerator = tl.sum(keys * weights, axis=0, keep_dims=True)
    denominator = tl.sum(weights, axis=0, keep_dims=True)
    return _normalize_pooled(numerator, denominator)


@triton.jit
def _reciprocal(value):
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        constraints="=v,v",
        args=[value],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _normalize_pooled(numerator, denominator):
    # Correct the reciprocal estimate before the first BF16 rounding.
    inv = _reciprocal(denominator)
    estimate = numerator * inv
    refined = tl.fma(tl.fma(-denominator, estimate, numerator), inv, estimate)
    # For infinite numerators the residual contains inf - inf.
    quotient = tl.where(tl.abs(numerator) < float("inf"), refined, estimate)
    return quotient.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _wave_hadamard(x, offs, layout: tl.constexpr):
    # TLE keeps columns 2*i and 2*i+1 adjacent in lane i's registers.
    x = tl.inline_asm_elementwise(
        _PAIR_BUTTERFLY,
        constraints="=&v,=&v,v,v",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=2,
    )
    x = tle.gpu.set_layout(x, layout)
    for bit in tl.static_range(1, 7):
        if bit == 1 or bit == 2 or bit == 4:
            # For these stages DPP can permute one input to the add.
            # Toggling x's sign on the upper lanes gives peer - x.
            sign = (offs[None, :].to(tl.uint32) << (31 - bit)) & 0x80000000
            x = tl.inline_asm_elementwise(
                _DPP_BUTTERFLIES[0 if bit == 1 else 1 if bit == 2 else 2],
                constraints="=&v,=&v,v,v,v,v",
                args=[x, sign],
                dtype=tl.float32,
                is_pure=True,
                pack=2,
            )
            x = tle.gpu.set_layout(x, layout)
        else:
            if bit < 6:
                other = tl.inline_asm_elementwise(
                    _TLE_SWIZZLES[bit - 1],
                    constraints="=&v,=&v,v,v",
                    args=[x],
                    dtype=tl.float32,
                    is_pure=True,
                    pack=2,
                )
                other = tle.gpu.set_layout(other, layout)
            else:
                other = tl.gather(x, (offs ^ (1 << bit))[None, :], axis=1)
            x = tl.where(
                (offs[None, :] & (1 << bit)) == 0, x + other, other - x
            )
    return x * 0.08838834764831845


@triton.jit
def _quantize_int8(x, ROUND_SCALE: tl.constexpr):
    absmax = tl.maximum(tl.max(tl.abs(x), 1, keep_dims=True), 1e-4)
    scale = absmax * (1.0 / 127.0)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    normalized = tl.minimum(tl.maximum(x / scale, -127.0), 127.0)
    return _round_to_nearest_int(normalized).to(tl.int8), scale


@triton.jit(do_not_specialize_on_alignment=_ROW_POINTERS)
def _assemble_k4(
    buf_i8_ptr,
    buf_fp32_ptr,
    chunk_k_ptr,
    chunk_score_ptr,
    tail_k_ptr,
    tail_score_ptr,
    req_pool_idx_ptr,
    n_from_tail_ptr,
    chunk_src_start_ptr,
    tail_logical_base_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    tail_stride_0: tl.constexpr,
    tail_stride_1: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    SLOTS_PER_PAGE: tl.constexpr,
    USE_TLE: tl.constexpr,
    PACKED_MASK: tl.constexpr,
):
    row = tl.program_id(0)
    if write_mask_ptr is not None:
        # Tie row to the scalar kernel arguments so gfx936 can load the
        # bases together before the first dependent row access. No tensor
        # is dereferenced by this register-only identity.
        row = tl.inline_asm_elementwise(
            "",
            constraints="=s,0,s,s,s,s,s,s,s,s,s,s,s,s",
            args=[
                row,
                req_pool_idx_ptr.to(tl.uint64),
                n_from_tail_ptr.to(tl.uint64),
                chunk_src_start_ptr.to(tl.uint64),
                tail_logical_base_ptr.to(tl.uint64),
                loc_ptr.to(tl.uint64),
                ape_ptr.to(tl.uint64),
                chunk_k_ptr.to(tl.uint64),
                chunk_score_ptr.to(tl.uint64),
                tail_k_ptr.to(tl.uint64),
                tail_score_ptr.to(tl.uint64),
                buf_fp32_ptr.to(tl.uint64),
                buf_i8_ptr.to(tl.uint64),
            ],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        tl.assume(row >= 0)
        if PACKED_MASK:
            word = tl.load(
                write_mask_ptr.to(tl.pointer_type(tl.uint32)) + row // 4
            )
            live = ((word >> (8 * (row % 4))) & 255) != 0
        else:
            live = tl.load(write_mask_ptr + row)
        if not live:
            return

    cols = tl.arange(0, 128)
    if USE_TLE:
        # Each of the 64 lanes owns two adjacent columns across all K slots.
        layout: tl.constexpr = tle.gpu.BlockEncoding(
            (1, 2), (1, 64), (1, 1), (1, 0)
        )
        cols = tle.gpu.set_layout(cols, tle.gpu.SlicedEncoding(0, layout))
    else:
        layout: tl.constexpr = None

    loc, n_tail, start, req, tail_base = _load_row_descriptors(
        row,
        loc_ptr,
        n_from_tail_ptr,
        chunk_src_start_ptr,
        req_pool_idx_ptr,
        tail_logical_base_ptr,
    )
    chunk = (chunk_k_ptr, chunk_score_ptr)
    tail = (tail_k_ptr, tail_score_ptr)
    meta = (n_tail, start, req, tail_base)
    if n_tail == 0:
        pooled = _pool_k4(
            chunk,
            tail,
            ape_ptr,
            meta,
            cols,
            layout,
            tail_stride_0,
            tail_stride_1,
            TAIL_SIZE,
            True,
        )
    else:
        pooled = _pool_k4(
            chunk,
            tail,
            ape_ptr,
            meta,
            cols,
            layout,
            tail_stride_0,
            tail_stride_1,
            TAIL_SIZE,
            False,
        )
    if USE_TLE:
        rotated = _wave_hadamard(pooled, cols, layout)
    else:
        rotated = tl.reshape(
            _hadamard128(tl.reshape(pooled, (128,))), (1, 128)
        )
    quantized, scale = _quantize_int8(
        rotated.to(tl.bfloat16).to(tl.float32), ROUND_SCALE
    )

    page, token = loc // SLOTS_PER_PAGE, loc % SLOTS_PER_PAGE
    key_offset = page * BUF_NUMEL_PER_PAGE + token * 128 + cols
    scale_offset = page * BUF_NUMEL_PER_PAGE // 4 + SLOTS_PER_PAGE * 32 + token
    tl.store(buf_i8_ptr + key_offset[None, :], quantized)
    # Keep the store in the wave layout. Only column zero writes the shared scale.
    scales = tl.broadcast_to(scale, (1, 128))
    if USE_TLE:
        scales = tle.gpu.set_layout(scales, layout)
    tl.store(
        buf_fp32_ptr + scale_offset + cols[None, :],
        scales,
        mask=cols[None, :] == 0,
    )


def _fits_i32_offsets(chunk_k, tail_k, buf):
    """Bound actual element offsets, including the outer strides of a tail view."""
    limit = 0x7FFFFFFF
    last_tail = (
        (tail_k.shape[0] - 1) * tail_k.stride(0)
        + (tail_k.shape[1] - 1) * tail_k.stride(1)
        + INDEX_HEAD_DIM
        - 1
    )
    return (
        max(chunk_k.numel(), last_tail, buf.numel(), tail_k.shape[1] + 4)
        <= limit
    )


def kpool_assemble_softmax_rotate_write_cache(
    pool,
    buf: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = False,
) -> None:
    if os.environ.get("SGLANG_KPOOL_CALL_LOG"):
        _log_kpool_wrapper_call(
            "kpool_assemble_softmax_rotate_write_cache",
            pool=pool,
            buf=buf,
            chunk_k=chunk_k,
            chunk_score=chunk_score,
            tail_k=tail_k,
            tail_score=tail_score,
            req_pool_idx=req_pool_idx,
            n_from_tail=n_from_tail,
            chunk_src_start=chunk_src_start,
            tail_logical_base=tail_logical_base,
            ape=ape,
            loc=loc,
            write_mask=write_mask,
            round_scale=round_scale,
        )

    kpool = pool.index_kpool
    n_pools = req_pool_idx.shape[0]
    if n_pools == 0:
        return

    chunk_k = chunk_k.contiguous()
    chunk_score = chunk_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    has_write_mask = write_mask is not None
    if has_write_mask:
        write_mask = write_mask.contiguous()

    buf_i8 = buf.view(torch.int8)
    buf_fp32 = buf.view(torch.float32)
    slots_per_page = pool.slots_per_page

    use_tle = _selected_version == "v1"
    use_wave64 = (
        kpool == 4
        # v0 keeps the online kernel for small launches.
        and (use_tle or n_pools > 64)
        and chunk_k.is_cuda
        and _supports_wave64(chunk_k.get_device())
        and loc.dtype
        == chunk_src_start.dtype
        == req_pool_idx.dtype
        == torch.int64
        and n_from_tail.dtype == tail_logical_base.dtype == torch.int32
        # The scalar descriptor loads use a signed 32-bit byte offset (row * 8).
        and n_pools <= (1 << 28)
        and tail_k.shape[1] > 0
        and _fits_i32_offsets(chunk_k, tail_k, buf)
    )
    kernel = _assemble_k4 if use_wave64 else _assemble_online
    if use_wave64:
        use_mask_word = (
            has_write_mask
            and write_mask.dtype == torch.bool
            and n_pools % 4 == 0
            and write_mask.data_ptr() % 4 == 0
        )
        options = {"USE_TLE": use_tle, "PACKED_MASK": use_mask_word}
    else:
        # The online loop merges the chunk and tail offsets. Give both i64
        # row indices, including when a caller supplies an alternate dtype.
        req_pool_idx = req_pool_idx.long()
        chunk_src_start = chunk_src_start.long()
        options = dict(
            chunk_stride_0=chunk_k.stride(0),
            ape_stride_0=ape.stride(0),
            HEAD_DIM=INDEX_HEAD_DIM,
            BLOCK_D=INDEX_HEAD_DIM,
            S_OFFSET_NBYTES_IN_PAGE=slots_per_page * INDEX_HEAD_DIM,
            HAS_WRITE_MASK=has_write_mask,
            POOL_SIZE=kpool,
        )
    kernel.run(
        buf_i8,
        buf_fp32,
        chunk_k,
        chunk_score,
        tail_k,
        tail_score,
        req_pool_idx,
        n_from_tail,
        chunk_src_start,
        tail_logical_base,
        ape,
        loc,
        write_mask,
        tail_stride_0=tail_k.stride(0),
        tail_stride_1=tail_k.stride(1),
        BUF_NUMEL_PER_PAGE=buf.shape[1],
        TAIL_SIZE=tail_k.shape[1],
        ROUND_SCALE=round_scale,
        SLOTS_PER_PAGE=slots_per_page,
        grid=(n_pools,),
        warmup=False,
        num_warps=1 if use_wave64 else 4,
        **options,
    )


__all__ = ["kpool_assemble_softmax_rotate_write_cache"]
