# Copyright 2026, The FlagOS Contributors.
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

"""Fused eight-bit cast plus paged MLA KV write."""

import torch
import triton
import triton.language as tl

__all__ = ["set_mla_kv_buffer_fp8_quant"]

_BLOCK_D = 128

_FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
)


@triton.jit
def _set_mla_kv_buffer_fp8_quant_kernel(
    dst,
    nope,
    rope,
    loc,
    rows,
    DST_S0: tl.constexpr,
    NOPE_S0: tl.constexpr,
    ROPE_S0: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)

    offs = block * BLOCK + tl.arange(0, BLOCK)
    total_dim = NOPE_DIM + ROPE_DIM
    mask = offs < total_dim

    target = tl.load(loc + token).to(tl.int64)
    in_range = (target >= 0) & (target < rows)
    # Clamp the row before forming the address so a masked-out lane never
    # produces an out-of-range pointer value, and keep ``in_range`` in the
    # store mask so the clamped row is not actually written.
    safe_target = tl.where(in_range, target, 0)

    is_nope = offs < NOPE_DIM
    # Both column indices are clamped into their own part, so only lanes that
    # the mask keeps can ever be dereferenced.
    nope_col = tl.where(is_nope, offs, 0)
    rope_col = tl.where(is_nope, 0, offs - NOPE_DIM)

    value_nope = tl.load(
        nope + token * NOPE_S0 + nope_col,
        mask=mask & is_nope,
        other=0.0,
    )
    value_rope = tl.load(
        rope + token * ROPE_S0 + rope_col,
        mask=mask & ~is_nope,
        other=0.0,
    )
    value = tl.where(is_nope, value_nope, value_rope)

    # ``dst`` is an FP8 view of the buffer copy, so this store performs the
    # BF16/FP16 -> FP8 downcast with the saturating conversion and no explicit
    # clamp is needed here.
    tl.store(dst + safe_target * DST_S0 + offs, value, mask=mask & in_range)


def set_mla_kv_buffer_fp8_quant(
    kv_buffer, loc, cache_k_nope, cache_k_rope, fp8_dtype
):
    """Concatenate the two key parts, cast to ``fp8_dtype``, scatter-write rows.

    ``kv_buffer`` is a paged MLA key cache holding one concatenated row per
    slot; the NoPE and RoPE halves arrive separately in ``cache_k_nope`` and
    ``cache_k_rope`` and are joined along the last axis before the cast.

    The contract declares no mutation, so the write lands in an FP8 view of a
    copy of ``kv_buffer`` and the copy is returned. Slots outside
    ``[0, kv_buffer.shape[0])`` are ignored: they leave the copied row
    untouched instead of wrapping around or writing out of bounds. Repeated
    slots are undefined, since the racing programs may finish in any order.
    """
    if not isinstance(kv_buffer, torch.Tensor):
        raise TypeError("kv_buffer must be a torch.Tensor")
    if kv_buffer.dim() != 2:
        raise ValueError(
            f"kv_buffer must be 2-D, got shape {tuple(kv_buffer.shape)}"
        )
    if fp8_dtype not in _FP8_DTYPES:
        raise TypeError(f"fp8_dtype must be an FP8 dtype, got {fp8_dtype}")
    if kv_buffer.dtype != fp8_dtype and kv_buffer.element_size() != 1:
        # A byte-width mismatch would make ``view`` change the row length, so
        # only byte-typed buffers may be reinterpreted as FP8.
        raise TypeError(
            f"kv_buffer dtype {kv_buffer.dtype} cannot be viewed as "
            f"{fp8_dtype}; expected an FP8 or byte-typed buffer"
        )
    if kv_buffer.stride(1) != 1 or kv_buffer.stride(0) != kv_buffer.shape[1]:
        raise ValueError("kv_buffer must be row-major contiguous")

    for name, part in (
        ("cache_k_nope", cache_k_nope),
        ("cache_k_rope", cache_k_rope),
    ):
        if not isinstance(part, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if part.dim() != 2:
            raise ValueError(
                f"{name} must be 2-D, got shape {tuple(part.shape)}"
            )
        if part.stride(1) != 1 or part.stride(0) != part.shape[1]:
            raise ValueError(f"{name} must be row-major contiguous")

    if not isinstance(loc, torch.Tensor):
        raise TypeError("loc must be a torch.Tensor")
    if loc.dim() != 1:
        raise ValueError(f"loc must be 1-D, got shape {tuple(loc.shape)}")
    if loc.dtype not in (
        torch.int32,
        torch.int64,
        torch.int16,
        torch.uint8,
    ):
        raise TypeError(f"loc must be an integer tensor, got {loc.dtype}")

    count = loc.numel()
    if cache_k_nope.shape[0] != count or cache_k_rope.shape[0] != count:
        raise ValueError(
            "cache_k_nope and cache_k_rope must have one row per loc entry, "
            f"got {cache_k_nope.shape[0]} / {cache_k_rope.shape[0]} rows "
            f"for {count} slots"
        )

    out = kv_buffer.clone()
    # The FP8 view shares storage with ``out`` and is only the write target, so
    # the caller gets its copy back in the dtype it handed in.
    buffer = out if out.dtype == fp8_dtype else out.view(fp8_dtype)

    if count == 0:
        return out

    nope_dim = cache_k_nope.shape[1]
    rope_dim = cache_k_rope.shape[1]
    total_dim = nope_dim + rope_dim
    if total_dim > kv_buffer.shape[1]:
        raise ValueError(
            f"concatenated width {total_dim} exceeds the buffer row width "
            f"{kv_buffer.shape[1]}"
        )

    grid = (count, triton.cdiv(total_dim, _BLOCK_D))
    _set_mla_kv_buffer_fp8_quant_kernel[grid](
        buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        kv_buffer.shape[0],
        buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        _BLOCK_D,
        num_warps=4,
        num_stages=1,
    )
    return out
