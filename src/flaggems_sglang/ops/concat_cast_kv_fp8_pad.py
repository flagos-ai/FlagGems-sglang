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

"""Fused non-prefix Q8KV8 KV preparation for DSA sparse prefill.

The bf16 path materialises the whole row in bf16 and then converts it:

1. ``torch.cat([k_nope, k_rope], -1)`` allocates ``[tokens, 576]`` bf16,
2. a ``.copy_`` cast writes it into the persistent fp8 ``kv_buf``,
3. ``.zero_()`` clears the trailing pad band.

All three are fused here into a single row-per-program pass that casts and
writes straight into the fp8 buffer, and zeroes the pad band in the same
launch. The band exists because the sparse-attention index clamp maps a
``-1`` sentinel to the last row, so those rows must hold zeros rather than
whatever the allocator left behind.

A row is ``[NOPE fp8 bytes | ROPE fp8 bytes]`` with no per-tile scales --
unlike the packed layout the dequantize kernels consume, this buffer stores
plain e4m3, matching what the gather kernel writes (bit-identical bytes).

``ROPE == 0`` is a first-class case: NoPE models (zero-width rope views)
pass it and ``tl.arange`` rejects empty ranges, so the rope segment is
skipped statically rather than masked at runtime.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _concat_cast_kv_fp8_pad_kernel(
    out_ptr,
    k_ptr,
    kr_ptr,
    num_tokens,
    k_stride,
    kr_stride,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    NOPE_P2: tl.constexpr,
    ROPE_P2: tl.constexpr,
):
    """One program per destination row.

    Real rows (``row < num_tokens``) write ``cast(k) || cast(k_rope)``;
    the trailing pad band writes zeros. ``*_P2`` are the next powers of two
    so ``tl.arange`` is legal for widths that are not powers of two (they
    are masked back to the exact width, so 512/64 behave exactly as
    unmasked code would).
    """
    row = tl.program_id(0).to(tl.int64)
    head = NOPE + ROPE

    offs_n = tl.arange(0, NOPE_P2)
    mask_n = offs_n < NOPE

    if row < num_tokens:
        v_n = tl.load(k_ptr + row * k_stride + offs_n, mask=mask_n, other=0.0)
        tl.store(
            out_ptr + row * head + offs_n,
            v_n.to(out_ptr.dtype.element_ty),
            mask=mask_n,
        )
        if ROPE > 0:
            offs_r = tl.arange(0, ROPE_P2)
            mask_r = offs_r < ROPE
            v_r = tl.load(
                kr_ptr + row * kr_stride + offs_r, mask=mask_r, other=0.0
            )
            tl.store(
                out_ptr + row * head + NOPE + offs_r,
                v_r.to(out_ptr.dtype.element_ty),
                mask=mask_r,
            )
    else:
        # Go through fp32 so the zero is exact for any fp8 flavour.
        zero_n = tl.zeros([NOPE_P2], dtype=tl.float32).to(
            out_ptr.dtype.element_ty
        )
        tl.store(out_ptr + row * head + offs_n, zero_n, mask=mask_n)
        if ROPE > 0:
            offs_r = tl.arange(0, ROPE_P2)
            mask_r = offs_r < ROPE
            zero_r = tl.zeros([ROPE_P2], dtype=tl.float32).to(
                out_ptr.dtype.element_ty
            )
            tl.store(out_ptr + row * head + NOPE + offs_r, zero_r, mask=mask_r)


def concat_cast_kv_fp8_pad(
    out: torch.Tensor,
    k: torch.Tensor,
    k_rope: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Cast-concat ``k`` (NoPE latent) and ``k_rope`` into an fp8 KV buffer.

    Args:
        out: ``[total_rows, NOPE + ROPE]`` ``float8_e4m3fn`` destination slice,
            where ``total_rows = num_tokens + pad_rows``. Rows past
            ``num_tokens`` are zeroed. Contents before the call are irrelevant.
        k: ``[num_tokens, NOPE]`` (or any shape with that view) bf16 NoPE latent.
        k_rope: ``[num_tokens, ROPE]`` bf16 rope segment; ``ROPE`` may be 0.
        num_tokens: Number of real rows; the rest of ``out`` is the pad band.

    Returns:
        ``out``, written in place.

    The last-dimension stride of both sources must be 1: the kernel addresses
    them as ``base + row * stride(0) + arange(width)``, so a non-unit inner
    stride would silently read the wrong elements.
    """
    if k.dim() != 2 and k.dim() != 3:
        raise ValueError(f"k must be 2-D or 3-D, got {tuple(k.shape)}")
    if k_rope.dim() != 2 and k_rope.dim() != 3:
        raise ValueError(
            f"k_rope must be 2-D or 3-D, got {tuple(k_rope.shape)}"
        )

    if out.dim() != 2:
        raise ValueError(f"out must be 2-D, got {tuple(out.shape)}")
    if out.dtype != torch.float8_e4m3fn:
        raise ValueError(f"out must be float8_e4m3fn, got {out.dtype}")

    nope = k.shape[-1]
    rope = k_rope.shape[-1] if k_rope.numel() else 0
    head = nope + rope
    if out.shape[-1] != head:
        raise ValueError(f"out width {out.shape[-1]} != nope + rope = {head}")
    total_rows = out.shape[0]
    if num_tokens > total_rows:
        raise ValueError(
            f"num_tokens {num_tokens} exceeds total rows {total_rows}"
        )
    if head == 0:
        return out

    if k.stride(-1) != 1:
        raise ValueError("k must have a unit last-dimension stride")
    if rope > 0 and k_rope.stride(-1) != 1:
        raise ValueError("k_rope must have a unit last-dimension stride")

    k2 = k.contiguous().view(num_tokens, nope)
    kr2 = (
        k_rope.contiguous().view(num_tokens, rope)
        if rope > 0
        else k_rope.new_empty((num_tokens, 0))
    )

    nope_p2 = triton.next_power_of_2(max(nope, 1))
    rope_p2 = triton.next_power_of_2(max(rope, 1))

    _concat_cast_kv_fp8_pad_kernel[(total_rows,)](
        out,
        k2,
        kr2,
        num_tokens,
        k2.stride(0),
        kr2.stride(0),
        NOPE=nope,
        ROPE=rope,
        NOPE_P2=nope_p2,
        ROPE_P2=rope_p2,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["concat_cast_kv_fp8_pad"]
