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

"""Get target/free cache locations for speculative verification (Enflame GCU).

GCU adaptation notes:

- torch_gcu stores int64 tensors in 4-byte physical slots (metadata still
  reports 8B) unless ``TORCH_GCU_ENABLE_INT64_AND_UINT64=1``. Indexing those
  buffers through int64 pointers strides 8B over 4B slots, which scrambles
  loads and overruns stores (the pre-fix baseline: wrong values for small
  shapes, device abort at the next sync for large ones). Values in this op
  fit int32 by contract (draft/free counts < num_verify_tokens, cache locs
  < 2^20), so the previous fix normalised every tensor with host-side
  ``.to(torch.int32)`` casts — correct, but five tiny cast kernels whose
  launch overhead dominated the op (bench kernel-mode ~92us vs ~32-40us for
  the native generic path).
- This version keeps a single kernel launch and removes all cast kernels:
  with the int64 env off, the kernel reinterprets the int64 buffer pointers
  as int32 pointers (pointer bitcast; the physical slots are int32, and all
  tensors entering the fast path are contiguous with storage_offset 0, so
  ``data_ptr`` equals the first physical slot). With the env on the same
  tensors are true 8-byte int64 and the kernel handles them natively.
- The kernel writes target rows into ``tgt_buf`` and free rows into
  ``free_buf``; the two row totals (n_tgt, n_free) go into a small int32
  scratch tensor. The host reads the totals through one 2-element D2H copy
  instead of staging the input counts (2 copies + ``tolist`` over 2*bs
  values + Python sums). Returned tensors are 0-offset prefix slices of the
  two buffers: ATen reads them as ordinary (fake-)int64 tensors and
  ``data_ptr`` still points at the first physical slot, so downstream
  Triton pointer-cast consumers stay correct. The int32 counts channel also
  matters for speed: copying from a (fake-)int64 slice takes a
  layout-converting slow path (~+12us on the bench meter), while a real
  int32 source is a plain memcpy.
"""

import os

import torch
import triton
import triton.language as tl


# Single-CTA scatter. One program handles the whole batch: a BLOCK-wide
# arange covers every row, so the target/free offsets come straight from a
# single cumsum with no cross-CTA prefix to stitch together. This is the only
# path: a former multi-CTA `_kernel` tripped a FlagTree layout-propagation
# crash (RemoveLayoutConversions cloneElementwise, `operandEnc` assertion) for
# every bs > 64 when broadcasting the cumsum-derived row offset into the 2D
# store pointer, so it could never actually compile.
@triton.jit
def _kernel_small(
    out_ptr,
    drafts_ptr,
    frees_ptr,
    tgt_ptr,
    free_ptr,
    counts_ptr,
    bs,
    NV: tl.constexpr,
    NVP: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKED: tl.constexpr,
):
    if PACKED:
        # int64 tensors are physically packed int32 slots (no env): same
        # address, 4-byte elements. All fast-path buffers are offset-0.
        out_ptr = out_ptr.to(tl.pointer_type(tl.int32))
        drafts_ptr = drafts_ptr.to(tl.pointer_type(tl.int32))
        frees_ptr = frees_ptr.to(tl.pointer_type(tl.int32))
        tgt_ptr = tgt_ptr.to(tl.pointer_type(tl.int32))
        free_ptr = free_ptr.to(tl.pointer_type(tl.int32))
    offs = tl.arange(0, BLOCK)
    m = offs < bs
    nb = tl.load(drafts_ptr + offs, mask=m, other=0).to(tl.int32)
    fb = tl.load(frees_ptr + offs, mask=m, other=0).to(tl.int32)
    tot_n = tl.sum(nb, axis=0)
    tot_f = tl.sum(fb, axis=0)
    cty = counts_ptr.dtype.element_ty
    tl.store(counts_ptr, (tot_n + bs).to(cty))
    tl.store(counts_ptr + 1, tot_f.to(cty))
    if tot_f == 0:
        zty = free_ptr.dtype.element_ty
        tl.store(free_ptr, (tot_n - tot_n).to(zty))
    t_off = tl.cumsum(nb, axis=0) - nb + offs
    f_off = tl.cumsum(fb, axis=0) - fb

    cols = tl.arange(0, NVP)[None, :]
    base = offs[:, None] * NV
    rm2 = m[:, None] & (cols < NV)
    row = tl.load(out_ptr + base + cols, mask=rm2, other=0)

    keep = rm2 & (cols < (nb + 1)[:, None])
    tl.store(tgt_ptr + t_off[:, None] + cols, row, mask=keep)

    fd = f_off - NV + fb
    fmask = rm2 & (cols >= (NV - fb)[:, None])
    tl.store(free_ptr + fd[:, None] + cols, row, mask=fmask)


def _pow2_at_least(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _gcu_int64_enabled():
    return os.environ.get(
        "TORCH_GCU_ENABLE_INT64_AND_UINT64", ""
    ).lower() not in (
        "",
        "0",
        "false",
        "off",
        "no",
    )


# Per-device internal scratch for the 2-element counts channel: a device
# int32 tensor the kernel writes and a pinned int32 staging buffer the host
# reads after the stream sync. Both are overwritten at the start of every
# call and read only after the sync, so single-threaded callers (sglang
# scheduler per rank, benchmarks) are safe.
_SCRATCH = {}


def _scratch(dev):
    scratch = _SCRATCH.get(dev)
    if scratch is None:
        scratch = (
            torch.empty(2, dtype=torch.int32, device=dev),
            torch.empty(2, dtype=torch.int32, pin_memory=True),
        )
        _SCRATCH[dev] = scratch
    return scratch


def _run_packed(drafts, frees, out, bs, nv, packed):
    """Single-kernel path over same-dtype contiguous offset-0 tensors."""
    dtype = out.dtype
    dev = out.device
    cap = bs * nv
    tgt_buf = torch.empty(cap, dtype=dtype, device=dev)
    free_buf = torch.empty(cap, dtype=dtype, device=dev)
    counts, counts_host = _scratch(dev)

    _kernel_small[(1,)](
        out,
        drafts,
        frees,
        tgt_buf,
        free_buf,
        counts,
        bs,
        NV=nv,
        NVP=_pow2_at_least(nv),
        BLOCK=_pow2_at_least(bs),
        PACKED=packed,
        num_warps=1,
    )

    counts_host.copy_(counts, non_blocking=True)
    counts_ready = torch.cuda.Event()
    counts_ready.record()
    counts_ready.synchronize()
    n_tgt, n_free = counts_host.tolist()

    tgt = tgt_buf[:n_tgt]
    free = free_buf[:n_free] if n_free else free_buf[:1]
    return tgt, free


def get_target_cache_loc(
    num_correct_drafts, to_free_num_slots, out_cache_loc, num_verify_tokens
):
    bs = num_correct_drafts.shape[0]
    dtype = out_cache_loc.dtype
    dev = out_cache_loc.device
    if bs == 0:
        z = torch.zeros(1, dtype=dtype, device=dev)
        return z, z.clone()

    if (
        dtype in (torch.int64, torch.int32)
        and num_correct_drafts.dtype == dtype
        and to_free_num_slots.dtype == dtype
        and num_correct_drafts.is_contiguous()
        and num_correct_drafts.storage_offset() == 0
        and to_free_num_slots.is_contiguous()
        and to_free_num_slots.storage_offset() == 0
        and out_cache_loc.is_contiguous()
        and out_cache_loc.storage_offset() == 0
    ):
        packed = dtype == torch.int64 and not _gcu_int64_enabled()
        return _run_packed(
            num_correct_drafts,
            to_free_num_slots,
            out_cache_loc,
            bs,
            int(num_verify_tokens),
            packed,
        )

    # Rare shapes (mixed dtypes, narrow ints, sliced/strided inputs): keep the
    # value-preserving int32 staging of the original fix and cast back.
    drafts = num_correct_drafts.contiguous()
    if drafts.dtype != torch.int32:
        drafts = drafts.to(torch.int32)
    frees = to_free_num_slots.contiguous()
    if frees.dtype != torch.int32:
        frees = frees.to(torch.int32)
    out = out_cache_loc.contiguous()
    if out.dtype != torch.int32:
        out = out.to(torch.int32)
    tgt, free = _run_packed(
        drafts, frees, out, bs, int(num_verify_tokens), False
    )
    return tgt.to(dtype), free.to(dtype)


__all__ = ["get_target_cache_loc"]
