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

import torch
import triton
import triton.language as tl

_SMALL_CHUNK = 16

_LIST_SUM_MAX_BS = 256

_SINGLE_CTA_MAX_BS = 64


@triton.jit
def _kernel_small(
    out_ptr,
    drafts_ptr,
    frees_ptr,
    buf_ptr,
    bs,
    NV: tl.constexpr,
    NVP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    m = offs < bs
    nb = tl.load(drafts_ptr + offs, mask=m, other=0).to(tl.int32)
    fb = tl.load(frees_ptr + offs, mask=m, other=0).to(tl.int32)
    tot_n = tl.sum(nb, axis=0)
    tot_f = tl.sum(fb, axis=0)
    elem_ty = buf_ptr.dtype.element_ty
    tl.store(buf_ptr, (tot_n + bs).to(elem_ty))
    tl.store(buf_ptr + 1, tot_f.to(elem_ty))
    if tot_f == 0:
        tl.store(buf_ptr + 2 + bs * NV, (tot_n - tot_n).to(elem_ty))
    t_off = tl.cumsum(nb, axis=0) - nb + offs
    f_off = tl.cumsum(fb, axis=0) - fb

    cols = tl.arange(0, NVP)[None, :]
    base = offs[:, None] * NV
    rm2 = m[:, None] & (cols < NV)
    row = tl.load(out_ptr + base + cols, mask=rm2, other=0)

    keep = rm2 & (cols < (nb + 1)[:, None])
    tl.store(buf_ptr + 2 + t_off[:, None] + cols, row, mask=keep)

    fd = f_off - NV + fb
    fmask = rm2 & (cols >= (NV - fb)[:, None])
    tl.store(buf_ptr + 2 + bs * NV + fd[:, None] + cols, row, mask=fmask)


@triton.jit
def _kernel(
    out_ptr,
    drafts_ptr,
    frees_ptr,
    buf_ptr,
    bs,
    NV: tl.constexpr,
    NVP: tl.constexpr,
    BLOCK: tl.constexpr,
    CH: tl.constexpr,
):
    pid = tl.program_id(0)
    r0 = pid * CH

    offs = tl.arange(0, BLOCK)
    m = offs < bs
    n = tl.load(drafts_ptr + offs, mask=m, other=0).to(tl.int32)
    f = tl.load(frees_ptr + offs, mask=m, other=0).to(tl.int32)
    tot_n = tl.sum(n, axis=0)
    tot_f = tl.sum(f, axis=0)
    if pid == 0:
        elem_ty = buf_ptr.dtype.element_ty
        tl.store(buf_ptr, (tot_n + bs).to(elem_ty))
        tl.store(buf_ptr + 1, tot_f.to(elem_ty))
        if tot_f == 0:
            tl.store(buf_ptr + 2 + bs * NV, (tot_n - tot_n).to(elem_ty))
    head_n = tl.sum(tl.where(offs < r0, n, 0), axis=0)
    head_f = tl.sum(tl.where(offs < r0, f, 0), axis=0)

    rows = r0 + tl.arange(0, CH)
    rm = rows < bs
    nb = tl.load(drafts_ptr + rows, mask=rm, other=0).to(tl.int32)
    fb = tl.load(frees_ptr + rows, mask=rm, other=0).to(tl.int32)
    t_off = head_n + tl.cumsum(nb, axis=0) - nb + rows
    f_off = head_f + tl.cumsum(fb, axis=0) - fb

    cols = tl.arange(0, NVP)[None, :]
    base = rows[:, None] * NV
    rm2 = rm[:, None] & (cols < NV)
    row = tl.load(out_ptr + base + cols, mask=rm2, other=0)

    keep = rm2 & (cols < (nb + 1)[:, None])
    tl.store(buf_ptr + 2 + t_off[:, None] + cols, row, mask=keep)

    fd = f_off - NV + fb
    fmask = rm2 & (cols >= (NV - fb)[:, None])
    tl.store(buf_ptr + 2 + bs * NV + fd[:, None] + cols, row, mask=fmask)


def _pow2_at_least(n):
    return 1 << (n - 1).bit_length() if n > 1 else 1


def get_target_cache_loc(
    num_correct_drafts, to_free_num_slots, out_cache_loc, num_verify_tokens
):
    bs = num_correct_drafts.shape[0]
    dtype = out_cache_loc.dtype
    dev = out_cache_loc.device
    if bs == 0:
        z = torch.zeros(1, dtype=dtype, device=dev)
        return z, z.clone()

    drafts = (
        num_correct_drafts
        if num_correct_drafts.is_contiguous()
        else num_correct_drafts.contiguous()
    )
    frees = (
        to_free_num_slots
        if to_free_num_slots.is_contiguous()
        else to_free_num_slots.contiguous()
    )
    out = (
        out_cache_loc
        if out_cache_loc.is_contiguous()
        else out_cache_loc.contiguous()
    )
    nv = int(num_verify_tokens)
    nvp = _pow2_at_least(nv)

    cnt = torch.empty(2 * bs, dtype=drafts.dtype, pin_memory=True)
    cnt[:bs].copy_(drafts, non_blocking=True)
    cnt[bs:].copy_(frees, non_blocking=True)
    counts_ready = torch.cuda.Event()
    counts_ready.record()

    buf = torch.empty(2 + 2 * bs * nv, dtype=dtype, device=dev)

    if bs <= _SINGLE_CTA_MAX_BS:
        block = _pow2_at_least(bs)
        _kernel_small[(1,)](
            out,
            drafts,
            frees,
            buf,
            bs,
            NV=nv,
            NVP=nvp,
            BLOCK=block,
            num_warps=1,
        )
    else:
        block = _pow2_at_least(bs)
        ch = _SMALL_CHUNK if bs <= 256 else _SINGLE_CTA_MAX_BS
        _kernel[((bs + ch - 1) // ch,)](
            out,
            drafts,
            frees,
            buf,
            bs,
            NV=nv,
            NVP=nvp,
            BLOCK=block,
            CH=ch,
            num_warps=1 if bs <= 256 else 8,
        )

    counts_ready.synchronize()
    if bs <= _LIST_SUM_MAX_BS:
        flat = cnt.tolist()
        n_tgt = sum(flat[:bs]) + bs
        n_free = sum(flat[bs:])
    else:
        arr = cnt.numpy()
        n_tgt = int(arr[:bs].sum()) + bs
        n_free = int(arr[bs:].sum())

    free0 = 2 + bs * nv
    tgt = buf[2 : 2 + n_tgt]
    free = buf[free0 : free0 + n_free] if n_free else buf[free0 : free0 + 1]
    return tgt, free


__all__ = ["get_target_cache_loc"]
