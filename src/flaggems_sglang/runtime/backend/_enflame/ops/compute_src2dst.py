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

"""compute_src2dst (routing-permutation inverse scatter) -- Enflame GCU specialization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_src2dst_kernel(
    reorder_ptr,  # int32 ids, or int32 word view of int64 ids
    out_ptr,  # int32 src2dst output
    n,  # number of elements
    ID_STRIDE: tl.constexpr,  # word stride of one id (1 packed / 2 little-endian)
    BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
    EXACT: tl.constexpr,  # n % (NPROG*BLOCK) == 0: drop bounds masks
):
    pid = tl.program_id(0)
    num_iters = tl.cdiv(n, BLOCK * NPROG)
    for i in range(num_iters):
        dst = (i * NPROG + pid) * BLOCK + tl.arange(0, BLOCK)
        if EXACT:
            src = tl.load(reorder_ptr + ID_STRIDE * dst)
            tl.store(out_ptr + src, dst.to(tl.int32))
        else:
            mask = dst < n
            src = tl.load(reorder_ptr + ID_STRIDE * dst, mask=mask, other=0)
            tl.store(out_ptr + src, dst.to(tl.int32), mask=mask)


@triton.jit
def _compute_src2dst_flat_kernel(
    reorder_ptr,
    out_ptr,
    n,
    ID_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,  # n % BLOCK == 0: drop bounds masks
):
    dst = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EXACT:
        src = tl.load(reorder_ptr + ID_STRIDE * dst)
        tl.store(out_ptr + src, dst.to(tl.int32))
    else:
        mask = dst < n
        src = tl.load(reorder_ptr + ID_STRIDE * dst, mask=mask, other=0)
        tl.store(out_ptr + src, dst.to(tl.int32), mask=mask)


# Flat-grid block ladder (L600 measurements, do_bench kernel mode): the
# sweet-spot block grows with n so the grid stays in the ~4-32 program band;
# wide vector ops (>=512 lanes) with num_warps=4 unlock the memory throughput
# that the old 128-lane/nw=1 sweep left on the table (22.5 -> 9.5 us @131072).
def _flat_block(n):
    if n <= 2048:
        return 128
    if n <= 4096:
        return 256
    if n <= 16384:
        return 512
    if n <= 32768:
        return 2048
    return 4096


def _launch(reorder_words, src2dst, num_toks, id_stride):
    n = num_toks
    if n <= 128:
        # One program covering the whole input; launch floor dominates here
        # (measured 3.09 us for every config up to n=128).
        block = triton.next_power_of_2(max(n, 8))
        _compute_src2dst_flat_kernel[(1,)](
            reorder_words,
            src2dst,
            n,
            ID_STRIDE=id_stride,
            BLOCK=block,
            EXACT=(n == block),
            num_warps=4 if block >= 128 else 2,
        )
        return
    if n <= 65536:
        block = _flat_block(n)
        _compute_src2dst_flat_kernel[(triton.cdiv(n, block),)](
            reorder_words,
            src2dst,
            n,
            ID_STRIDE=id_stride,
            BLOCK=block,
            EXACT=(n % block == 0),
            num_warps=4,
        )
        return
    if n <= 131072:
        # 24-program sweep of 1024-wide blocks: beats wider persistent forms
        # in this band (probe 131072: 8.91 vs pers16xB8192 9.53 us) and keeps
        # the grid at the scatter saturation point (24 programs).
        _compute_src2dst_kernel[(24,)](
            reorder_words,
            src2dst,
            n,
            ID_STRIDE=id_stride,
            BLOCK=1024,
            NPROG=24,
            EXACT=(n % (24 * 1024) == 0),
            num_warps=2,
        )
        return
    if n < 524288:
        # 24-program sweep of 2048-wide blocks (probe 262144: 14.83 vs
        # pers16xB8192 15.83 us).
        _compute_src2dst_kernel[(24,)](
            reorder_words,
            src2dst,
            n,
            ID_STRIDE=id_stride,
            BLOCK=2048,
            NPROG=24,
            EXACT=(n % (24 * 2048) == 0),
            num_warps=2,
        )
        return
    # Throughput regime: flat 4096-wide blocks win again above ~512k
    # (1048576: 49.3 vs pers24xB2048 50.3 us). Guard the grid.x hardware cap
    # by falling back to the persistent form for absurdly large n.
    if triton.cdiv(n, 4096) <= 65535:
        _compute_src2dst_flat_kernel[(triton.cdiv(n, 4096),)](
            reorder_words,
            src2dst,
            n,
            ID_STRIDE=id_stride,
            BLOCK=4096,
            EXACT=(n % 4096 == 0),
            num_warps=4,
        )
    else:
        _compute_src2dst_kernel[(16,)](
            reorder_words,
            src2dst,
            n,
            ID_STRIDE=id_stride,
            BLOCK=8192,
            NPROG=16,
            EXACT=False,
            num_warps=4,
        )


def _probe_i64_layout(device) -> int:
    """Return the int32 word stride of an int64 element on this backend.

    Probes by writing a known small value through the int64 tensor and reading
    the int32 view: true 64-bit little-endian storage puts the value in word
    0 and zero in word 1 (stride 2); packed-32-bit storage puts consecutive
    values in consecutive words (stride 1).
    """
    probe = torch.zeros(4, dtype=torch.int64, device=device)
    probe[2] = 7
    words = probe.view(torch.int32).tolist()
    if words[4] == 7 and words[5] == 0:
        return 2
    if words[2] == 7:
        return 1
    # Unexpected layout; assume the standard little-endian one.
    return 2


_I64_WORD_STRIDE = None  # int or None until first int64 call (plain flag)


def compute_src2dst(reorder_ids, num_toks):
    src2dst = torch.empty(
        num_toks, dtype=torch.int32, device=reorder_ids.device
    )
    if num_toks == 0:
        return src2dst
    if reorder_ids.dtype == torch.int64:
        # Read ids via the int32 word view (see module docstring); some
        # backends reject 64-bit loads outright.
        global _I64_WORD_STRIDE
        if _I64_WORD_STRIDE is None:
            _I64_WORD_STRIDE = _probe_i64_layout(reorder_ids.device)
        reorder_words = reorder_ids.view(torch.int32)
        id_stride = _I64_WORD_STRIDE
    else:
        # int32 (or narrower) ids already: backends without int64 support
        # hand out int32 tensors directly.
        reorder_words = reorder_ids
        id_stride = 1
    _launch(reorder_words, src2dst, num_toks, id_stride)
    return src2dst


__all__ = ["compute_src2dst"]
