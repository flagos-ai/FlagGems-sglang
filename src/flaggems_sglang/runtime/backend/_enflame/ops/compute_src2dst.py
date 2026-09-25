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


def _launch(reorder_words, src2dst, num_toks, id_stride):
    if num_toks <= 4096:
        # One program; size the block so masked-out lanes stay minimal. A
        # wider warp count wins here: the whole block fits in flight and the
        # extra lanes hide store latency (measured: n=512 w=4 vs w=1 ~ -30%).
        block = triton.next_power_of_2(max(num_toks, 8))
        num_warps = 2 if block <= 16 else (4 if block <= 2048 else 8)
        _compute_src2dst_kernel[(1,)](
            reorder_words,
            src2dst,
            num_toks,
            ID_STRIDE=id_stride,
            BLOCK=block,
            NPROG=1,
            EXACT=(num_toks == block),
            num_warps=num_warps,
        )
    else:
        # Persistent strided sweep: 64 programs up to 8192 elements (one loop
        # iteration each), then 256 — the flat grid is capped so launch cost
        # stops growing with n. Traversal order matches the flat grid, so
        # scatter locality is unchanged (measured: 131072 -1.6%,
        # 1048576 -0.1..-0.5% vs flat grid; NPROG=128 loses ~2% at 131072,
        # NPROG=512 ~0.2%). EXACT fires when the tail is full (every pow2 /
        # multiple of 32768, and all of 8192 with NPROG=64): bounds masks on
        # the load and the scattered store are pure overhead then (~1% at
        # 1048576).
        nprog = 64 if num_toks <= 8192 else 256
        _compute_src2dst_kernel[(nprog,)](
            reorder_words,
            src2dst,
            num_toks,
            ID_STRIDE=id_stride,
            BLOCK=128,
            NPROG=nprog,
            EXACT=(num_toks % (nprog * 128) == 0),
            num_warps=1,
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
