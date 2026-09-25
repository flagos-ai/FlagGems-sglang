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

"""compute_src2dst (routing-permutation inverse scatter) -- MetaX specialization.
"""

import functools

import torch
import triton
import triton.language as tl
from triton.runtime import driver as _driver

_LOG_BUCKET = 8  # elements per bucket (256 int64 = 2KB output region)
_BUCKET = 1 << _LOG_BUCKET
_LARGE_THRESH = 262144  # use the bucketed pipeline above this


@triton.jit(do_not_specialize=["n"])
def _src2dst_kernel(reorder_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(reorder_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + idx, offs.to(tl.int32), mask=mask)


@functools.lru_cache(maxsize=None)
def _compiled_kernel(in_dtype, dev, block, warps):
    """Compile (once) via the standard warmup API and return the handle."""
    return _src2dst_kernel.warmup(
        in_dtype,
        torch.int32,
        1,
        BLOCK=block,
        num_warps=warps,
        num_stages=1,
        grid=(1,),
    )


# ---- large-n bucketed pipeline -------------------------------------------


@triton.jit(do_not_specialize=["n"])
def _bucket_pass_a(
    reorder_ptr,
    tmp_ptr,
    cnt_ptr,
    out_ptr,
    n,
    ORDER: tl.constexpr,
    LOG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Sequential read; append (dst, d) into dst's bucket; tail written direct.

    ``ORDER`` is the largest power of two <= n, i.e. exactly the span covered
    by the ``ORDER >> LOG`` buckets.  Destinations at or above it are the tail
    of the permutation (never produced by a ``randperm``-style reorder_ids) and
    are stored straight through, which also makes non-power-of-two n correct.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(reorder_ptr + offs, mask=mask, other=0)
    bucketed = mask & (idx < ORDER)
    tail = mask & (idx >= ORDER)
    tl.store(out_ptr + idx, offs.to(tl.int32), mask=tail)
    b = idx >> LOG
    # relaxed: each element only claims a private slot; pass B runs in a
    # separate launch, so no ordering with respect to other memory ops matters
    pos = tl.atomic_add(cnt_ptr + b, 1, mask=bucketed, sem="relaxed")
    packed = (idx.to(tl.int64) << 32) | (offs.to(tl.int64) & 0xFFFFFFFF)
    tl.store(tmp_ptr + (b << LOG) + pos, packed, mask=bucketed)


@triton.jit
def _bucket_pass_b(
    tmp_ptr, out_ptr, cnt_ptr, LOG: tl.constexpr, BLOCK: tl.constexpr
):
    b = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    c = tl.load(cnt_ptr + b)
    mask = offs < c
    packed = tl.load(tmp_ptr + (b << LOG) + offs, mask=mask, other=0)
    idx = (packed >> 32).to(tl.int32)
    d = (packed & 0xFFFFFFFF).to(tl.int32)
    tl.store(out_ptr + idx, d, mask=mask)


@triton.jit
def _bucket_zero(cnt_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(cnt_ptr + offs, 0, mask=offs < n)


def _pick_cfg(num_toks):
    # Deterministic size heuristic (head-to-head do_bench sweeps):
    # tiny blocks minimize launch floor latency, wider blocks / more warps
    # win once the scatter's write-allocate traffic is throughput-bound
    # (more concurrent programs issue more independent store streams).
    # The mid band (8192 < n <= 262144) sits just above the ~11us launch
    # floor, where 2 warps per 128-element tile beat 4 warps: at this size
    # the random stores are still L2-resident, so extra warps only add
    # sub-tile serialization without buying store throughput.
    if num_toks > 8192:
        return 128, 2
    return 64, 1


def _large_src2dst(reorder_ids, src2dst, num_toks):
    # Bucketed two-pass pipeline: localize the random stores into small
    # L2-resident buckets so the 4B scatter stops dominating.
    n = num_toks
    order = 1 << (n.bit_length() - 1)  # largest power of two <= n
    nb = order >> _LOG_BUCKET
    tmp = torch.empty(order, dtype=torch.int64, device=src2dst.device)
    cnt = torch.empty(nb, dtype=torch.int32, device=src2dst.device)
    _bucket_zero[(triton.cdiv(nb, 256),)](cnt, nb, BLOCK=256)
    _bucket_pass_a[(triton.cdiv(n, 256),)](
        reorder_ids,
        tmp,
        cnt,
        src2dst,
        n,
        ORDER=order,
        LOG=_LOG_BUCKET,
        BLOCK=256,
        num_warps=4,
    )
    _bucket_pass_b[(nb,)](
        tmp, src2dst, cnt, LOG=_LOG_BUCKET, BLOCK=_BUCKET, num_warps=2
    )
    return src2dst


def compute_src2dst(reorder_ids, num_toks):
    src2dst = torch.empty_like(reorder_ids, dtype=torch.int32)
    if num_toks <= 0:
        return src2dst
    if num_toks > _LARGE_THRESH:
        return _large_src2dst(reorder_ids, src2dst, num_toks)
    block, warps = _pick_cfg(num_toks)
    device = _driver.active.get_current_device()
    ck = _compiled_kernel(reorder_ids.dtype, device, block, warps)
    # Fast relaunch: re-issue the same low-level call JITFunction.run makes,
    # skipping its per-call arg-binding/cache-key overhead (the dominant
    # cost for small problems). launch_metadata/hooks are None exactly as
    # when no profiling hook is installed.
    stream = _driver.active.get_current_stream(device)
    ck.run(
        triton.cdiv(num_toks, block),
        1,
        1,
        stream,
        ck.function,
        ck.packed_metadata,
        None,
        None,
        None,
        reorder_ids.data_ptr(),
        src2dst.data_ptr(),
        num_toks,
        block,
    )
    return src2dst


__all__ = ["compute_src2dst"]
