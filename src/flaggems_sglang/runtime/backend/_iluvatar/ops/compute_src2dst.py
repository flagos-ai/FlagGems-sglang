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

"""compute_src2dst (routing-permutation inverse scatter) -- Iluvatar specialization.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # small n — launch overhead dominates; tiny blocks, few programs
        triton.Config({"BLOCK": 128, "LOOP": 1}, num_warps=1),
        triton.Config({"BLOCK": 128, "LOOP": 1}, num_warps=2),
        triton.Config({"BLOCK": 256, "LOOP": 1}, num_warps=2),
        triton.Config({"BLOCK": 256, "LOOP": 1}, num_warps=4),
        # mid n — a couple of chunks per program amortizes scheduling
        triton.Config({"BLOCK": 256, "LOOP": 2}, num_warps=4),
        triton.Config({"BLOCK": 512, "LOOP": 1}, num_warps=2),
        triton.Config({"BLOCK": 512, "LOOP": 1}, num_warps=4),
        triton.Config({"BLOCK": 512, "LOOP": 2}, num_warps=8),
        triton.Config({"BLOCK": 512, "LOOP": 4}, num_warps=8),
        triton.Config({"BLOCK": 1024, "LOOP": 1}, num_warps=8),
        triton.Config({"BLOCK": 1024, "LOOP": 2}, num_warps=8),
        triton.Config({"BLOCK": 1024, "LOOP": 4}, num_warps=4),
        triton.Config({"BLOCK": 1024, "LOOP": 8}, num_warps=4),
        # large n — wide waves of single-shot programs beat few persistent
        # ones (measured: LOOP=1 configs win at 1M elems on 16-SM parts)
        triton.Config({"BLOCK": 256, "LOOP": 4}, num_warps=8),
        triton.Config({"BLOCK": 2048, "LOOP": 2}, num_warps=4),
        triton.Config({"BLOCK": 2048, "LOOP": 2}, num_warps=8),
        triton.Config({"BLOCK": 2048, "LOOP": 4}, num_warps=4),
        triton.Config({"BLOCK": 2048, "LOOP": 4}, num_warps=8),
        triton.Config({"BLOCK": 4096, "LOOP": 2}, num_warps=8),
        triton.Config({"BLOCK": 4096, "LOOP": 4}, num_warps=8),
        triton.Config({"BLOCK": 8192, "LOOP": 2}, num_warps=8),
        triton.Config({"BLOCK": 8192, "LOOP": 4}, num_warps=8),
    ],
    key=["n"],
)
@triton.jit
def _src2dst_kernel(
    reorder_ids_ptr,  # int64 [n], reinterpreted below as int32 low words
    out_ptr,  # int32 [n]
    n,
    BLOCK: tl.constexpr,
    LOOP: tl.constexpr,
):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    # reorder_ids values are permutation indices < n <= 2^31, so the high
    # int32 word of each int64 element is zero; read only the low word. The
    # stream is consumed once — evict lines after use to spare L2 for the
    # scatter store set.
    src32_ptr = reorder_ids_ptr.to(tl.pointer_type(tl.int32))
    # Pass 1 (statically unrolled): issue every chunk's load before any store
    # so all load requests are in flight together and none is ordered behind
    # a scatter store (the compiler must assume out_ptr may alias
    # reorder_ids). LOOP is 1..8 — guarded constexpr branches below.
    offs0 = pid * BLOCK + tl.arange(0, BLOCK)
    mask0 = offs0 < n
    src0 = tl.load(
        src32_ptr + 2 * offs0,
        mask=mask0,
        other=0,
        eviction_policy="evict_first",
    )
    if LOOP >= 2:
        offs1 = (pid + npg) * BLOCK + tl.arange(0, BLOCK)
        mask1 = offs1 < n
        src1 = tl.load(
            src32_ptr + 2 * offs1,
            mask=mask1,
            other=0,
            eviction_policy="evict_first",
        )
    if LOOP >= 4:
        offs2 = (pid + 2 * npg) * BLOCK + tl.arange(0, BLOCK)
        mask2 = offs2 < n
        src2 = tl.load(
            src32_ptr + 2 * offs2,
            mask=mask2,
            other=0,
            eviction_policy="evict_first",
        )
        offs3 = (pid + 3 * npg) * BLOCK + tl.arange(0, BLOCK)
        mask3 = offs3 < n
        src3 = tl.load(
            src32_ptr + 2 * offs3,
            mask=mask3,
            other=0,
            eviction_policy="evict_first",
        )
    if LOOP >= 8:
        offs4 = (pid + 4 * npg) * BLOCK + tl.arange(0, BLOCK)
        mask4 = offs4 < n
        src4 = tl.load(
            src32_ptr + 2 * offs4,
            mask=mask4,
            other=0,
            eviction_policy="evict_first",
        )
        offs5 = (pid + 5 * npg) * BLOCK + tl.arange(0, BLOCK)
        mask5 = offs5 < n
        src5 = tl.load(
            src32_ptr + 2 * offs5,
            mask=mask5,
            other=0,
            eviction_policy="evict_first",
        )
        offs6 = (pid + 6 * npg) * BLOCK + tl.arange(0, BLOCK)
        mask6 = offs6 < n
        src6 = tl.load(
            src32_ptr + 2 * offs6,
            mask=mask6,
            other=0,
            eviction_policy="evict_first",
        )
        offs7 = (pid + 7 * npg) * BLOCK + tl.arange(0, BLOCK)
        mask7 = offs7 < n
        src7 = tl.load(
            src32_ptr + 2 * offs7,
            mask=mask7,
            other=0,
            eviction_policy="evict_first",
        )
    # Pass 2: scatter dst = d to out[src]; values are a permutation of
    # [0, n) so no extra range predicate is needed beyond the tail mask.
    tl.store(out_ptr + src0, offs0.to(tl.int32), mask=mask0)
    if LOOP >= 2:
        tl.store(out_ptr + src1, offs1.to(tl.int32), mask=mask1)
    if LOOP >= 4:
        tl.store(out_ptr + src2, offs2.to(tl.int32), mask=mask2)
        tl.store(out_ptr + src3, offs3.to(tl.int32), mask=mask3)
    if LOOP >= 8:
        tl.store(out_ptr + src4, offs4.to(tl.int32), mask=mask4)
        tl.store(out_ptr + src5, offs5.to(tl.int32), mask=mask5)
        tl.store(out_ptr + src6, offs6.to(tl.int32), mask=mask6)
        tl.store(out_ptr + src7, offs7.to(tl.int32), mask=mask7)


def compute_src2dst(reorder_ids, num_toks):
    src2dst = torch.empty(
        num_toks, dtype=torch.int32, device=reorder_ids.device
    )
    n = num_toks
    # grid follows the (BLOCK, LOOP) the autotuner picks so coverage of n
    # always holds: cdiv(n, BLOCK*LOOP) programs x LOOP chunks of BLOCK
    _src2dst_kernel[
        (lambda META: (triton.cdiv(n, META["BLOCK"] * META["LOOP"]),))
    ](
        reorder_ids,
        src2dst,
        n,
    )
    return src2dst


__all__ = ["compute_src2dst"]
