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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""sgemm_lora_a: Triton kernel for the LoRA "A" (down-projection) segmented GEMM.

For each segment ``b`` (row range ``seg_indptr[b]:seg_indptr[b+1]`` routed to
adapter ``weight_indices[b]``) we compute ``out[rows] = x[rows] @ weights[w].T``
in float32 and cast back to ``x.dtype``. ``rows`` comes from ``permutation``
when present, otherwise is the contiguous range ``arange(start, end)``.

Performance notes (GCU / enflame backend, gcu410, measured 2026-11-08 with the
official bench protocol do_bench warmup 1000ms / rep 100ms, bf16):

- The kernel is a plain masked-tile GEMM loop; all tuning lives in the host
  dispatch (``_pick_config``) which replaces the previous autotune. Measured
  facts that drove the table (see perf_playbook / case log 2026-11-08):
  * ``num_warps=2`` is decisive: BM in {32, 64, 128} with nw2 keeps the MMA
    fast path; nw4 drops to a much slower path (e.g. 64x64x1024: 18.7us nw2
    vs 22.7us nw4; 64x32x1024: 15.4us nw2 vs ~63ms-era scalar fallback).
  * Bigger BLOCK_K wins: per-CTA streaming is the wall and the dot's SIP
    barrier serializes load vs dot inside the K-loop (num_stages does not
    enable software pipelining here), so fewer, larger K-iterations amortise
    better: BK 512 -> 1024 -> 2048 -> 4096 = 19.2 -> 15.1 -> 12.8 -> 11.2us
    on the seg64x8/R32/K4096 shape.
  * More CTAs do not add bandwidth once every CTA is streaming (8 vs 16 vs 32
    CTAs flat-to-worse), so BM is kept at 64 and the grid stays small.
  * DSM per program is exactly ``itemsize * BLOCK_K * (BLOCK_M + BLOCK_N)``
    (single-buffered; verified against compiled dsm_memory_size), so
    BLOCK_K is maximised under the 917504B hard limit by ``_pick_config``.
    That yields BK=4096 (whole-K single dot) for the R=32/K=4096 bench shape
    and BK=2048 for R=64/K=4096.
  * Measured: seg64x8/R32/K4096 13.0 -> 11.2us, seg256x4/R64/K4096
    20.3 -> 16.9us vs the previous autotune winners.
- Manual rolling-prefetch (double-buffer) variants were tried and vetoed: the
  extra live tiles double DSM (pushing configs past the 917504B limit or
  back to 24-way residency) and measured slower on both bench shapes
  (13.4-15.1us vs 11.2us case0; 18.7-26.5us vs 16.9us case1).
- Split-K stays rejected: the GCU backend has no float atomics, so split-K
  needs a [SPLIT, S, R] partial buffer + a second reduce kernel, breaking
  the single-launch contract for a launch-bound 11-20us op.
- Grouping segments by adapter was tried upstream and rejected: building the
  per-adapter row list in-kernel overflows local memory, and host grouping
  needs a ~109us device->host sync of seg_indptr/weight_indices.
- ``input_precision="ieee"`` matches the reference fp32 matmul semantics.
  (The default precision measured identical latency and passed the 1.5e-2
  bf16 gate on this stack, but ieee is kept as the zero-risk choice.)
- Indices stay int32 (GCU lacks int64; S*K fits int32 for the batch sizes
  this op serves).
"""

import torch
import triton
import triton.language as tl

# gcu410 hard per-program DSM limit (compiled dsm_memory_size must stay <=).
_MAX_DSM_BUDGET = 900_000


def _next_pow2(v):
    p = 1
    while p < v:
        p <<= 1
    return p


@triton.jit
def _sgemm_lora_a_kernel(
    x_ptr,  # [S, K]
    w_ptr,  # [num_lora, R, K]
    out_ptr,  # [S, R] in x.dtype
    seg_indptr_ptr,  # [bs+1] int32
    weight_indices_ptr,  # [bs] int32
    permutation_ptr,  # [S] int32 or None
    K,
    R,
    NM,  # M-tiles per segment (for the flattened 2D grid)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    # Flattened 2D grid (bs * NM, n_tiles): keeps grid.y small for arbitrary
    # max_len (the 3D form hits the grid.y<=255 limit at max_len>16320 for
    # BLOCK_M=64) and measured identical latency to the 3D grid.
    pid_b = tl.program_id(0) // NM
    pid_m = tl.program_id(0) % NM
    pid_n = tl.program_id(1)

    start = tl.load(seg_indptr_ptr + pid_b)
    end = tl.load(seg_indptr_ptr + pid_b + 1)
    seg_len = end - start

    m_off = pid_m * BLOCK_M
    if m_off >= seg_len:
        return

    w_idx = tl.load(weight_indices_ptr + pid_b)

    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_off < R

    m_idx = tl.arange(0, BLOCK_M)
    m_valid = m_idx < (seg_len - m_off)
    # segment-local position -> global row (int32; GCU lacks int64)
    if HAS_PERM:
        rows = tl.load(
            permutation_ptr + start + m_off + m_idx, mask=m_valid, other=0
        )
    else:
        rows = start + m_off + m_idx

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_base = tl.arange(0, BLOCK_K)

    for k0 in range(0, K, BLOCK_K):
        k_off = k0 + k_base
        k_mask = k_off < K

        # x[rows, k_off] -> [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + rows[:, None] * K + k_off[None, :]
        x_block = tl.load(
            x_ptrs, mask=m_valid[:, None] & k_mask[None, :], other=0.0
        )

        # w[w_idx, n_off, k_off] -> tile [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + w_idx * R * K + n_off[None, :] * K + k_off[:, None]
        w_block = tl.load(
            w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
        )

        acc += tl.dot(x_block, w_block, input_precision="ieee")

    # Direct store (no split-K scratch buffer): cast fp32 accumulator to the
    # output dtype. Each output element is written by exactly one program.
    out_ptrs = out_ptr + rows[:, None] * R + n_off[None, :]
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_valid[:, None] & n_mask[None, :],
    )


def _pick_config(max_len, R, K, itemsize):
    """Static dispatch (replaces autotune; measured 2026-11-08, gcu410).

    BM: 16..64, largest pow2 tile of the longest segment; BM=64 is the
    measured sweet spot for the bench shapes.
    BN: 16..64, covers R in one tile when possible.
    BK: largest pow2 that both divides into K-friendly bounds and keeps the
    per-program DSM (= itemsize * BK * (BM + BN)) under the 917504B limit;
    large BK amortises the K-loop best (fewest dot SIP barriers).
    num_warps=2 everywhere (MMA fast path); num_stages=2 (no effect while
    soft pipelining is blocked by the dot barrier, measured flat).
    """
    BM = max(16, min(64, _next_pow2(max_len if max_len > 0 else 16)))
    BN = max(16, min(64, _next_pow2(R)))

    bk_cap = _MAX_DSM_BUDGET // (itemsize * (BM + BN))
    BK = _next_pow2(min(K, bk_cap)) if K > 0 else 16
    if BK > bk_cap:
        BK >>= 1
    BK = max(16, min(BK, 4096))
    return BM, BN, BK


def sgemm_lora_a(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    R = weights.shape[1]
    # ``torch.empty`` instead of ``torch.zeros``: the kernel writes every
    # ``(row, n)`` element exactly once, so no element is left uninitialised
    # and the host-side memset (~12us on GCU) is skipped.
    out = torch.empty((S, R), dtype=x.dtype, device=x.device)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation
    has_perm = permutation is not None

    # max segment length (host) -> BLOCK_M / grid sizing. Prefer the
    # host-side ``batch_info.max_len`` int; fall back to a D2H sync only if
    # the attribute is missing.
    host_max_len = getattr(batch_info, "max_len", None)
    if host_max_len is None and S > 0:
        seg_lens = seg_indptr[1:] - seg_indptr[:-1]
        host_max_len = int(seg_lens.max().item())
    max_len = host_max_len if host_max_len is not None else 0

    bs = batch_info.bs

    if bs <= 0 or max_len <= 0 or S == 0:
        # nothing to compute (defensive; benchmark cases never hit this)
        out.zero_()
        return out

    BM, BN, BK = _pick_config(max_len, R, K, x.element_size())

    nm = (max_len + BM - 1) // BM
    nn = (R + BN - 1) // BN

    perm_ptr = (
        permutation if has_perm else x
    )  # dummy ptr, unused when HAS_PERM=False

    _sgemm_lora_a_kernel[(bs * nm, nn)](
        x,
        weights,
        out,
        seg_indptr,
        weight_indices,
        perm_ptr,
        K,
        R,
        nm,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        HAS_PERM=has_perm,
        num_warps=2,
        num_stages=2,
    )
    return out


__all__ = ["sgemm_lora_a"]
