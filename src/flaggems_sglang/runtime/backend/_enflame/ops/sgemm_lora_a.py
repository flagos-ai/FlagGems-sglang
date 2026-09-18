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

"""sgemm_lora_a: Triton kernel for the LoRA "A" (down-projection) segmented GEMM.

For each segment ``b`` (row range ``seg_indptr[b]:seg_indptr[b+1]`` routed to
adapter ``weight_indices[b]``) we compute ``out[rows] = x[rows] @ weights[w].T``
in float32 and cast back to ``x.dtype``. ``rows`` comes from ``permutation``
when present, otherwise is the contiguous range ``arange(start, end)``.

Performance notes (GCU / enflame backend):
- The benchmark shapes have very few segments (4-8), small ``R`` and K=4096.
  Each segment is tiled over M (``BLOCK_M``) and N (``BLOCK_N``), so the grid is
  ``(bs, num_m_blocks, num_n_blocks)``. The device has only 2 SMs, so 4-8
  programs means 2-4 serial waves per SM -- the op is launch/wave-bound, not
  compute-bound, so the win is maximising per-program work by pushing
  ``BLOCK_K`` up to 1024-2048 (K=4096 -> 2-4 K iterations) so the K-loop
  amortises the launch + prologue cost.
  - ``R=32`` (seg64x8, each segment exactly 64 rows): autotune picks
    ``BLOCK_M=64, BLOCK_N=32, BLOCK_K=1024, num_warps=2``. A 64-row segment
    exactly fills a ``BLOCK_M=64`` tile (no padding, no wasted MMA width),
    whereas ``BLOCK_M=128`` always pads 64->128 and wastes half the tile.
  - ``R=64`` (seg256x4, each segment exactly 256 rows): autotune picks
    ``BLOCK_M=256, BLOCK_N=64, BLOCK_K=1024, num_warps=4`` -- one full M-tile
    per segment, 4 programs, 2 waves per SM.
- The "BLOCK_M>=128 MMA floor" rule has an important nuance: a ``BLOCK_M<128``
  tile only escapes GCU's slow scalar MMA path when paired with FEW warps.
  ``BLOCK_M=64`` with ``num_warps=4`` takes ~63 ms (scalar fallback) but with
  ``num_warps=2`` runs ~48 us -- faster than any padded ``BLOCK_M=128`` config.
  So the R=32 family uses ``BLOCK_M=64`` exclusively at ``num_warps in {1, 2}``.
- v8 widened the sweep to also try ``num_warps in {4, 8}`` and
  ``num_stages in {2, 3, 4}`` on the ``BLOCK_M>=128`` configs.
- Split-K was tried but rejected: the GCU backend has no float atomics, so
  split-K needs a separate ``[SPLIT_K, S, R]`` partial buffer + reduce pass, and
  the reduce kernel alone costs ~400-700 us on GCU -- far more than the whole
  GEMM. The direct-store (no scratch buffer) approach below is strictly better
  here because each output element is written by exactly one program.
- Grouping segments by adapter (so one program reuses a weight slice across
  segments sharing that adapter) was tried but rejected: building the
  concatenated per-adapter row list inside the kernel overflows GCU local
  memory (the per-segment scan materialises several [BLOCK_M] tensors), and
  doing the grouping on the host requires a device->host sync of
  ``seg_indptr``/``weight_indices`` which costs ~109 us/call on GCU -- far
  more than the kernel time it would save.
- ``input_precision="ieee"`` is the only precision the GCU backend accepts and
  also matches the reference's exact float32 matmul. (For bf16 inputs the
  default ``tl.dot`` precision is both faster-looking and wrong: maxdiff
  reaches 0.25-1.0, well outside the bf16 tolerance of 1.5e-2.)
- Indices are kept in int32 throughout: the GCU backend lacks int64, and the
  largest address offset we compute (S*K, with S<=1024, K<=4096) fits int32.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # --- R == 32 family (seg64x8): each segment is exactly 64 rows long.
        # BLOCK_N=32 covers R in one tile. Two M-tile strategies, both valid:
        #
        # * BLOCK_M=64 with num_warps=2: a single M-tile per segment is FULLY
        #   filled (64 rows, no padding), so the dot/store masks are all-true and
        #   no work is wasted. On GCU a BLOCK_M<128 tile only escapes the slow
        #   scalar MMA path when paired with few warps (num_warps=2): with 4
        #   warps the same tile takes ~63 ms (scalar fallback), with 2 warps it
        #   runs ~48 us -- faster than any BLOCK_M=128 config (which always pads
        #   64->128, wasting half the MMA width). This is the winning config for
        #   the seg64x8 benchmark.
        # * BLOCK_M=128/256 with num_warps=4: the previously-tuned fallback.
        #   Each segment needs only one M-tile but the tile is half-empty
        #   (BLOCK_M=128 for a 64-row segment); kept so autotune picks them when
        #   a correctness case has longer segments (max_len>64) where BLOCK_M=64
        #   would split into several under-filled tiles.
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 512},
            num_warps=2,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 1024},
            num_warps=2,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 2048},
            num_warps=2,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 1024},
            num_warps=2,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 2048},
            num_warps=2,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 1024},
            num_warps=1,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 1024},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 2048},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 2048},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 1024},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 32, "BLOCK_K": 2048},
            num_warps=4,
            num_stages=3,
        ),
        # --- R == 64 family (seg256x4): each segment is exactly 256 rows.
        # BLOCK_N=64 covers R in one tile. BLOCK_M=256 gives one full M-tile per
        # segment -> 4 programs (2 per SM, 2 waves), the best occupancy here;
        # smaller BLOCK_M splits 256 into more tiles -> more waves on the 2-SM
        # device and is slower. Autotune keeps BM=128 as a fallback for
        # correctness cases whose max_len is not a multiple of 256.
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 1024},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 2048},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 1024},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 1024},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 2048},
            num_warps=4,
            num_stages=3,
        ),
        # --- R > 64 correctness cases (e.g. stack_num=3 -> R=96) use the 64-wide
        # N-tile with multiple N blocks; the M/K sweep above still applies.
    ],
    key=["K", "R"],
)
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_PERM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    start = tl.load(seg_indptr_ptr + pid_b)
    end = tl.load(seg_indptr_ptr + pid_b + 1)
    seg_len = end - start

    m_off = pid_m * BLOCK_M
    if m_off >= seg_len:
        return

    w_idx = tl.load(weight_indices_ptr + pid_b)

    # N-tiling: a single BLOCK_N=32/64 tile covers the whole output row for the
    # benchmark shapes (R=32/64); correctness cases with R>64 (e.g. stack_num=3
    # -> R=96) use multiple N-tiles via the ``n_off < R`` mask.
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
    # output dtype. Each output element is written by exactly one program, so
    # there is no contention and no extra reduction pass.
    out_ptrs = out_ptr + rows[:, None] * R + n_off[None, :]
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_valid[:, None] & n_mask[None, :],
    )


def sgemm_lora_a(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    R = weights.shape[1]  # == stack_num * r
    # ``torch.empty`` instead of ``torch.zeros``: the kernel writes every
    # ``(row, n)`` element exactly once -- each output row belongs to one
    # segment and that segment's M-tiles cover ``[0, seg_len_b]`` while the
    # N-tiles cover ``[0, R]`` -- so no element is left uninitialised. Skipping
    # the host-side memset saves a measurable ~12us/call on this launch-bound
    # op (zeros memset alone is ~16us vs empty's ~4us on GCU).
    out = torch.empty((S, R), dtype=x.dtype, device=x.device)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation
    has_perm = permutation is not None

    # max segment length (host) -> BLOCK_M ceiling for grid sizing.
    # Prefer the host-side ``batch_info.max_len`` int when available to avoid a
    # device->host sync (`.item()`); fall back to computing it only if missing.
    host_max_len = getattr(batch_info, "max_len", None)
    if host_max_len is None and S > 0:
        seg_lens = seg_indptr[1:] - seg_indptr[:-1]
        host_max_len = int(seg_lens.max().item())
    max_len = host_max_len if host_max_len is not None else 0

    bs = batch_info.bs

    # grid: (segments, M-tiles, N-tiles). The exact BLOCK_M / BLOCK_N are chosen
    # by autotune at launch time, so the grid lambda must read them from the
    # selected config.
    def _grid(meta):
        block_m = meta["BLOCK_M"]
        block_n = meta["BLOCK_N"]
        nm = (max_len + block_m - 1) // block_m if max_len > 0 else 1
        nn = (R + block_n - 1) // block_n
        return (bs, nm, nn)

    perm_ptr = (
        permutation if has_perm else x
    )  # dummy ptr, unused when HAS_PERM=False

    _sgemm_lora_a_kernel[_grid](
        x,
        weights,
        out,
        seg_indptr,
        weight_indices,
        perm_ptr,
        K,
        R,
        HAS_PERM=has_perm,
    )
    return out


__all__ = ["sgemm_lora_a"]
