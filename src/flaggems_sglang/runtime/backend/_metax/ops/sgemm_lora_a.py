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

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    cfgs = []
    for bm in (16, 32):
        for bn in (16, 32, 64):
            for bk in (128, 256, 512):
                for nw in (4, 8):
                    for ns in (2, 3, 4):
                        cfgs.append(
                            triton.Config(
                                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                                num_warps=nw,
                                num_stages=ns,
                            )
                        )
    return cfgs


@triton.autotune(configs=_configs(), key=("K", "R", "COMPUTE_F32"))
@triton.jit
def _sgemm_lora_a_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    perm_ptr,  # permutation (int32) or nullptr when absent
    seg_indptr_ptr,  # [bs+1] segment row pointers (int32)
    weight_indices_ptr,  # [bs] adapter index per segment (int32)
    # strides
    x_row_stride,  # K
    w_lora_stride,  # R*K (outer dim of weights)
    w_row_stride,  # K   (row of a single adapter -> weights[i]: [R, K])
    out_row_stride,  # R
    K,  # reduction dim
    R,  # output width (stack_num * r)
    HAS_PERM: tl.constexpr,
    COMPUTE_F32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # segment index
    pid_m = tl.program_id(1)  # m-tile index within the segment
    pid_n = tl.program_id(2)  # n-tile index

    # --- this segment's row range + adapter -------------------------------
    start = tl.load(seg_indptr_ptr + pid_b).to(tl.int64)
    end = tl.load(seg_indptr_ptr + pid_b + 1).to(tl.int64)
    seg_len = end - start
    w_idx = tl.load(weight_indices_ptr + pid_b).to(tl.int64)

    # m-tile bounds within this segment.
    m_start = pid_m * BLOCK_M
    # Idle programs (m-tile past the segment end) exit immediately.
    if m_start >= seg_len:
        return
    nrows = tl.minimum(BLOCK_M, seg_len - m_start)

    # --- output-column tile ----------------------------------------------
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    n_mask = n_off < R  # [BLOCK_N]

    # --- physical row indices for this tile ------------------------------
    local_tok = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M] within-seg offset
    row_mask = tl.arange(0, BLOCK_M) < nrows  # [BLOCK_M]
    logical_tok = start + local_tok  # [BLOCK_M] logical token index
    if HAS_PERM:
        phys_row = tl.load(perm_ptr + logical_tok, mask=row_mask, other=0).to(
            tl.int64
        )  # [BLOCK_M]
    else:
        phys_row = logical_tok  # [BLOCK_M]

    # --- accumulator -----------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Base pointer for this adapter's weight tile: weights[w_idx] is [R, K].
    w_base = w_ptr + w_idx * w_lora_stride

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_off < K  # [BLOCK_K]

        # x[phys_row, k_off] -> [BLOCK_M, BLOCK_K]
        x_off = phys_row[:, None] * x_row_stride + k_off[None, :]
        x_mask = row_mask[:, None] & k_mask[None, :]
        a = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # weights[w_idx][n_off, k_off] -> [BLOCK_K, BLOCK_N] (the orientation
        # tl.dot wants), gathered as w_base + k_off[:,None] + n_off[None,:]*w_row_stride.
        # This avoids the explicit .T in tl.dot(a, b.T) the previous layout
        # used (weight transposition is a real op on this device).
        w_off = k_off[:, None] + n_off[None, :] * w_row_stride
        w_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(w_base + w_off, mask=w_mask, other=0.0)

        if COMPUTE_F32:
            acc += tl.dot(
                a.to(tl.float32), b.to(tl.float32), input_precision="ieee"
            )
        else:
            acc += tl.dot(a, b, input_precision="ieee")

    # --- store out[phys_row, n_off] --------------------------------------
    out_off = phys_row[:, None] * out_row_stride + n_off[None, :]
    out_mask = row_mask[:, None] & n_mask[None, :]
    tl.store(
        out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=out_mask
    )


def sgemm_lora_a(x, weights, batch_info, stack_num=1):
    S, K = x.shape
    R = weights.shape[1]

    # Every output row is written exactly once: the segments partition the
    # token range [0, S) (seg_indptr is a cumulative pointer ending at S), and
    # whether or not a permutation is present, ``rows`` is a permutation of
    # [start, end) so each token index in [0, S) lands in exactly one segment's
    # store. Empty segments (start == end) contribute no rows but the
    # indptr still keeps the partition contiguous, so no output element is
    # left unwritten. ``torch.empty`` therefore skips the redundant zero
    # memset that ``torch.zeros`` would issue — a measurable win because the
    # kernel is so fast that the memset was a noticeable fraction of the call.
    out = torch.empty((S, R), dtype=x.dtype, device=x.device)

    seg_indptr = batch_info.seg_indptr
    weight_indices = batch_info.weight_indices
    permutation = batch_info.permutation

    # Empty-input fast path.
    if S == 0:
        return out

    bs = batch_info.bs
    max_len = batch_info.max_len

    # Both the m-tile and n-tile widths are autotuned, so the whole grid is
    # built per-config from the meta values. The (b) axis is fixed by the input;
    # the m axis is ``cdiv(max_len, BLOCK_M)`` and the n axis is
    # ``cdiv(R, BLOCK_N)`` per config. Idle m-tiles (past a segment's end)
    # early-exit inside the kernel, so a coarse BLOCK_M still handles the
    # short correctness-case segments.
    def grid(meta):
        num_m_blocks = triton.cdiv(max_len, meta["BLOCK_M"])
        num_n_blocks = triton.cdiv(R, meta["BLOCK_N"])
        return (bs, num_m_blocks, num_n_blocks)

    x_row_stride = x.stride(0)
    out_row_stride = out.stride(0)
    w_lora_stride = weights.stride(0)
    w_row_stride = weights.stride(1)

    has_perm = permutation is not None
    # Compute in float32 only when inputs are float32 (matches the reference's
    # exact float32 path). For bf16/f16, use the native precision (still IEEE
    # matmul) to leverage tensor cores and stay within the loose tolerance.
    compute_f32 = x.dtype == torch.float32

    _sgemm_lora_a_kernel[grid](
        x,
        weights,
        out,
        permutation if has_perm else None,
        seg_indptr,
        weight_indices,
        x_row_stride,
        w_lora_stride,
        w_row_stride,
        out_row_stride,
        K,
        R,
        HAS_PERM=has_perm,
        COMPUTE_F32=compute_f32,
    )

    return out


__all__ = ["sgemm_lora_a"]
