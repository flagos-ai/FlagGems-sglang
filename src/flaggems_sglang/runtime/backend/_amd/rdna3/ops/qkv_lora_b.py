# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""qkv_lora_b for RDNA3 (gfx1100).

Changes vs the generic op:
- Use batch_info.max_len when present so the wrapper does not call
  seg_lens.max() (a host sync on every launch).
- tl.split/tl.join on the WMMA product so base load / store become 16-byte
  accesses instead of 2-byte scalar traffic.
- Larger tiles and more warps on typical decode shapes; fp32 keeps a
  narrower tile like the other gfx1100 LoRA-B ops.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qkv_lora_b_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    seg_ptr,
    widx_ptr,
    ranks_ptr,
    scalings_ptr,
    perm_ptr,
    offset_ptr,
    STRIDE_SEG: tl.constexpr,
    STRIDE_WIDX: tl.constexpr,
    STRIDE_RANKS: tl.constexpr,
    STRIDE_PERM: tl.constexpr,
    STRIDE_OFFSET: tl.constexpr,
    R: tl.constexpr,
    STRIDE_X_ROW: tl.constexpr,
    STRIDE_X_COL: tl.constexpr,
    STRIDE_W_LORA: tl.constexpr,
    STRIDE_W_N: tl.constexpr,
    STRIDE_W_R: tl.constexpr,
    STRIDE_OUT_ROW: tl.constexpr,
    STRIDE_OUT_N: tl.constexpr,
    N_TILES: tl.constexpr,
    HAS_PERM: tl.constexpr,
    USE_DOT: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    b = tl.program_id(1)
    slice_id = tl.program_id(2)
    pid_s = tl.program_id(0) // N_TILES
    pid_n = tl.program_id(0) % N_TILES

    seg_start = tl.load(seg_ptr + b * STRIDE_SEG)
    seg_end = tl.load(seg_ptr + (b + 1) * STRIDE_SEG)
    w_index = tl.load(widx_ptr + b * STRIDE_WIDX)
    rank = tl.load(ranks_ptr + w_index * STRIDE_RANKS)
    scaling = tl.load(scalings_ptr + w_index)

    o_start = tl.load(offset_ptr + slice_id * STRIDE_OFFSET)
    o_end = tl.load(offset_ptr + (slice_id + 1) * STRIDE_OFFSET)

    offs_p = seg_start + pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    live = (offs_p < seg_end) & (rank > 0)

    if HAS_PERM:
        rows = tl.load(perm_ptr + offs_p * STRIDE_PERM, mask=live, other=0)
    else:
        rows = offs_p

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R)
    in_r = offs_r < R
    in_n = offs_n < o_end - o_start

    if USE_DOT:
        x_tile = tl.load(
            x_ptr
            + rows[:, None] * STRIDE_X_ROW
            + (slice_id * R + offs_r[None, :]) * STRIDE_X_COL,
            mask=live[:, None] & in_r[None, :],
            other=0.0,
        )
        w_tile = tl.load(
            w_ptr
            + w_index * STRIDE_W_LORA
            + offs_r[:, None] * STRIDE_W_R
            + (o_start + offs_n[None, :]) * STRIDE_W_N,
            mask=in_r[:, None] & in_n[None, :],
            other=0.0,
        )
        acc = tl.dot(x_tile, w_tile, input_precision="ieee")
        # Without this round trip the AMD backend loads base and stores the
        # result in the WMMA layout, two bytes a lane.
        low, high = tl.split(tl.reshape(acc, (BLOCK_S, BLOCK_N // 2, 2)))
        acc = tl.reshape(tl.join(low, high), (BLOCK_S, BLOCK_N))
    else:
        acc = tl.zeros((BLOCK_S, BLOCK_N), dtype=tl.float32)
        for r in range(0, R):
            x_vals = tl.load(
                x_ptr + rows * STRIDE_X_ROW + (slice_id * R + r) * STRIDE_X_COL,
                mask=live,
                other=0.0,
            ).to(tl.float32)
            w_vals = tl.load(
                w_ptr
                + w_index * STRIDE_W_LORA
                + r * STRIDE_W_R
                + (o_start + offs_n) * STRIDE_W_N,
                mask=in_n,
                other=0.0,
            ).to(tl.float32)
            acc += x_vals[:, None] * w_vals[None, :]

    out_ptrs = (
        out_ptr
        + rows[:, None] * STRIDE_OUT_ROW
        + (o_start + offs_n[None, :]) * STRIDE_OUT_N
    )
    store_mask = live[:, None] & in_n[None, :]
    base = tl.load(out_ptrs, mask=store_mask, other=0.0).to(tl.float32)
    if USE_DOT:
        result = tl.fma(acc, scaling, base)
    else:
        result = base + acc * scaling
    tl.store(
        out_ptrs, result.to(out_ptr.dtype.element_ty), mask=store_mask
    )


def _as_index(tensor):
    if tensor.dtype != torch.int64:
        return tensor, 1
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.view(torch.int32), 2


def _block_r_for(rank):
    if rank <= 16:
        return 16
    if rank <= 32:
        return 32
    if rank <= 64:
        return 64
    if rank <= 128:
        return 128
    return 256


def qkv_lora_b(
    x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim, base_output
):
    x = x.contiguous()
    weights = qkv_lora_b.contiguous()
    # Write into a clone so uncovered elements keep base_output (same contract).
    out = base_output.clone()

    rank_dim = weights.shape[-1]
    n_slices = output_offset.numel() - 1
    bs = int(batch_info.bs)

    seg, stride_seg = _as_index(batch_info.seg_indptr)
    widx, stride_widx = _as_index(batch_info.weight_indices)
    ranks, stride_ranks = _as_index(batch_info.lora_ranks)
    offsets, stride_offset = _as_index(output_offset)
    permutation = batch_info.permutation
    has_perm = permutation is not None
    perm, stride_perm = _as_index(permutation) if has_perm else (seg, 1)

    max_len = getattr(batch_info, "max_len", None)
    if max_len is None and bs > 0:
        seg_lens = (
            seg[stride_seg : (bs + 1) * stride_seg : stride_seg]
            - seg[0 : bs * stride_seg : stride_seg]
        )
        max_len = int(seg_lens.max())
    max_len = int(max_len or 0)

    use_dot = x.dtype != torch.float32
    # gfx1100 tile / warp choices aligned with gate_up_lora_b / sgemm_lora_b.
    if not use_dot:
        block_s, block_n, num_warps = (
            (16, 64, 16) if max_len <= 32 else (32, 128, 8)
        )
    elif max_len <= 32:
        block_s, block_n, num_warps = 32, 64, 8
    else:
        block_s, block_n, num_warps = 32, 128, 8 if rank_dim <= 64 else 8

    block_r = _block_r_for(rank_dim)
    n_tiles = -(-int(max_qkv_out_dim) // block_n)
    s_tiles = -(-max_len // block_s) if max_len > 0 else 1

    sx = x.stride()
    sw = weights.stride()
    so = out.stride()

    _qkv_lora_b_kernel[
        (max(s_tiles * n_tiles, 1), max(bs, 1), max(n_slices, 1))
    ](
        x,
        weights,
        out,
        seg,
        widx,
        ranks,
        batch_info.scalings,
        perm,
        offsets,
        stride_seg,
        stride_widx,
        stride_ranks,
        stride_perm,
        stride_offset,
        rank_dim,
        sx[0],
        sx[1],
        sw[0],
        sw[1],
        sw[2],
        so[0],
        so[1],
        n_tiles,
        has_perm,
        use_dot,
        block_s,
        block_n,
        block_r,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


__all__ = ["qkv_lora_b"]
