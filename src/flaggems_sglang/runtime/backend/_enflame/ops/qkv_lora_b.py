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

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# GCU compiler-toolchain compatibility shim (compiler_issues/
# sglang_l2norm_masked2d_slice_pad_start.md, 4th op in that family; also
# gate_up_lora_b.py / l2norm.py in this directory). The process-global
# dialect shim installed by extend_attention strips the trailing zero
# operandSegmentSizes segment from every ``memref_ext.slice_*`` line, but
# the external gcu-compiler-opt requires the full 8-segment form emitted
# for row-masked 2D tile loads, so first-compiles of this op's kernels
# fail with "size mismatch in attribute conversion: 7 vs 8". Retry those
# compiles (and only those, matched on this op's kernel symbol) straight
# against the original external tool with the untouched IR; fallback-only
# so every already-compilable kernel keeps its exact code path. Remove
# once the external tool re-syncs with the in-process passes.
# ---------------------------------------------------------------------------


def _qkv_lora_b_install_gcuir_compat():
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_qkv_lora_b_gcu_compat", False):
        return
    import subprocess

    _prev = toolkit.gcu_compiler_opt

    def _raw_gcu_compiler_opt(text, *args):
        # toolkit._run_command itself is wrapped by decode_grouped_attention's
        # process-global strip (it would re-strip this retry's IR), so invoke
        # the external tool exactly like the upstream gcu_compiler_opt does.
        tool = toolkit._require_toolkit_tool("gcu-compiler-opt")
        result = subprocess.run(
            [tool, "-mlir-print-op-generic", *args],
            input=text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise Exception(result.stderr)
        return result.stdout

    def gcu_compiler_opt_compat(content, *args):
        try:
            return _prev(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if "@_qkv_lora_b" not in text:
                raise first_error
            try:
                return _raw_gcu_compiler_opt(text, *args)
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._qkv_lora_b_gcu_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _qkv_lora_b_install_gcuir_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


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
    acc = tl.dot(x_tile, w_tile, input_precision="ieee") * scaling

    out_ptrs = (
        out_ptr
        + rows[:, None] * STRIDE_OUT_ROW
        + (o_start + offs_n[None, :]) * STRIDE_OUT_N
    )
    store_mask = live[:, None] & in_n[None, :]
    base = tl.load(out_ptrs, mask=store_mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptrs, (base + acc).to(out_ptr.dtype.element_ty), mask=store_mask
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


@triton.jit
def _qkv_lora_b_fast_kernel(
    x_ptr,
    w_ptr,
    seg_ptr,
    widx_ptr,
    ranks_ptr,
    scalings_ptr,
    perm_ptr,
    base_ptr,
    out_ptr,
    offset_ptr,
    R: tl.constexpr,
    STRIDE_X_ROW: tl.constexpr,
    STRIDE_W_LORA: tl.constexpr,
    STRIDE_W_N: tl.constexpr,
    STRIDE_OUT_ROW: tl.constexpr,
    BLOCKS_M: tl.constexpr,
    BLOCKS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_PERM: tl.constexpr,
    EVEN_K: tl.constexpr,
    BF16_EPI: tl.constexpr,
    NSLICES: tl.constexpr,
    PACKED: tl.constexpr,
):
    segment_block = tl.program_id(0)
    segment = segment_block // BLOCKS_M
    row_block = segment_block - segment * BLOCKS_M
    slice_block = tl.program_id(1)

    # Packed layout: grid.y counts live n-blocks across all slices with no
    # per-slice padding. Each program decodes (slice_id, local_block) from the
    # n+1 device offsets once; bound slack past the true live total exits.
    # Dispatch floor is ~54ns/program (2026-11-13 ladder), so skipping the
    # dead kv tail blocks of the legacy n*blocks_n layout is worth -11%/-19%
    # on the core shapes.
    if PACKED:
        prefix = 0
        slice_id = 0
        local_block = 0
        found = 0
        for i in tl.static_range(NSLICES):
            slice_start = tl.load(offset_ptr + i)
            slice_end = tl.load(offset_ptr + i + 1)
            blocks_i = (slice_end - slice_start + BLOCK_N - 1) // BLOCK_N
            if (found == 0) and (slice_block < prefix + blocks_i):
                slice_id = i
                local_block = slice_block - prefix
                found = 1
            prefix += blocks_i
        if slice_block >= prefix:
            return
        o_start = tl.load(offset_ptr + slice_id)
        o_end = tl.load(offset_ptr + slice_id + 1)
    else:
        slice_id = slice_block // BLOCKS_N
        local_block = slice_block - slice_id * BLOCKS_N
        o_start = tl.load(offset_ptr + slice_id)
        o_end = tl.load(offset_ptr + slice_id + 1)
        # Slices narrower than the widest one (kv vs q) leave whole n-blocks
        # idle; exiting before any load/dot keeps them off the MMA unit
        # (measured -18%/-22% on the two core shapes, gcu400).
        if local_block * BLOCK_N >= o_end - o_start:
            return

    seg_start = tl.load(seg_ptr + segment)
    seg_end = tl.load(seg_ptr + segment + 1)

    positions = seg_start + row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = positions < seg_end
    if HAS_PERM:
        rows = tl.load(perm_ptr + positions, mask=mask_m, other=0)
    else:
        rows = positions

    offs_n = local_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < o_end - o_start
    columns = o_start + offs_n
    offs_k = tl.arange(0, BLOCK_K)

    if EVEN_K:
        x_tile = tl.load(
            x_ptr
            + rows[:, None] * STRIDE_X_ROW
            + slice_id * R
            + offs_k[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
    else:
        x_tile = tl.load(
            x_ptr
            + rows[:, None] * STRIDE_X_ROW
            + slice_id * R
            + offs_k[None, :],
            mask=mask_m[:, None] & (offs_k[None, :] < R),
            other=0.0,
        )

    w_index = tl.load(widx_ptr + segment)
    active_rank = tl.load(ranks_ptr + w_index)
    scaling = tl.where(active_rank == 0, 0.0, tl.load(scalings_ptr + w_index))
    if EVEN_K:
        w_tile = tl.load(
            w_ptr
            + w_index * STRIDE_W_LORA
            + (columns[None, :]) * STRIDE_W_N
            + offs_k[:, None],
            mask=mask_n[None, :],
            other=0.0,
        )
    else:
        w_tile = tl.load(
            w_ptr
            + w_index * STRIDE_W_LORA
            + (columns[None, :]) * STRIDE_W_N
            + offs_k[:, None],
            mask=(offs_k[:, None] < R) & mask_n[None, :],
            other=0.0,
        )

    update = tl.dot(x_tile, w_tile)

    out_ptrs = out_ptr + rows[:, None] * STRIDE_OUT_ROW + (columns[None, :])
    store_mask = mask_m[:, None] & mask_n[None, :]
    if BF16_EPI:
        base = tl.load(
            base_ptr + rows[:, None] * STRIDE_OUT_ROW + columns[None, :],
            mask=store_mask,
            other=0.0,
        )
        tl.store(
            out_ptrs,
            base + (scaling * update).to(base.dtype),
            mask=store_mask,
        )
    else:
        base = tl.load(
            base_ptr + rows[:, None] * STRIDE_OUT_ROW + columns[None, :],
            mask=store_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(out_ptrs, base + scaling * update, mask=store_mask)


# gcu400 on-chip DSM per program (bytes), same measurement model as
# gate_up_lora_b.py: block_m*block_n*8 + 2*block_k*(block_m + block_n) at
# num_warps=1 and block_m*block_n*4 + ... at num_warps=4, valid while
# block_m*block_n <= 65536.
_DSM_LIMIT = 917504

# (block_m, block_n, num_warps, num_stages). Re-swept on 2026-11-13 after
# the packed-grid rework moved the small-shape sweet spot (bench A/B on the
# bf16 core shapes, do_bench w1000/r100 median): (64,512) beats the
# 2026-11-08 pick (64,1024) by ~3.7% on case0; (256,128) loses to (256,256)
# by ~5% in the official bench harness even though the bare-kernel probe
# showed the opposite, so the large-shape pick stays (256,256).
_SMALL_TILES = ((64, 512, 4, 2), (64, 1024, 4, 2), (64, 256, 4, 2))
_LARGE_TILES = (
    (256, 256, 4, 2),
    (128, 512, 4, 2),
    (128, 256, 4, 2),
    (256, 128, 4, 2),
)


def _fast_launch_plan(max_len, rank):
    block_k = triton.next_power_of_2(rank)
    candidates = _SMALL_TILES if max_len <= 64 else _LARGE_TILES
    for block_m, block_n, num_warps, num_stages in candidates:
        tile = block_m * block_n
        dsm = tile * (8 if num_warps == 1 else 4) + 2 * block_k * (
            block_m + block_n
        )
        if tile <= 65536 and dsm <= _DSM_LIMIT:
            return block_m, block_n, num_warps, num_stages
    return None


def _fast_path_ok(x, weights, batch_info, output_offset, base_output):
    permutation = batch_info.permutation
    return (
        x.element_size() == 2
        and x.is_contiguous()
        and weights.is_contiguous()
        and base_output.is_contiguous()
        and output_offset.dtype == torch.int32
        and output_offset.is_contiguous()
        and batch_info.seg_indptr.dtype == torch.int32
        and batch_info.weight_indices.dtype == torch.int32
        and batch_info.lora_ranks.dtype == torch.int32
        # Gather row bases (permutation) blow the fat-tile DSM budget
        # (measured 1.99MB > 917KB at 64x1024); routed to the legacy
        # kernel until gather-aware tiles are tuned.
        and permutation is None
        and isinstance(batch_info.max_len, int)
        and batch_info.max_len >= 0
    )


def _packed_grid_y(n_slices, out_dim, max_dim, block_n, blocks_n_max):
    """Upper bound on live n-blocks across slices, without reading the
    device-side output_offset (a D2H sync would be needed for exact
    per-slice widths). For slice widths w_i summing to out_dim with each
    w_i <= max_dim (the max_qkv_out_dim contract the generic kernel's
    grid already relies on), sum(ceil(w_i/B)) is bounded by
      min(n*ceil(max/B), ceil(max/B)+ceil((out-max)/B)+n-1, n+ceil((out-n)/B))
    (brute-force verified over 121800 partitions; jobs
    j20261113174213_ba88d0a4 tmp/check_bound.py). The kernel decodes the
    exact live blocks and returns on the slack, so the bound only has to
    be >= the true live count, never smaller. Falls back to the legacy
    n*blocks_n layout when the bound is no tighter (uniform slices) or
    exceeds the grid.y hardware limit.
    """
    bound = min(
        n_slices * blocks_n_max,
        blocks_n_max + -(-(out_dim - max_dim) // block_n) + n_slices - 1,
        (
            n_slices + -(-(out_dim - n_slices) // block_n)
            if out_dim >= n_slices
            else n_slices
        ),
    )
    packed = bound < n_slices * blocks_n_max and bound <= 255
    return (bound if packed else n_slices * blocks_n_max), packed


def _qkv_lora_b_fast(
    x,
    weights,
    batch_info,
    output_offset,
    max_qkv_out_dim,
    base_output,
    plan,
    grid_y,
    packed,
):
    block_m, block_n, num_warps, num_stages = plan
    rank_dim = weights.shape[-1]
    n_slices = output_offset.numel() - 1
    bs = int(batch_info.bs)
    max_len = max(int(batch_info.max_len), 1)

    blocks_m = -(-max_len // block_m)
    blocks_n = -(-int(max_qkv_out_dim) // block_n)

    permutation = batch_info.permutation
    out = torch.empty_like(base_output)
    _qkv_lora_b_fast_kernel[(bs * blocks_m, grid_y)](
        x,
        weights,
        batch_info.seg_indptr,
        batch_info.weight_indices,
        batch_info.lora_ranks,
        batch_info.scalings,
        x if permutation is None else permutation,
        base_output,
        out,
        output_offset,
        rank_dim,
        x.shape[1],
        weights.stride(0),
        weights.stride(1),
        base_output.shape[1],
        blocks_m,
        blocks_n,
        block_m,
        block_n,
        triton.next_power_of_2(rank_dim),
        permutation is not None,
        triton.next_power_of_2(rank_dim) == rank_dim,
        x.element_size() == 2 and base_output.element_size() == 2,
        n_slices,
        packed,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def qkv_lora_b(
    x, qkv_lora_b, batch_info, output_offset, max_qkv_out_dim, base_output
):
    if max_qkv_out_dim >= 4096 and _fast_path_ok(
        x, qkv_lora_b, batch_info, output_offset, base_output
    ):
        rank_dim = qkv_lora_b.shape[-1]
        max_len = max(int(batch_info.max_len), 1)
        plan = _fast_launch_plan(max_len, rank_dim)
        if plan is not None:
            block_m, block_n = plan[0], plan[1]
            blocks_m = -(-max_len // block_m)
            blocks_n = -(-int(max_qkv_out_dim) // block_n)
            n_slices = output_offset.numel() - 1
            grid_y, packed = _packed_grid_y(
                n_slices,
                base_output.shape[1],
                int(max_qkv_out_dim),
                block_n,
                blocks_n,
            )
            if int(batch_info.bs) * blocks_m <= 65535 and grid_y <= 255:
                return _qkv_lora_b_fast(
                    x,
                    qkv_lora_b,
                    batch_info,
                    output_offset,
                    max_qkv_out_dim,
                    base_output,
                    plan,
                    grid_y,
                    packed,
                )
    x = x.contiguous()
    weights = qkv_lora_b.contiguous()
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

    seg_lens = (
        seg[stride_seg : (bs + 1) * stride_seg : stride_seg]
        - seg[0 : bs * stride_seg : stride_seg]
    )
    max_len = int(seg_lens.max()) if bs > 0 else 0

    block_s = 16
    block_n = 128
    block_r = _block_r_for(rank_dim)
    n_tiles = -(-int(max_qkv_out_dim) // block_n)
    s_tiles = -(-max_len // block_s)

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
        block_s,
        block_n,
        block_r,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["qkv_lora_b"]
