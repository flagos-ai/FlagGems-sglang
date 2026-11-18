# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import re

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_up_lora_b_fast_kernel(
    x,
    weights,
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    permutation,
    base_output,
    output,
    rank: tl.constexpr,
    output_dim: tl.constexpr,
    blocks_m: tl.constexpr,
    blocks_n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_permutation: tl.constexpr,
    even_k: tl.constexpr,
    even_n: tl.constexpr,
    bf16_epi: tl.constexpr,
):
    segment_block = tl.program_id(0)
    segment = segment_block // blocks_m
    row_block = segment_block - segment * blocks_m
    slice_block = tl.program_id(1)
    slice_index = slice_block // blocks_n
    local_block = slice_block - slice_index * blocks_n
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    adapter = tl.load(weight_indices + segment)
    active_rank = tl.load(lora_ranks + adapter)
    scaling = tl.where(active_rank == 0, 0.0, tl.load(scalings + adapter))
    block_start = segment_start + row_block * block_m
    positions = block_start + tl.arange(0, block_m)
    offsets_n = local_block * block_n + tl.arange(0, block_n)
    columns = slice_index * output_dim + offsets_n
    offsets_k = tl.arange(0, block_k)
    # Uniform segment lengths (all core benchmark shapes) put every row
    # block fully in-bounds; a runtime scalar branch gives those programs a
    # hot path with no row masks at all (measured -3.1% on the 128x256 nw2
    # tile and -5.6% on 64x1024 nw4, gcu400 2026-11-13). Tail blocks of
    # ragged segments take the masked cold path below.
    if block_start + block_m <= segment_end:
        if has_permutation:
            rows = tl.load(permutation + positions)
        else:
            rows = positions
        if even_k:
            # rank == block_k: the k mask term is statically all-true.
            x_values = tl.load(
                x
                + rows[:, None] * (2 * rank)
                + slice_index * rank
                + offsets_k[None, :],
            )
        else:
            x_values = tl.load(
                x
                + rows[:, None] * (2 * rank)
                + slice_index * rank
                + offsets_k[None, :],
                mask=offsets_k[None, :] < rank,
                other=0.0,
            )
        if even_k and even_n:
            # block_k == rank and block_n divides output_dim: weight tile
            # is fully in-bounds, drop the 2D predicate entirely.
            weight_values = tl.load(
                weights
                + adapter * (2 * output_dim * rank)
                + columns[None, :] * rank
                + offsets_k[:, None],
            )
        elif even_k:
            weight_values = tl.load(
                weights
                + adapter * (2 * output_dim * rank)
                + columns[None, :] * rank
                + offsets_k[:, None],
                mask=offsets_n[None, :] < output_dim,
                other=0.0,
            )
        else:
            weight_values = tl.load(
                weights
                + adapter * (2 * output_dim * rank)
                + columns[None, :] * rank
                + offsets_k[:, None],
                mask=(offsets_k[:, None] < rank)
                & (offsets_n[None, :] < output_dim),
                other=0.0,
            )
        update = tl.dot(x_values, weight_values)
        output_offsets = rows[:, None] * (2 * output_dim) + columns[None, :]
        if even_n:
            if bf16_epi:
                # Keep base_output in its 2-byte dtype and fold the scaling
                # result to the same dtype before a single low-precision
                # add: avoids the full-tile fp32 base materialization
                # (measured -25% on the 64x512 tile, gcu400).
                base = tl.load(base_output + output_offsets)
                tl.store(
                    output + output_offsets,
                    base + (scaling * update).to(base.dtype),
                )
            else:
                base = tl.load(base_output + output_offsets).to(tl.float32)
                tl.store(output + output_offsets, base + scaling * update)
        else:
            out_mask = offsets_n[None, :] < output_dim
            if bf16_epi:
                base = tl.load(
                    base_output + output_offsets, mask=out_mask, other=0.0
                )
                tl.store(
                    output + output_offsets,
                    base + (scaling * update).to(base.dtype),
                    mask=out_mask,
                )
            else:
                base = tl.load(
                    base_output + output_offsets, mask=out_mask, other=0.0
                ).to(tl.float32)
                tl.store(
                    output + output_offsets,
                    base + scaling * update,
                    mask=out_mask,
                )
    else:
        mask_m = positions < segment_end
        if has_permutation:
            rows = tl.load(permutation + positions, mask=mask_m, other=0)
        else:
            rows = positions
        mask_n = offsets_n < output_dim
        if even_k:
            x_values = tl.load(
                x
                + rows[:, None] * (2 * rank)
                + slice_index * rank
                + offsets_k[None, :],
                mask=mask_m[:, None],
                other=0.0,
            )
        else:
            x_values = tl.load(
                x
                + rows[:, None] * (2 * rank)
                + slice_index * rank
                + offsets_k[None, :],
                mask=mask_m[:, None] & (offsets_k[None, :] < rank),
                other=0.0,
            )
        if even_k and even_n:
            weight_values = tl.load(
                weights
                + adapter * (2 * output_dim * rank)
                + columns[None, :] * rank
                + offsets_k[:, None],
            )
        elif even_k:
            weight_values = tl.load(
                weights
                + adapter * (2 * output_dim * rank)
                + columns[None, :] * rank
                + offsets_k[:, None],
                mask=mask_n[None, :],
                other=0.0,
            )
        else:
            weight_values = tl.load(
                weights
                + adapter * (2 * output_dim * rank)
                + columns[None, :] * rank
                + offsets_k[:, None],
                mask=(offsets_k[:, None] < rank) & mask_n[None, :],
                other=0.0,
            )
        update = tl.dot(x_values, weight_values)
        output_offsets = rows[:, None] * (2 * output_dim) + columns[None, :]
        if even_n:
            output_mask = mask_m[:, None]
        else:
            output_mask = mask_m[:, None] & mask_n[None, :]
        if bf16_epi:
            base = tl.load(
                base_output + output_offsets, mask=output_mask, other=0.0
            )
            tl.store(
                output + output_offsets,
                base + (scaling * update).to(base.dtype),
                mask=output_mask,
            )
        else:
            base = tl.load(
                base_output + output_offsets, mask=output_mask, other=0.0
            ).to(tl.float32)
            tl.store(
                output + output_offsets,
                base + scaling * update,
                mask=output_mask,
            )


@triton.jit
def _gate_up_lora_b_accurate_kernel(
    x,
    weights,
    seg_indptr,
    weight_indices,
    lora_ranks,
    scalings,
    permutation,
    base_output,
    output,
    rank: tl.constexpr,
    output_dim: tl.constexpr,
    blocks_m: tl.constexpr,
    blocks_n: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    has_permutation: tl.constexpr,
):
    segment_block = tl.program_id(0)
    segment = segment_block // blocks_m
    row_block = segment_block - segment * blocks_m
    slice_block = tl.program_id(1)
    slice_index = slice_block // blocks_n
    local_block = slice_block - slice_index * blocks_n
    segment_start = tl.load(seg_indptr + segment)
    segment_end = tl.load(seg_indptr + segment + 1)
    positions = segment_start + row_block * block_m + tl.arange(0, block_m)
    mask_m = positions < segment_end
    if has_permutation:
        rows = tl.load(permutation + positions, mask=mask_m, other=0)
    else:
        rows = positions
    offsets_n = local_block * block_n + tl.arange(0, block_n)
    mask_n = offsets_n < output_dim
    columns = slice_index * output_dim + offsets_n
    offsets_k = tl.arange(0, block_k)
    x_values = tl.load(
        x
        + rows[:, None] * (2 * rank)
        + slice_index * rank
        + offsets_k[None, :],
        mask=mask_m[:, None] & (offsets_k[None, :] < rank),
        other=0.0,
    )
    adapter = tl.load(weight_indices + segment)
    active_rank = tl.load(lora_ranks + adapter)
    scaling = tl.where(active_rank == 0, 0.0, tl.load(scalings + adapter))
    weight_values = tl.load(
        weights
        + adapter * (2 * output_dim * rank)
        + columns[None, :] * rank
        + offsets_k[:, None],
        mask=(offsets_k[:, None] < rank) & mask_n[None, :],
        other=0.0,
    )
    update = tl.dot(
        x_values,
        weight_values,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    output_offsets = rows[:, None] * (2 * output_dim) + columns[None, :]
    output_mask = mask_m[:, None] & mask_n[None, :]
    base = tl.load(
        base_output + output_offsets, mask=output_mask, other=0.0
    ).to(tl.float32)
    tl.store(
        output + output_offsets, base + scaling * update, mask=output_mask
    )


# ---------------------------------------------------------------------------
# GCU compiler-toolchain compatibility shim (compiler_issues/
# sglang_gate_up_lora_b.md; same family as chunk_local_cumsum_vector.py and
# vllm_moe_sum.md). The 2026-11-07 20:22 in-process pass library emits GCUIR
# whose memref_ext.slice_* ops carry a trailing empty variadic segment in
# operandSegmentSizes (and gcu.matmul an inputPrecision property) that the
# legacy external gcu-compiler-opt rejects as "size mismatch in attribute
# conversion: 8 vs 7" on every fresh compile of this op's 2D-tile kernels.
# The rewrite below is fallback-only and scoped to this op's kernel names;
# remove it once the external tool re-syncs with the in-process passes.
# ---------------------------------------------------------------------------

_GATE_UP_LORA_B_COMPAT_KERNELS = (
    "_gate_up_lora_b_fast_kernel",
    "_gate_up_lora_b_accurate_kernel",
)


def _gate_up_lora_b_downgrade_gcuir_for_external_tool(text):
    """Rewrite new-format GCUIR op encodings back to the old external-tool form."""

    def _strip_trailing_zero_segments(match):
        entries = [entry.strip() for entry in match.group(2).split(",")]
        while len(entries) > 1 and entries[-1] == "0":
            entries.pop()
        return match.group(1) + ", ".join(entries) + match.group(3)

    out_lines = []
    for line in text.splitlines():
        if '"memref_ext.slice_' in line or '"gcu.matmul"' in line:
            if "operandSegmentSizes = array<i32:" in line:
                line = re.sub(
                    r"(operandSegmentSizes = array<i32: )"
                    r"((?:\s*\d+\s*,\s*)*\s*\d+\s*)(>)",
                    _strip_trailing_zero_segments,
                    line,
                )
            if '"gcu.matmul"' in line:
                # Legacy gcu.matmul keeps one empty variadic segment entry.
                line = re.sub(
                    r"operandSegmentSizes = array<i32: (\d+(?:, \d+){3})>",
                    r"operandSegmentSizes = array<i32: \1, 0>",
                    line,
                )
                line = re.sub(r"inputPrecision = \d+ : i32, ", "", line)
                line = re.sub(r", inputPrecision = \d+ : i32", "", line)
        if '"memref_ext.memset_start"' in line:
            line = re.sub(
                r" <\{operandSegmentSizes = array<i32: [0-9, ]+>\}>", "", line
            )
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _gate_up_lora_b_install_gcuir_text_compat():
    """Wrap enflame toolkit.gcu_compiler_opt with a scoped downgrade retry."""
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_gate_up_lora_b_gcu_compat", False):
        return
    original = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return original(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if not any(
                name in text for name in _GATE_UP_LORA_B_COMPAT_KERNELS
            ):
                raise first_error
            try:
                return toolkit._run_command(
                    "gcu-compiler-opt",
                    _gate_up_lora_b_downgrade_gcuir_for_external_tool(text),
                    "-mlir-print-op-generic",
                    *args,
                )
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._gate_up_lora_b_gcu_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _gate_up_lora_b_install_gcuir_text_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


def _launch(
    kernel,
    x,
    weights,
    batch_info,
    output_dim,
    base_output,
    block_m,
    block_n,
    num_warps,
    num_stages,
    fast_flags=None,
):
    rank = weights.shape[-1]
    max_segment_length = max(1, batch_info.max_len)
    blocks_m = triton.cdiv(max_segment_length, block_m)
    blocks_n = triton.cdiv(output_dim, block_n)
    output = torch.empty_like(base_output)
    permutation = batch_info.permutation
    if fast_flags is None:
        kernel[(batch_info.bs * blocks_m, 2 * blocks_n)](
            x,
            weights,
            batch_info.seg_indptr,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            batch_info.scalings,
            x if permutation is None else permutation,
            base_output,
            output,
            rank,
            output_dim,
            blocks_m,
            blocks_n,
            block_m,
            block_n,
            triton.next_power_of_2(rank),
            has_permutation=permutation is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        even_k, even_n, bf16_epi = fast_flags
        kernel[(batch_info.bs * blocks_m, 2 * blocks_n)](
            x,
            weights,
            batch_info.seg_indptr,
            batch_info.weight_indices,
            batch_info.lora_ranks,
            batch_info.scalings,
            x if permutation is None else permutation,
            base_output,
            output,
            rank,
            output_dim,
            blocks_m,
            blocks_n,
            block_m,
            block_n,
            triton.next_power_of_2(rank),
            has_permutation=permutation is not None,
            even_k=even_k,
            even_n=even_n,
            bf16_epi=bf16_epi,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return output


# gcu400 on-chip DSM per program (bytes). The fast kernel's per-program DSM,
# measured on gcu400 across tile configs, is
#   block_m*block_n*8 + 2*block_k*(block_m + block_n)   at num_warps=1
#   block_m*block_n*4 + 2*block_k*(block_m + block_n)   at num_warps=4
# and only holds while block_m*block_n <= 65536: larger tiles fall back to
# full fp32 materialization (256x512 at any num_warps requests 1146880 B >
# 917504 B and fails to compile with OutOfResources, with or without the
# bf16 epilogue; the original 256x512 nw1 config failed the same way).
# With the full-block hot/cold branch below the compiler liveness-accounts
# both bodies and requests the *8 staging form regardless of num_warps
# (measured on the 2026-11-13 stack: 64x512 nw4 299008 B, 128x256 nw2
# 311296 B, 256x256 nw4 589824 B; the branch-free kernel requested 167936 /
# 180224 / 327680 B). The tuned tiles stay under the 448KB dual-program
# concurrency line, so the ladder prices every entry at the *8 rate.
_DSM_LIMIT = 917504

# (block_m, block_n, num_warps, num_stages), fastest first per branch,
# measured on the two core benchmark shapes (bf16, fast kernel with the
# even_k/even_n/bf16_epi constexpr path and the full-block hot branch,
# 2026-11-13 stack):
#   seg=(64,)*8 r=32  -> 64x512 nw4 st2 28.5 us (nw8 28.8, 64x1024 nw4 28.8)
#   seg=(256,)*4 r=64 -> 128x256 nw2 st2 45.1 us (256x128 nw2 46.2,
#     256x256 nw4 50.1; all finer/broader tiles measured worse)
_SMALL_TILES = ((64, 512, 4, 2), (64, 512, 2, 2), (64, 256, 4, 2))
_LARGE_TILES = (
    (128, 256, 2, 2),
    (256, 128, 2, 2),
    (256, 256, 4, 2),
    (128, 512, 4, 2),
    (128, 128, 4, 2),
)


def _fast_launch_plan(max_len, rank):
    """Pick fast-kernel tiles that fit the gcu400 DSM budget, or None."""
    block_k = triton.next_power_of_2(rank)
    candidates = _SMALL_TILES if max_len <= 64 else _LARGE_TILES
    for block_m, block_n, num_warps, num_stages in candidates:
        tile = block_m * block_n
        dsm = tile * 8 + 2 * block_k * (block_m + block_n)
        if tile <= 65536 and dsm <= _DSM_LIMIT:
            return block_m, block_n, num_warps, num_stages
    return None


def gate_up_lora_b(x, gate_up_lora_b, batch_info, output_dim, base_output):
    if output_dim >= 4096 and x.element_size() == 2:
        plan = _fast_launch_plan(batch_info.max_len, gate_up_lora_b.shape[-1])
        if plan is not None:
            block_m, block_n, num_warps, num_stages = plan
            rank = gate_up_lora_b.shape[-1]
            fast_flags = (
                triton.next_power_of_2(rank) == rank,
                output_dim % block_n == 0,
                base_output.element_size() == 2,
            )
            return _launch(
                _gate_up_lora_b_fast_kernel,
                x,
                gate_up_lora_b,
                batch_info,
                output_dim,
                base_output,
                block_m,
                block_n,
                num_warps,
                num_stages,
                fast_flags=fast_flags,
            )
    return _launch(
        _gate_up_lora_b_accurate_kernel,
        x,
        gate_up_lora_b,
        batch_info,
        output_dim,
        base_output,
        32,
        64,
        4,
        2,
    )


__all__ = ["gate_up_lora_b"]
