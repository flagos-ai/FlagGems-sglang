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


def _patch_enflame_operand_segments():
    """Compatibility shim for the 2026-11-07 triton-gcu backend stack skew.

    The in-process ``libtriton_gcu400_core.so`` (2026-11-07 20:22 build)
    emits ``operandSegmentSizes`` arrays with trailing zero entries
    (``gcu.matmul``: 8 entries instead of 5, ``memref_ext.slice_start`` /
    ``slice_pad_start``: one extra trailing zero) that the installed external
    ``gcu-compiler-opt`` (2026-10-20 build) rejects with "size mismatch in
    attribute conversion: N vs M". Trailing zero segments are empty variadic
    segments, so stripping them keeps the operand list and semantics
    identical and restores the pre-11-07 textual form the external tool
    accepts (verified bit-identical against the form emitted by processes
    that register the older dialect). Idempotent and best-effort: on any
    internal change of the toolkit this degrades to the unpatched behavior.
    See compiler_issues/sglang_decode_grouped_attention.md.
    """
    try:
        import triton.backends.enflame.toolkit as toolkit
    except ImportError:
        return
    if getattr(toolkit._run_command, "_operand_segments_patched", False):
        return
    original = toolkit._run_command
    fix_ops = ("gcu.matmul", "memref_ext.slice_start", "memref_ext.slice_pad_start")
    array_re = re.compile(r"(operandSegmentSizes = array<i32: )([^>]*?)(>})")

    def strip_trailing_zeros(match):
        nums = [part.strip() for part in match.group(2).split(",")]
        while nums and nums[-1] == "0":
            nums.pop()
        return match.group(1) + ", ".join(nums) + match.group(3)

    def run_command(cmd, content, *args):
        if not isinstance(content, str):
            content = str(content)
        if "operandSegmentSizes" in content:
            fixed_lines = []
            for line in content.split("\n"):
                if "operandSegmentSizes" in line and any(op in line for op in fix_ops):
                    line = array_re.sub(strip_trailing_zeros, line)
                fixed_lines.append(line)
            content = "\n".join(fixed_lines)
        return original(cmd, content, *args)

    run_command._operand_segments_patched = True
    toolkit._run_command = run_command


# _patch_enflame_operand_segments()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


@triton.jit
def _row_kernel(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    output,
    sm_scale,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    batch_id = tl.program_id(0)
    query_head = tl.program_id(1)
    kv_head = query_head // GROUP_SIZE
    d_offsets = tl.arange(0, BLOCK_D)
    dv_offsets = tl.arange(0, BLOCK_DV)
    q_values = tl.load(
        q + (batch_id * H_Q + query_head) * D + d_offsets,
        mask=d_offsets < D,
        other=0.0,
    ).to(tl.float32)
    start = tl.load(kv_indptr + batch_id)
    end = tl.load(kv_indptr + batch_id + 1)
    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    for position in tl.range(start, end):
        page = tl.load(kv_indices + position)
        k_values = tl.load(
            k_buffer + (page * H_KV + kv_head) * D + d_offsets,
            mask=d_offsets < D,
            other=0.0,
        ).to(tl.float32)
        score = tl.sum(q_values * k_values, axis=0) * sm_scale
        next_max = tl.maximum(running_max, score)
        old_scale = tl.exp(running_max - next_max)
        probability = tl.exp(score - next_max)
        v_values = tl.load(
            v_buffer + (page * H_KV + kv_head) * D_V + dv_offsets,
            mask=dv_offsets < D_V,
            other=0.0,
        ).to(tl.float32)
        accumulator = accumulator * old_scale + probability * v_values
        running_sum = running_sum * old_scale + probability
        running_max = next_max
    output_offsets = (batch_id * H_Q + query_head) * D_V + dv_offsets
    tl.store(
        output + output_offsets,
        accumulator / running_sum,
        mask=dv_offsets < D_V,
    )


@triton.jit
def _flash_kernel(
    q,
    k_buffer,
    v_buffer,
    kv_indptr,
    kv_indices,
    output,
    sm_scale,
    head_size_rt,
    value_size_rt,
    sq_b,
    sq_hi,
    sq_lo,
    sq_d,
    sk0,
    sk_h,
    sk_d,
    sv0,
    sv_h,
    sv_d,
    so_b,
    so_hi,
    so_lo,
    so_d,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    # Load-form notes (2026-11-12 second-round optimization, gcu410):
    #  - The 2026-11-08 //64/%64 dual-unknown-term page decomposition and the
    #    [BLOCK_D, BLOCK_N] (column-major) K orientation were workarounds for
    #    the 11-07 toolchain-skew compile failures, later shown to be shim
    #    artifacts (P01a resolved 2026-11-10, raw pipeline self-consistent).
    #  - K is now loaded row-major ([BLOCK_N, BLOCK_D], one 256B contiguous
    #    row per page, single runtime page term) and transposed in-register
    #    for the QK^T dot: the column-major tile forced per-lane 2B gathers
    #    (page jumps between adjacent lanes) and cost ~40% of the kernel.
    #  - The V tile load is issued right after the pages load (before the K
    #    load / dot chain) so both gather latencies overlap with the QK dot
    #    and softmax math; the scheduler does not hoist it on its own.
    #  - BLOCK_N must stay <= 64 with masks/wheres present: uint16-view tiles
    #    with 128-lane non-uniform bools still trigger the ld.lld relocation
    #    failure (compiler_issues/ld_lld_relocation.md, P03). Maskless
    #    full-block loops and [128, 128] trans tiles were both measured
    #    slower or wrong (2026-11-12 variant matrix, see case log).
    #  - num_warps=1 remains the sweet spot after the structural change
    #    (nw2: +19x at B=4096; nw4: +11x); num_stages is neutral (1 kept).
    batch_id = tl.program_id(0)
    query_block = tl.program_id(1)
    kv_head = tl.program_id(2)
    query_offsets = (
        kv_head * GROUP_SIZE + query_block * BLOCK_H + tl.arange(0, BLOCK_H)
    )
    q_hi = query_offsets // 8
    q_lo = query_offsets % 8
    d_offsets = tl.arange(0, BLOCK_D)
    dv_offsets = tl.arange(0, BLOCK_DV)
    d_mask = d_offsets < head_size_rt
    dv_mask = dv_offsets < value_size_rt
    query_mask = query_offsets < (kv_head + 1) * GROUP_SIZE
    q_values = tl.load(
        q
        + batch_id * sq_b
        + q_hi[:, None] * sq_hi
        + q_lo[:, None] * sq_lo
        + d_offsets[None, :] * sq_d,
        mask=query_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    running_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)
    start = tl.load(kv_indptr + batch_id)
    end = tl.load(kv_indptr + batch_id + 1)
    num_blocks = tl.cdiv(end - start, BLOCK_N)
    for block_idx in range(0, num_blocks):
        sequence_offsets = start + block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        sequence_mask = sequence_offsets < end
        pages = tl.load(kv_indices + sequence_offsets, mask=sequence_mask, other=0)
        v_values = tl.load(
            v_buffer
            + pages[:, None] * sv0
            + kv_head * sv_h
            + dv_offsets[None, :] * sv_d,
            mask=sequence_mask[:, None] & dv_mask[None, :],
            other=0,
        ).to(tl.bfloat16, bitcast=True)
        k_values = tl.trans(
            tl.load(
                k_buffer
                + pages[:, None] * sk0
                + kv_head * sk_h
                + d_offsets[None, :] * sk_d,
                mask=sequence_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.bfloat16, bitcast=True)
        )
        scores = tl.dot(q_values, k_values) * sm_scale
        scores = tl.where(
            query_mask[:, None] & sequence_mask[None, :], scores, -float("inf")
        )
        block_max = tl.max(scores, axis=1)
        next_max = tl.where(query_mask, tl.maximum(running_max, block_max), 0.0)
        old_scale = tl.where(
            query_mask,
            tl.exp2((running_max - next_max) * 1.4426950408889634),
            0.0,
        )
        probabilities = tl.where(
            query_mask[:, None] & sequence_mask[None, :],
            tl.exp2((scores - next_max[:, None]) * 1.4426950408889634),
            0.0,
        )
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            probabilities.to(tl.bfloat16), v_values
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        running_max = next_max
    tl.store(
        output
        + batch_id * so_b
        + q_hi[:, None] * so_hi
        + q_lo[:, None] * so_lo
        + dv_offsets[None, :] * so_d,
        accumulator / running_sum[:, None],
        mask=query_mask[:, None] & dv_mask[None, :],
    )


def decode_grouped_attention(
    q, k_buffer, v_buffer, kv_indptr, kv_indices, sm_scale
):
    batch_size, query_heads, head_size = q.shape
    kv_heads = k_buffer.shape[1]
    value_size = v_buffer.shape[2]
    group_size = query_heads // kv_heads
    output = torch.empty(
        (batch_size, query_heads, value_size),
        dtype=torch.float32,
        device=q.device,
    )
    fast_path = (
        q.dtype == torch.bfloat16
        and k_buffer.dtype == torch.bfloat16
        and v_buffer.dtype == torch.bfloat16
        and query_heads == 128
        and kv_heads == 1
        and head_size == 128
        and value_size == 128
        and q.is_contiguous()
        and k_buffer.is_contiguous()
        and v_buffer.is_contiguous()
    )
    if fast_path:
        # uint16 views of the contiguous bf16 caches (same itemsize, strides
        # preserved): keeps the masked-gather lowering stable for the tiles
        # feeding the dots (2026-11-12: direct bf16 measured identical, the
        # views are kept as the proven form).
        k_arg = k_buffer.view(torch.uint16)
        v_arg = v_buffer.view(torch.uint16)
        # 2026-11-12 scan on gcu410 (5 core shapes, after the K row-major +
        # tl.trans + V-hoist restructure): num_warps=1 dominates (nw2 +19x at
        # B=4096, nw4 +11x); num_stages is neutral (1 kept); BLOCK_N=64 stays
        # the ceiling set by the uint16-view ld.lld trap (P03) and beats 32;
        # BLOCK_H=128 keeps single-pass K/V per batch (64/32 regress B>=64
        # via re-reads).
        block_h, block_n, num_warps = 128, 64, 1
        _flash_kernel[
            (batch_size, triton.cdiv(group_size, block_h), kv_heads)
        ](
            q,
            k_arg,
            v_arg,
            kv_indptr,
            kv_indices,
            output,
            sm_scale,
            head_size,
            value_size,
            q.stride(0),
            8 * q.stride(1),
            q.stride(1),
            q.stride(2),
            k_buffer.stride(0),
            k_buffer.stride(1),
            k_buffer.stride(2),
            v_buffer.stride(0),
            v_buffer.stride(1),
            v_buffer.stride(2),
            output.stride(0),
            8 * output.stride(1),
            output.stride(1),
            output.stride(2),
            query_heads,
            kv_heads,
            head_size,
            value_size,
            group_size,
            block_h,
            block_n,
            triton.next_power_of_2(head_size),
            triton.next_power_of_2(value_size),
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        _row_kernel[(batch_size, query_heads)](
            q,
            k_buffer,
            v_buffer,
            kv_indptr,
            kv_indices,
            output,
            sm_scale,
            query_heads,
            kv_heads,
            head_size,
            value_size,
            group_size,
            triton.next_power_of_2(head_size),
            triton.next_power_of_2(value_size),
            num_warps=8 if max(head_size, value_size) >= 256 else 4,
            num_stages=1,
        )
    return output


__all__ = ["decode_grouped_attention"]
