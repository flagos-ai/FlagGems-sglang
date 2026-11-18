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

import re

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# GCU dialect shim (ported verbatim from _enflame/ops/extend_attention.py /
# fused_moe_gemm.py, also production-verified on the vllm side at d1a7005).
#
# The 2026-11-07 20:22 FlagTree wheel emits `memref_ext.*` / `gcu.matmul`
# ops with trailing zero operand segments that the external
# `gcu-compiler-opt` (2026-10-20) rejects on fresh JIT compiles
# (`slice_start/slice_pad_start/slice_transpose_start ... size mismatch in
# attribute conversion`). Carrying the shim inside this module makes the op
# self-contained instead of relying on a sibling op file being imported
# first. It is idempotent and safe alongside the copies in other modules.
# ---------------------------------------------------------------------------


_GCU_SENTINEL = "-9223372036854775808"


def _fix_gcu_attr(attr_match, line, op_name):
    inner = attr_match.group(1)
    seg_match = re.search(r"operandSegmentSizes = array<i32: ([^>]*)>", inner)
    if op_name == "memref_ext.memset_start":
        return ""
    if op_name.startswith("memref_ext.") or op_name == "gcu.matmul":
        if seg_match:
            segs = [s.strip() for s in seg_match.group(1).split(",")]
            while len(segs) > 1 and segs[-1] == "0":
                segs.pop()
            new_seg = "array<i32: " + ", ".join(segs) + ">"
            rest = re.sub(r"operandSegmentSizes = array<i32:[^>]*>", "", inner)
            rest = rest.strip().strip(",").strip()
            if rest:
                return f" <{{operandSegmentSizes = {new_seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {new_seg}}}>"
        return attr_match.group(0)
    if (
        op_name in ("memref.reinterpret_cast", "memref.subview")
        and not seg_match
    ):
        opers = re.search(r"\(([^()]*)\)\s*<\{", line)
        n_ops = (
            len([x for x in opers.group(1).split(",") if x.strip()])
            if opers
            else 1
        )
        offs = re.search(r"static_offsets = array<i64: ([^>]*)>", line)
        sizes = re.search(r"static_sizes = array<i64: ([^>]*)>", line)
        strides = re.search(r"static_strides = array<i64: ([^>]*)>", line)
        n_off = offs.group(1).count(_GCU_SENTINEL) if offs else 0
        n_size = sizes.group(1).count(_GCU_SENTINEL) if sizes else 0
        n_stride = strides.group(1).count(_GCU_SENTINEL) if strides else 0
        if 1 + n_off + n_size + n_stride == n_ops:
            seg = f"array<i32: 1, {n_off}, {n_size}, {n_stride}"
            rest = inner.strip()
            if rest:
                return f" <{{operandSegmentSizes = {seg}, {rest}}}>"
            return f" <{{operandSegmentSizes = {seg}}}>"
    return attr_match.group(0)


def _fix_gcu_dialect_line(line):
    op_match = re.search(r'"([a-z0-9_.]+)"\(', line)
    if not op_match:
        return line
    op_name = op_match.group(1)
    if not (
        op_name.startswith("memref_ext.")
        or op_name.startswith("gcu.")
        or op_name in ("memref.reinterpret_cast", "memref.subview")
    ):
        return line
    return re.sub(
        r" <\{([^}]*)\}>",
        lambda m: _fix_gcu_attr(m, line, op_name),
        line,
        count=1,
    )


def _patch_gcu_compiler_opt():
    import triton.backends.enflame.toolkit as _toolkit

    if getattr(_toolkit.gcu_compiler_opt, "_flaggems_dialect_shim", False):
        return
    _orig = _toolkit.gcu_compiler_opt

    def _shim(content, *args):
        if isinstance(content, str) and "operandSegmentSizes" in content:
            content = "\n".join(
                _fix_gcu_dialect_line(line) for line in content.split("\n")
            )
        return _orig(content, *args)

    _shim._flaggems_dialect_shim = True
    _toolkit.gcu_compiler_opt = _shim


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-11 重放剥离: 私有 GCUIR shim 剥尾零误伤已修复工具链
# (fresh 编译 memref_ext.slice_start '6 vs 7' 签名复现,P01a 定案);现行策略=禁用(同 qkv_lora_b.py 等)


def _downgrade_gcuir_for_external_tool(text):
    out_lines = []
    for line in text.splitlines():
        line = _fix_gcu_dialect_line(line)
        if '"gcu.matmul"' in line:
            line = re.sub(
                r"operandSegmentSizes = array<i32: (\d+(?:, \d+){3})>",
                r"operandSegmentSizes = array<i32: \1, 0>",
                line,
            )
            line = re.sub(r"inputPrecision = \d+ : i32, ", "", line)
            line = re.sub(r", inputPrecision = \d+ : i32", "", line)
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _patch_gcu_matmul_compat():
    import triton.backends.enflame.toolkit as _toolkit

    if getattr(
        _toolkit.gcu_compiler_opt, "_flaggems_kda_matmul_compat", False
    ):
        return
    _prev = _toolkit.gcu_compiler_opt

    def _compat(content, *args):
        try:
            return _prev(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if '"gcu.matmul"' not in text or "gcu.matmul" not in str(
                first_error
            ):
                raise first_error
            try:
                return _toolkit._run_command(
                    "gcu-compiler-opt",
                    _downgrade_gcuir_for_external_tool(text),
                    *args,
                )
            except Exception:
                raise first_error

    _compat._flaggems_kda_matmul_compat = True
    _toolkit.gcu_compiler_opt = _compat


# _patch_gcu_matmul_compat()  # DISABLED 2026-11-11: 同上,私有 GCUIR shim 禁用


@triton.jit
def _bmm_chunk_kernel(
    a,
    b,
    output,
    SEQLEN: tl.constexpr,
    NGROUPS: tl.constexpr,
    K: tl.constexpr,
    NCHUNKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    batch_chunk_group = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)
    group = batch_chunk_group % NGROUPS
    batch_chunk = batch_chunk_group // NGROUPS
    chunk = batch_chunk % NCHUNKS
    batch = batch_chunk // NCHUNKS
    rows_m = tile_m * 256 + tl.arange(0, 256)
    rows_n = tile_n * 256 + tl.arange(0, 256)
    offsets_k = tl.arange(0, 64)
    sequence_m = chunk * CHUNK_SIZE + rows_m
    sequence_n = chunk * CHUNK_SIZE + rows_n
    accumulator = tl.zeros((256, 256), tl.float32)

    for k_start in range(0, K, 64):
        current_k = k_start + offsets_k
        offsets_a = (
            (batch * SEQLEN + sequence_m[:, None]) * NGROUPS * K
            + group * K
            + current_k[None, :]
        )
        offsets_b = (
            (batch * SEQLEN + sequence_n[None, :]) * NGROUPS * K
            + group * K
            + current_k[:, None]
        )
        mask_a = (rows_m[:, None] < CHUNK_SIZE) & (current_k[None, :] < K)
        mask_b = (current_k[:, None] < K) & (rows_n[None, :] < CHUNK_SIZE)
        values_a = tl.load(a + offsets_a, mask=mask_a, other=0.0)
        values_b = tl.load(b + offsets_b, mask=mask_b, other=0.0)
        accumulator = tl.dot(values_a, values_b, accumulator)

    output_base = batch_chunk_group * CHUNK_SIZE * CHUNK_SIZE
    offsets_output = (
        output_base + rows_m[:, None] * CHUNK_SIZE + rows_n[None, :]
    )
    output_mask = (rows_m[:, None] < CHUNK_SIZE) & (
        rows_n[None, :] < CHUNK_SIZE
    )
    tl.store(output + offsets_output, accumulator, mask=output_mask)


@triton.jit
def _bmm_chunk_pair_kernel(
    a,
    b,
    output,
    SEQLEN: tl.constexpr,
    NGROUPS: tl.constexpr,
    K: tl.constexpr,
    NCHUNKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    # Each program computes two consecutive (batch, chunk, group) units.
    # The second unit's tiles are loaded before the first unit's dot so the
    # DMA staging overlaps the MMA (GCU dot path serializes load->wait->dot;
    # pairing restores the overlap without extra kernel launches).
    # Exact-fit fast path: CHUNK_SIZE == 256 and K == 64 (no masks needed).
    pid = tl.program_id(0)
    rows = tl.arange(0, 256)
    offsets_k = tl.arange(0, 64)

    unit = pid * 2
    group = unit % NGROUPS
    batch_chunk = unit // NGROUPS
    chunk = batch_chunk % NCHUNKS
    batch = batch_chunk // NCHUNKS
    sequence = chunk * CHUNK_SIZE + rows
    base_m = (batch * SEQLEN + sequence[:, None]) * NGROUPS * K + group * K
    base_n = (batch * SEQLEN + sequence[None, :]) * NGROUPS * K + group * K
    values_a = tl.load(a + base_m + offsets_k[None, :])
    values_b = tl.load(b + base_n + offsets_k[:, None])

    unit2 = pid * 2 + 1
    group2 = unit2 % NGROUPS
    batch_chunk2 = unit2 // NGROUPS
    chunk2 = batch_chunk2 % NCHUNKS
    batch2 = batch_chunk2 // NCHUNKS
    sequence2 = chunk2 * CHUNK_SIZE + rows
    base_m2 = (batch2 * SEQLEN + sequence2[:, None]) * NGROUPS * K + group2 * K
    base_n2 = (batch2 * SEQLEN + sequence2[None, :]) * NGROUPS * K + group2 * K
    values_a2 = tl.load(a + base_m2 + offsets_k[None, :])
    values_b2 = tl.load(b + base_n2 + offsets_k[:, None])

    accumulator = tl.dot(values_a, values_b)
    output_base = unit * CHUNK_SIZE * CHUNK_SIZE
    offsets_output = output_base + rows[:, None] * CHUNK_SIZE + rows[None, :]
    tl.store(output + offsets_output, accumulator)

    accumulator2 = tl.dot(values_a2, values_b2)
    output_base2 = unit2 * CHUNK_SIZE * CHUNK_SIZE
    offsets_output2 = output_base2 + rows[:, None] * CHUNK_SIZE + rows[None, :]
    tl.store(output + offsets_output2, accumulator2)


def bmm_chunk(a, b, chunk_size, causal=False):
    batch, seqlen, ngroups, k = a.shape
    nchunks = seqlen // chunk_size
    output = torch.empty(
        (batch, nchunks, ngroups, chunk_size, chunk_size),
        device=a.device,
        dtype=a.dtype,
    )
    units = batch * nchunks * ngroups
    if chunk_size == 256 and k == 64 and units % 2 == 0:
        _bmm_chunk_pair_kernel[(units // 2, 1, 1)](
            a,
            b,
            output,
            SEQLEN=seqlen,
            NGROUPS=ngroups,
            K=k,
            NCHUNKS=nchunks,
            CHUNK_SIZE=chunk_size,
            num_warps=4,
            num_stages=1,
        )
    else:
        tiles = triton.cdiv(chunk_size, 256)
        _bmm_chunk_kernel[(units, tiles, tiles)](
            a,
            b,
            output,
            SEQLEN=seqlen,
            NGROUPS=ngroups,
            K=k,
            NCHUNKS=nchunks,
            CHUNK_SIZE=chunk_size,
            num_warps=4,
            num_stages=1,
        )
    return output


__all__ = ["bmm_chunk"]
