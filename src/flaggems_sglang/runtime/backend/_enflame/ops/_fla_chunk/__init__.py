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

"""GCU chunked gated-delta-rule forward pipeline.

Ported from FlagGems-vllm ``_enflame/ops/FLA`` (dense path verified on GCU):
fused cumsum+KKT+solve_tril, recompute_w_u, chunked state recurrence (h) and
output (o).  Only the dense (non-cu_seqlens) path is used by
``chunk_gated_delta_rule_fwd``; the varlen branches are kept for source parity
with the upstream kernels but are not dispatched to on GCU.
"""

# ===========================================================================
# GCU compiler-toolchain compatibility shim (same family as
# compiler_issues/vllm_chunk_gated_delta_rule_fwd.md; ported from
# FlagGems-vllm ``_enflame/ops/chunk_gated_delta_rule.py``)
# ===========================================================================
# The 2026-11-07 20:22 in-process pass library emits GCUIR whose `gcu.matmul`
# (and `memref_ext.slice_*`) ops carry a trailing empty variadic segment in
# operandSegmentSizes plus an inputPrecision property. The legacy external
# gcu-compiler-opt rejects every derived encoding, most notably as
# "size mismatch in attribute conversion: 4 vs 5" / "7 vs 6" / "8 vs 7" once
# other chain shims strip the trailing zeros. This pipeline's kernels
# (tl.cumsum, 2D block_ptr tiles, broadcast arithmetic over [BT, BT] tiles)
# cannot fresh-compile on GCU until the toolchain re-syncs. The rewrite below
# is fallback-only and scoped to this op family's kernel names, so
# already-compilable kernels keep their exact code path. Remove the shim once
# the external tool re-syncs with the in-process passes.

_CHUNK_GDR_COMPAT_KERNELS = (
    "chunk_gated_delta_rule_fused_cumsum_kkt_solve_tril_kernel",
    "recompute_w_u_fwd_kernel",
    "recompute_w_u_fwd_tle_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
    "_chunk_gated_delta_rule_fused_tail_vblock_kernel",
)


def _chunk_gdr_downgrade_gcuir_for_external_tool(text):
    """Rewrite new-format GCUIR op encodings back to the old external-tool form."""
    import re

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


def _install_chunk_gdr_gcuir_compat():
    """Wrap enflame toolkit.gcu_compiler_opt with a scoped downgrade retry."""
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_chunk_gdr_compat", False):
        return
    _prev = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return _prev(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if not any(name in text for name in _CHUNK_GDR_COMPAT_KERNELS):
                raise first_error
            try:
                # Bypass the wrapper chain: inner shims would strip the
                # trailing zero segment that the legacy tool requires.
                return toolkit._run_command(
                    "gcu-compiler-opt",
                    _chunk_gdr_downgrade_gcuir_for_external_tool(text),
                    "-mlir-print-op-generic",
                    *args,
                )
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._chunk_gdr_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _install_chunk_gdr_gcuir_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归(降级重试路径
# 搞挂合法 kernel,同 conftest shim 事故);详见 commit b333be6/e1acdf6 与
# compiler_issues/vllm_memref_slice_pad_start.md。编译器修复后统一处置。
