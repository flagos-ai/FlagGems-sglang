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

"""GCU rewrite of quantization/per_group_transpose.

``out`` re-lays out each expert's row group ``[start:end)`` of a contiguous
``(M, K)`` tensor as its transpose: the flat segment
``[start*K, end*K)`` of the output holds ``a[start:end].t()`` flattened.

The stock kernel (one program per (expert, 32-wide K block) looping over
32-row blocks with a row-major ``[32, 32]`` masked load + ``tl.trans`` +
masked store) runs at 20-25 GB/s effective bandwidth on gcu400 -- per-tile
iteration cost is ~4 us regardless of tile size, so the tiny 2 KB tiles make
the kernel op-count bound (2692 us at M=8192/K=4096 vs a 50 us flat-copy
floor). The stock source also no longer freshly compiles on the 2026-11-07
20:22 toolchain (``memref_ext.*`` ``operandSegmentSizes`` size mismatch; the
baseline binaries are old-cache hits), so this module embeds the same dialect
shim used by ``fused_moe_gemm.py``.

The rewrite keeps the single-launch contract and changes three things:

* **Transposed-affine load**: the input tile is loaded directly in
  ``[BLOCK_K, BLOCK_M]`` orientation (``row_offsets[None, :] * K +
  k_offsets[:, None]``), so the DMA transpose happens inside the load path and
  the store is a plain row-major ``[BLOCK_K, BLOCK_M]`` tile with the
  contiguous fast axis. Measured at the same cost as an identical-shape pure
  copy (the in-register ``tl.trans`` variant costs +40-70% on top of it).
* **Big tiles**: (BM, BK) from a dispatch table on host-visible metadata only
  (``K``, ``M // num_experts``), ranging from ``[64, 1024]`` for skinny
  groups to ``[512, 512]`` for large ones; the compiled DSM requirement is
  ~1.75x ``BM*BK*itemsize`` against the 917504 B hardware limit, so 4-byte
  dtypes shrink the tile through the same byte budget (<= 512 KiB per tile).
* **Row-split grid**: a third grid dimension strides the per-group row loop
  (``row_start = split*BM`` step ``NSPLIT*BM``) so small-K shapes still fill
  all 48 SPs; ragged/empty groups stay correct because every program derives
  its own bounds from ``expert_offsets``.

Measured (gcu400, bench protocol warmup 1000 ms / iter 100 ms, kernel mode,
bf16, 8 experts, M = 8*rows_per_expert):

  K, rows/expert   stock (old cache)   round-1 rewrite   2026-11-12 re-sweep
  128,    16              7.8 us           4.7 us            4.7 us
  128,   128             16.8 us           5.0 us            4.8 us
  128,  1024            104.1 us           6.7 us            6.5 us
  512,    16             19.3 us           4.8 us            4.8 us
  512,   128             51.6 us           6.6 us            5.8 us
  512,  1024            359.5 us          13.6 us           12.8 us
  4096,   16            124.7 us           7.3 us            7.0 us
  4096,  128            361.7 us          15.2 us           12.8 us  (dispatch fix)
  4096,  1024          2692.3 us          66.2 us           62.9 us  (dispatch fix)

The 2026-11-12 second-round sweep (job j20261112222404_d9d0193a: 5-round
same-process interleaved do_bench, fresh input per variant, DSM size read
from the compiled llir) re-swept tile/warp/split/mirror/num-stages variants
per shape. Outcome: every round-1 cell is optimal or within noise EXCEPT the
K >= 4096 column with avg_rows >= 128, where wide loads with two warps win
big -- (BLOCK_M, BLOCK_K, num_warps) = (128, 1024, 2) beats (128, 512, 2) /
(512, 512, 4) by 14.2% / 4.9% at rows=128 / 1024 (K=4096) and 4.6% (K=8192,
rows=128); for avg_rows >= 512 and K >= 8192 (256, 1024, 2) wins by 1-2%.
Round-1 had missed this because its small-BLOCK_M probes on large shapes all
carried row-split NSPLIT > 1, whose 2/1/1-per-split tile imbalance loses
badly; NSPLIT=1 + wide BLOCK_K + num_warps=2 is the winning combination.
K = 2048 keeps the round-1 tiles: its (256, 1024, 2) s1 variant only wins at
avg_rows >= 512 (-6% at rows=1024) but the host SP-fill logic launches it
with NSPLIT=2/3 there, which gives the gain back (verified 2026-11-12).
Store-run width (BLOCK_M * itemsize) remains the dominant cost on the big
uniform shapes: [512, 512] bf16 compiles to exactly 524288 B of DSM (> 448
KB, 24-way) yet still beats 48-way [256, 512] by 3%, and shrinking BLOCK_M
below 512 at K=4096/rows=1024 costs 2x once stores drop under 256 B runs.
"""

import re

import torch
import triton
import triton.language as tl

# ===========================================================================
# GCU compiler-toolchain dialect compatibility shim
# ===========================================================================
# The installed FlagTree wheel emits MLIR where several `memref_ext.*` /
# `gcu.matmul` ops carry a trailing zero operand segment, while the external
# `gcu-compiler-opt` tool shipped in /opt/triton_gcu still registers the older
# operand layouts. The mismatch makes every kernel with 2-D tensor accesses
# fail to compile with "operandSegmentSizes ... size mismatch" (first hit for
# this op: the stock kernel's masked 2-D load; see
# compiler_issues/sglang_fused_moe_gemm.md). The shim below rewrites the
# generated MLIR text to the layout the external tool accepts (ported verbatim
# from fused_moe_gemm.py, which ported it from the vllm enflame backend,
# router_gemm commit d1a7005, where it is production-verified). It is
# idempotent and leaves already-compatible IR (e.g. pure 1-D kernels)
# untouched.

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


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


@triton.jit
def _per_group_transpose_kernel(
    input_ptr,
    output_ptr,
    expert_offsets_ptr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    NSPLIT: tl.constexpr,
):
    expert = tl.program_id(0)
    k_block = tl.program_id(1)
    split = tl.program_id(2)

    group_start = tl.load(expert_offsets_ptr + expert)
    group_end = tl.load(expert_offsets_ptr + expert + 1)
    group_rows = group_end - group_start
    group_base = group_start * K

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    if EVEN_K:
        for row_start in tl.range(
            split * BLOCK_M, group_rows, NSPLIT * BLOCK_M
        ):
            row_offsets = row_start + tl.arange(0, BLOCK_M)
            row_mask = row_offsets < group_rows
            values = tl.load(
                input_ptr
                + group_base
                + row_offsets[None, :] * K
                + k_offsets[:, None],
                mask=row_mask[None, :],
                other=0,
            )
            tl.store(
                output_ptr
                + group_base
                + k_offsets[:, None] * group_rows
                + row_offsets[None, :],
                values,
                mask=row_mask[None, :],
            )
    else:
        k_mask = k_offsets < K
        for row_start in tl.range(
            split * BLOCK_M, group_rows, NSPLIT * BLOCK_M
        ):
            row_offsets = row_start + tl.arange(0, BLOCK_M)
            row_mask = row_offsets < group_rows
            values = tl.load(
                input_ptr
                + group_base
                + row_offsets[None, :] * K
                + k_offsets[:, None],
                mask=row_mask[None, :] & k_mask[:, None],
                other=0,
            )
            tl.store(
                output_ptr
                + group_base
                + k_offsets[:, None] * group_rows
                + row_offsets[None, :],
                values,
                mask=k_mask[:, None] & row_mask[None, :],
            )


# Dispatch table keyed on host-visible metadata only (no device reads):
# avg_rows = M // num_experts and K. K classes: 0 = K < 256, 1 = 256 <= K < 2048,
# 2 = 2048 <= K < 4096, 3 = 4096 <= K < 8192, 4 = K >= 8192. Cell values are
# (BLOCK_M, BLOCK_K, num_warps). The K >= 4096 cells with avg_rows >= 128 were
# re-swept 2026-11-12 (see module docstring): wide loads (BLOCK_K = 1024) with
# num_warps = 2 win by 2-14%; every other cell was re-verified optimal or
# within noise, so it keeps its round-1 value.
_DISPATCH = {
    # avg_rows >= 512 (deep, uniform groups)
    (True, 4): (256, 1024, 2),  # K >= 8192
    (True, 3): (128, 1024, 2),  # 4096 <= K < 8192
    (True, 2): (512, 512, 4),  # 2048 <= K < 4096
    (True, 1): (256, 256, 8),  # 256 <= K < 2048
    (True, 0): (512, 128, 4),  # K < 256
    # 128 <= avg_rows < 512
    (False, 4): (128, 1024, 2),  # K >= 8192
    (False, 3): (128, 1024, 2),  # 4096 <= K < 8192
    (False, 2): (128, 512, 2),  # 2048 <= K < 4096
    (False, 1): (256, 256, 4),  # 256 <= K < 2048
    (False, 0): (256, 128, 4),  # K < 256
}
# avg_rows < 128 (skinny groups): match BLOCK_M to the group height, widen the
# load instead. Keyed by the same K classes.
_DISPATCH_SKINNY = {
    4: (64, 1024, 4),  # K >= 8192
    3: (64, 1024, 4),  # 4096 <= K < 8192
    2: (64, 1024, 4),  # 2048 <= K < 4096
    1: (64, 512, 4),  # 256 <= K < 2048
    0: (256, 128, 4),  # K < 256
}

# Compiled DSM requirement is ~1.75x BM*BK*itemsize against the 917504 B
# hardware limit; bf16 [512, 512] sits exactly at the limit and passes, so
# the tile byte budget is 512 KiB (4-byte dtypes shrink through it).
_TILE_BYTE_BUDGET = 524288
_NUM_SPS = 48


def _next_pow2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pick_config(k, m, num_experts, itemsize):
    avg_rows = max(1, m // max(1, num_experts))
    if k >= 8192:
        k_class = 4
    elif k >= 4096:
        k_class = 3
    elif k >= 2048:
        k_class = 2
    elif k >= 256:
        k_class = 1
    else:
        k_class = 0
    if avg_rows >= 512:
        block_m, block_k, num_warps = _DISPATCH[(True, k_class)]
    elif avg_rows >= 128:
        block_m, block_k, num_warps = _DISPATCH[(False, k_class)]
    else:
        block_m, block_k, num_warps = _DISPATCH_SKINNY[k_class]
    # Never let BLOCK_K exceed K's own pow2 (tiny-K shapes take the masked
    # arm otherwise, wasting a whole tile); floor at 2 to avoid width-1 tiles.
    block_k = min(block_k, max(2, _next_pow2(k)))
    while block_m * block_k * itemsize > _TILE_BYTE_BUDGET:
        if block_m > 64:
            block_m //= 2
        elif block_k > 128:
            block_k //= 2
        else:
            break
    return block_m, block_k, num_warps


def per_group_transpose(a, expert_offsets, m_alignment=1):
    _, k = a.shape
    num_experts = expert_offsets.numel() - 1
    output = torch.empty_like(a)
    if num_experts <= 0:
        return output

    block_m, block_k, num_warps = _pick_config(
        k, a.shape[0], num_experts, a.element_size()
    )
    k_blocks = triton.cdiv(k, block_k)
    # Row-split so small-K shapes still fill the machine; never split past
    # one block per group-height (extra programs would idle).
    nsplit = 1
    programs = num_experts * k_blocks
    if programs < _NUM_SPS:
        nsplit = min(
            max(1, _NUM_SPS // programs),
            max(1, a.shape[0] // num_experts // block_m),
        )
    grid = (num_experts, k_blocks, nsplit)
    _per_group_transpose_kernel[grid](
        a,
        output,
        expert_offsets,
        K=k,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        EVEN_K=(k % block_k == 0),
        NSPLIT=nsplit,
        num_warps=num_warps,
    )
    return output


__all__ = ["per_group_transpose"]
