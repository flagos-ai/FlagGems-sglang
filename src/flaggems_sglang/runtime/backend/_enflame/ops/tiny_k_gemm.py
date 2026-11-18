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

"""GCU rewrite of gemm/tiny_k_gemm: ``out[m, n] = sum_k x[m, k] * w[n, k]``.

Probe ladder on gcu400 (do_bench, cold-L2, warmup 1000 ms / iter 100 ms):

* empty-kernel launch floor: ~2.5 us at grid <= 24 (= SIP count), ~3.1 us at
  grid 48, ~5.9 us at grid 96 — wave quantization dominates tiny grids, so
  the config targets exactly one 24-program wave;
* w-load time is the whole "work" budget (dot is free on top of it); the
  natural-layout load and the transposed-affine load cost the same, so the
  kernel keeps the natural ``[BLOCK_N, BLOCK_K]`` load (K-contiguous rows)
  and transposes in-register — this also avoids the "K-contiguous affine
  transposed B operand fed to tl.dot" silent-wrong-value trap domain
  (compiler_traps.md), which the previous affine b-load sat in;
* an elementwise-reduce form (no tl.dot) measured 19-52 us at M=8 — vetoed.

Measured (vs the previous BN64/BK128/nw2/stages3 config, bit-identical
outputs on every probed shape):

  1..16 x 1536 x 128 bf16:  4.73 -> 4.71 us   (single-wave floor, ~flat)
  1..16 x 3072 x 128 bf16:  5.70 -> 5.34 us   (-6.4%, grid 48 -> 24)
  8 x 2048 x 256 fp32 out:  6.65 -> 5.62 us   (-16%, K=256 in one BLOCK_K)
  16 x 1536 x 256 bf16:     6.39 -> 5.37 us   (-16%)

The k-loop is a ``tl.static_range`` unroll (runtime loops around tl.dot with
>= 2 iterations are a known miscompile domain), and the grid gains an
M-block axis so M > 16 no longer silently drops rows.
"""

import re

import torch
import triton
import triton.language as tl

# ===========================================================================
# GCU compiler-toolchain dialect compatibility shim (same as fused_moe_gemm /
# per_group_transpose / chunk_scaled_dot_kkt: the 2026-11-07 20:22 stack
# emits operand layouts the external gcu-compiler-opt rejects for 2-D
# accesses; see compiler_issues/sglang_fused_moe_gemm.md).
# ===========================================================================

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


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-11 重放剥离: 私有 GCUIR shim 剥尾零
# 误伤已修复工具链（fresh 编译实测 slice_pad_start 7 vs 8 全挂，同 bmm_chunk 重放
# 6 vs 7 实证）；现行策略 = 禁用全部 op 私有 shim（5dc28b9），剥离后 2D load/
# tl.dot 原生可编译。注释保留现场。


@triton.jit
def _tiny_k_dot(
    x,
    w,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sxm,
    sxk,
    swn,
    swk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        ks = k0 + kk
        a = tl.load(
            x + rows[:, None] * sxm + ks[None, :] * sxk,
            mask=(rows[:, None] < M) & (ks[None, :] < K),
            other=0.0,
        )
        wv = tl.load(
            w + cols[:, None] * swn + ks[None, :] * swk,
            mask=(cols[:, None] < N) & (ks[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(wv))
    tl.store(
        out + rows[:, None] * N + cols[None, :],
        acc,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def tiny_k_gemm(x: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype):
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    if m == 0 or n == 0:
        return out
    # One 24-program wave (24 SIPs): 1536/64 = 24, 3072/128 = 24.
    block_n = 128 if n >= 2048 else 64
    # K <= 256 fits a single unrolled dot; a runtime k-loop around tl.dot
    # with >= 2 iterations is a known miscompile domain on this backend.
    block_k = min(max(_next_pow2(k), 128), 256)
    _tiny_k_dot[(triton.cdiv(n, block_n), triton.cdiv(m, 16))](
        x,
        w,
        out,
        M=m,
        N=n,
        K=k,
        sxm=x.stride(0),
        sxk=x.stride(1),
        swn=w.stride(0),
        swk=w.stride(1),
        BLOCK_M=16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["tiny_k_gemm"]
