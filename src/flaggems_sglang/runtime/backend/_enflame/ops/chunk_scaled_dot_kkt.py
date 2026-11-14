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

import re

import torch
import triton
import triton.language as tl

# The FlagTree wheel (0.7.0+enflame.git72408c41, 2026-11-07 20:22 build) emits
# MLIR that the external `gcu-compiler-opt` tool does not accept: `memref_ext.*`
# / `gcu.matmul` ops carry a trailing zero operand segment the older tool did
# not register, so every kernel with 2-D tensor access fails first-compile with
# "operandSegmentSizes ... size mismatch" (see
# compiler_issues/sglang_chunk_scaled_dot_kkt.md). The shim below rewrites the
# MLIR text to the layout the external tool accepts, mirroring the proven
# FlagGems-vllm kda64 precedent: `memref_ext.*` / `gcu.matmul` ops drop their
# trailing zero segments (pure metadata, no operand rebinding), and a failed
# `gcu.matmul` compile is retried with the legacy four-operand layout. It is
# process-global (all kernels compile through the same backend entry),
# idempotent, and leaves already-compatible IR untouched. Remove once the wheel
# and the external tool are aligned.

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


# _patch_gcu_compiler_opt()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6
# _patch_gcu_matmul_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归,详见 commit b333be6


@triton.jit
def _chunk_scaled_dot_kkt_gstore_kernel(
    k_ptr,
    beta_ptr,
    g_ptr,
    out_ptr,
    STRIDE_K_B: tl.constexpr,
    STRIDE_K_T: tl.constexpr,
    STRIDE_K_HG: tl.constexpr,
    STRIDE_K_K: tl.constexpr,
    STRIDE_BETA_B: tl.constexpr,
    STRIDE_BETA_T: tl.constexpr,
    STRIDE_BETA_H: tl.constexpr,
    STRIDE_G_B: tl.constexpr,
    STRIDE_G_T: tl.constexpr,
    STRIDE_G_H: tl.constexpr,
    STRIDE_O_B: tl.constexpr,
    STRIDE_O_T: tl.constexpr,
    STRIDE_O_H: tl.constexpr,
    STRIDE_O_BT: tl.constexpr,
    B,
    H: tl.constexpr,
    HG: tl.constexpr,
    RATIO: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
    IEEE: tl.constexpr,
    TRANS_LOAD: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    i_hg = pid % HG
    i_tb = pid // HG
    i_b = i_tb % B
    i_t = i_tb // B
    t0 = i_t * BT
    offs_i = tl.arange(0, BT)
    offs_j = tl.arange(0, BT)
    offs_k = tl.arange(0, BK)
    o_g = tl.arange(0, RATIO)
    k_base = k_ptr + i_b * STRIDE_K_B + i_hg * STRIDE_K_HG
    gram = tl.zeros((BT, BT), dtype=tl.float32)
    for k0 in range(0, K, BK):
        if EVEN_K:
            k_tile = tl.load(
                k_base
                + (t0 + offs_i[:, None]) * STRIDE_K_T
                + (k0 + offs_k[None, :]) * STRIDE_K_K
            )
            if TRANS_LOAD:
                kt_tile = tl.trans(k_tile)
            else:
                kt_tile = tl.load(
                    k_base
                    + (k0 + offs_k[:, None]) * STRIDE_K_K
                    + (t0 + offs_j[None, :]) * STRIDE_K_T
                )
        else:
            in_k = offs_k < K - k0
            k_tile = tl.load(
                k_base
                + (t0 + offs_i[:, None]) * STRIDE_K_T
                + (k0 + offs_k[None, :]) * STRIDE_K_K,
                mask=in_k[None, :],
                other=0.0,
            )
            if TRANS_LOAD:
                kt_tile = tl.trans(k_tile)
            else:
                kt_tile = tl.load(
                    k_base
                    + (k0 + offs_k[:, None]) * STRIDE_K_K
                    + (t0 + offs_j[None, :]) * STRIDE_K_T,
                    mask=in_k[:, None],
                    other=0.0,
                )
        if IEEE:
            gram = tl.dot(
                k_tile,
                kt_tile,
                gram,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
        else:
            gram = tl.dot(k_tile, kt_tile, gram, out_dtype=tl.float32)
    gram = tl.where(offs_i[:, None] > offs_j[None, :], gram, 0.0)
    h0 = i_hg * RATIO
    b_beta = tl.load(
        beta_ptr
        + i_b * STRIDE_BETA_B
        + (t0 + offs_i)[:, None] * STRIDE_BETA_T
        + (h0 + o_g)[None, :] * STRIDE_BETA_H
    ).to(tl.float32)
    a = gram[:, None, :] * b_beta[:, :, None]
    if USE_G:
        b_g = tl.load(
            g_ptr
            + i_b * STRIDE_G_B
            + (t0 + offs_i)[:, None] * STRIDE_G_T
            + (h0 + o_g)[None, :] * STRIDE_G_H
        ).to(tl.float32)
        b_g_t = tl.load(
            g_ptr
            + i_b * STRIDE_G_B
            + (t0 + offs_i)[None, :] * STRIDE_G_T
            + (h0 + o_g)[:, None] * STRIDE_G_H
        ).to(tl.float32)
        g_diff = b_g[:, :, None] - b_g_t[None, :, :]
        a = a * tl.where(
            g_diff <= 0, tl.math.exp2(g_diff * 1.4426950408889634), 0.0
        )
    tl.store(
        out_ptr
        + i_b * STRIDE_O_B
        + (t0 + offs_i)[:, None, None] * STRIDE_O_T
        + (h0 + o_g)[None, :, None] * STRIDE_O_H
        + offs_j[None, None, :] * STRIDE_O_BT,
        a,
    )


def _block_k(k_dim):
    if k_dim <= 16:
        return 16
    if k_dim <= 32:
        return 32
    if k_dim <= 64:
        return 64
    return 128


def chunk_scaled_dot_kkt(k, beta, g_cumsum=None, chunk_size=64):
    k = k.contiguous()
    beta = beta.contiguous()
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = int(chunk_size)
    NT = T // BT
    ratio = H // Hg
    out = torch.empty(B, T, H, BT, device=k.device, dtype=torch.float32)
    use_g = g_cumsum is not None
    if use_g:
        g_cumsum = g_cumsum.contiguous()
        sg = g_cumsum.stride()
        g_ptr = g_cumsum
    else:
        sg = (0, 0, 0)
        g_ptr = k
    sk = k.stride()
    sb = beta.stride()
    so = out.stride()
    ieee = k.dtype == torch.float32
    bk = _block_k(K)
    trans_load = k.dtype != torch.float32
    even_k = K % bk == 0
    _chunk_scaled_dot_kkt_gstore_kernel[NT * B * Hg,](
        k,
        beta,
        g_ptr,
        out,
        sk[0],
        sk[1],
        sk[2],
        sk[3],
        sb[0],
        sb[1],
        sb[2],
        sg[0],
        sg[1],
        sg[2],
        so[0],
        so[1],
        so[2],
        so[3],
        B,
        H,
        Hg,
        ratio,
        K,
        BT,
        bk,
        use_g,
        ieee,
        trans_load,
        even_k,
        num_warps=1,
        num_stages=1,
    )
    return out


__all__ = ["chunk_scaled_dot_kkt"]
