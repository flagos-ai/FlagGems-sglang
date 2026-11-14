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

"""Fused recurrent gated-delta-rule packed decode (single token).

Single Triton kernel that consumes the packed QKV projection directly and
performs the whole straight-line recurrence (gate, delta-rule update, readout,
state writeback) in one pass over the paged state.

Enflame GCU adaptation notes:
1. The launch grid is folded within the GCU limits (grid.x <= 65535,
   grid.y <= 255, grid.z <= 255); the kernel recombines a flat program id and
   guards ``pid < num_programs``. The generic 2-D ``(NV, B*HV)`` grid exceeds
   grid.y for B*HV > 255 (e.g. B=8, HV=32), which failed every benchmark
   shape beyond B=1.
2. The state tile is loaded *before* the small scalar/gate loads ("big DMA
   first"). Besides overlapping the state DMA with the gate chain, this load
   order is a correctness requirement under the current (2026-11-07 20:22)
   toolchain: with the state load placed after the gate/l2norm chain the
   multi-V-block form (e.g. V=192) silently miscomputes at any num_warps,
   and num_warps >= 2 miscomputes the output even with the early load
   (V=192, d_out ~6e-2). Early load + num_warps=1 passes an 8-shape
   matrix incl. l2norm, pad slots and masked tails.
3. Wave-aware BV widening (host metadata only): programs = B*HV*ceil(V/BV)
   each run a serial load->tree->rank-1-update->store chain, so wall clock
   ~ ceil(programs/48) waves. Widening BV 64->128 halves the program count
   for V=128 while the per-program cost grows sub-linearly; the dispatch
   widens only when it strictly reduces the wave count, the state tile
   stays <= 16384 fp32 elements (DSM-healthy domain) and the state rows
   are dense. num_warps=1 (2/4 measured 1.6-1.8x slower).
4. torch_gcu narrows int64 tensors to int32 storage while metadata still
   reports int64, so kernel-side int64-width index loads of narrowed storage
   read garbage slots; ssm_state_indices is normalized on the host with zero
   device work (narrowed storage becomes an int32 view; true 8-byte storage
   under TORCH_GCU_ENABLE_INT64_AND_UINT64 passes through for native int64
   loads, avoiding a per-call ATen cast kernel that would add a second
   launch). The slot base offset uses int64 arithmetic to stay valid for
   large state pools.
5. Toolchain-compat shim: the 2026-11-07 20:22 toolchain skew makes
   gcu-compiler-opt reject ``memref_ext.slice_*``/``gcu.matmul``/
   ``memset_start`` operandSegmentSizes encodings emitted for 2-D tile
   kernels, breaking every first-time compilation of this kernel. The
   scoped retry-path GCUIR downgrade below rewrites the op encodings back
   to the form the external tool accepts (same pattern as the
   chunk_local_cumsum_vector overlay; see compiler_issues/
   sglang_chunk_local_cumsum_vector.md). Remove once the toolchain
   versions realign.
"""

import os
import re

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Toolchain-compat shim (2026-11-07 20:22 skew): retry-path GCUIR downgrade
# scoped to this module's kernel. See the module docstring and
# compiler_issues/sglang_chunk_local_cumsum_vector.md for the full analysis.
# ---------------------------------------------------------------------------
_COMPAT_SCOPED_KERNELS = ("_packed_decode_kernel",)


def _downgrade_gcuir_for_external_tool(text):
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
                r" <\{operandSegmentSizes = array<i32: [0-9, ]+>}>", "", line
            )
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def _install_gcuir_compat():
    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_packed_decode_gcu_compat", False):
        return
    prev = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return prev(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if not any(name in text for name in _COMPAT_SCOPED_KERNELS):
                raise first_error
            try:
                return toolkit._run_command(
                    "gcu-compiler-opt",
                    _downgrade_gcuir_for_external_tool(text),
                    "-mlir-print-op-generic",
                    *args,
                )
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._packed_decode_gcu_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _install_gcuir_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归(降级重试路径
# 搞挂合法 kernel,同 conftest shim 事故);详见 commit b333be6/e1acdf6 与
# compiler_issues/vllm_memref_slice_pad_start.md。编译器修复后统一处置。

# GCU launch limits (triton.backends.enflame.backend.GcuLauncher):
# grid.x <= 65535, grid.y <= 255, grid.z <= 255.
_GRID_X_MAX = 65535
_GRID_Y_MAX = 255

# Wave-aware BV dispatch constant: number of stream processors per GCU card.
_GCU_NUM_SP = 48

# DSM-healthy state tile bound (fp32 elements) for the widened arm.
_MAX_STATE_TILE_ELEMS = 16384


@triton.jit
def _packed_decode_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    A_log_ptr,
    dt_bias_ptr,
    state_ptr,
    out_ptr,
    idx_ptr,
    scale,
    grid_x,
    grid_xy,
    num_programs,
    HV: tl.constexpr,
    H: tl.constexpr,
    stride_qkv_tok: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_hv: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_out_n: tl.constexpr,
    stride_out_hv: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    NV: tl.constexpr,
    EVEN_V: tl.constexpr,
    EVEN_K: tl.constexpr,
    USE_L2NORM: tl.constexpr,
):
    pid = tl.program_id(0) + tl.program_id(1) * grid_x + tl.program_id(2) * grid_xy
    if pid >= num_programs:
        return
    pid_v = pid % NV
    pid_nh = pid // NV
    i_n = pid_nh // HV
    i_hv = pid_nh % HV

    v_idx = pid_v * BV + tl.arange(0, BV)
    k_idx = tl.arange(0, BK)

    out_base = out_ptr + i_n * stride_out_n + i_hv * stride_out_hv

    slot = tl.load(idx_ptr + i_n)

    if slot <= 0:
        # NULL_BLOCK_ID / PAD_SLOT_ID: zero `out`, leave the state untouched.
        zval = tl.zeros([BV], dtype=tl.float32).to(out_ptr.dtype.element_ty)
        if EVEN_V:
            tl.store(out_base + v_idx, zval)
        else:
            tl.store(out_base + v_idx, zval, mask=v_idx < V)
    else:
        ratio = HV // H
        i_h = i_hv // ratio

        # The [BV, BK] state tile is the biggest DMA on the critical path;
        # it is emitted first, before the small scalar and qkv loads. Under
        # the 2026-11-07 toolchain this order is also a correctness
        # requirement (see the module docstring).
        s_base = (
            state_ptr
            + slot.to(tl.int64) * stride_state_slot
            + i_hv * stride_state_hv
        )
        offs = v_idx[:, None] * stride_state_v + k_idx[None, :] * stride_state_k
        if EVEN_V and EVEN_K:
            h = tl.load(s_base + offs).to(tl.float32)
        elif EVEN_V:
            h = tl.load(
                s_base + offs, mask=(k_idx < K)[None, :], other=0.0
            ).to(tl.float32)
        elif EVEN_K:
            h = tl.load(
                s_base + offs, mask=(v_idx < V)[:, None], other=0.0
            ).to(tl.float32)
        else:
            h = tl.load(
                s_base + offs,
                mask=(v_idx < V)[:, None] & (k_idx < K)[None, :],
                other=0.0,
            ).to(tl.float32)

        row = mixed_qkv_ptr + i_n * stride_qkv_tok
        qk_half = H * K
        q_base = row + i_h * K
        k_base = row + qk_half + i_h * K
        v_base = row + 2 * qk_half + i_hv * V

        if EVEN_K:
            q_vec = tl.load(q_base + k_idx).to(tl.float32)
            k_vec = tl.load(k_base + k_idx).to(tl.float32)
        else:
            km = k_idx < K
            q_vec = tl.load(q_base + k_idx, mask=km, other=0.0).to(tl.float32)
            k_vec = tl.load(k_base + k_idx, mask=km, other=0.0).to(tl.float32)
        if EVEN_V:
            v_vec = tl.load(v_base + v_idx).to(tl.float32)
        else:
            v_vec = tl.load(v_base + v_idx, mask=v_idx < V, other=0.0).to(
                tl.float32
            )

        a_v = tl.load(a_ptr + i_n * HV + i_hv).to(tl.float32)
        b_raw = tl.load(b_ptr + i_n * HV + i_hv)
        # deliberate round-trip through b's dtype, as in the reference kernel
        beta = tl.sigmoid(b_raw.to(tl.float32)).to(b_raw.dtype).to(tl.float32)
        alog = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dtb = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

        x = a_v + dtb
        sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))), x)
        decay = tl.exp(-tl.exp(alog) * sp)

        if USE_L2NORM:
            q_vec = q_vec / tl.sqrt(tl.sum(q_vec * q_vec) + 1e-6)
            k_vec = k_vec / tl.sqrt(tl.sum(k_vec * k_vec) + 1e-6)
        q_vec = q_vec * scale

        # pred = sum_k (decay*h)[v,k]*k[k] = decay * sum_k h[v,k]*k[k]
        hk = tl.sum(h * k_vec[None, :], axis=1)
        hq = tl.sum(h * q_vec[None, :], axis=1)
        kq = tl.sum(k_vec * q_vec, axis=0)

        pred = decay * hk
        vc = (v_vec - pred) * beta
        o = decay * hq + vc * kq
        h2 = decay * h + vc[:, None] * k_vec[None, :]

        o_val = o.to(out_ptr.dtype.element_ty)
        if EVEN_V:
            tl.store(out_base + v_idx, o_val)
        else:
            tl.store(out_base + v_idx, o_val, mask=v_idx < V)

        h2_val = h2.to(state_ptr.dtype.element_ty)
        if EVEN_V and EVEN_K:
            tl.store(s_base + offs, h2_val)
        elif EVEN_V:
            tl.store(s_base + offs, h2_val, mask=(k_idx < K)[None, :])
        elif EVEN_K:
            tl.store(s_base + offs, h2_val, mask=(v_idx < V)[:, None])
        else:
            tl.store(
                s_base + offs,
                h2_val,
                mask=(v_idx < V)[:, None] & (k_idx < K)[None, :],
            )


def _decode_grid(total):
    """Fold ``total`` logical programs into a GCU-safe 3-D grid.

    Any decomposition with ``grid_x * grid_y * grid_z >= total`` is correct;
    the kernel recombines the flat program id via ``grid_x``/``grid_xy``
    runtime arguments and guards ``pid < num_programs``. Covers up to
    65535 * 255 * 255 ~= 4.26e9 programs.
    """
    if total <= _GRID_X_MAX:
        return (total, 1, 1)
    grid_y = triton.cdiv(total, _GRID_X_MAX)
    if grid_y <= _GRID_Y_MAX:
        return (triton.cdiv(total, grid_y), grid_y, 1)
    grid_z = triton.cdiv(total, _GRID_X_MAX * _GRID_Y_MAX)
    grid_y = triton.cdiv(total, _GRID_X_MAX * grid_z)
    return (triton.cdiv(total, grid_y * grid_z), grid_y, grid_z)


def _normalize_index_tensor(tensor):
    """Normalize int64 index metadata for kernel loads, zero device work.

    torch_gcu narrows int64 tensors to int32 storage while the metadata
    still reports int64 (unless TORCH_GCU_ENABLE_INT64_AND_UINT64 enables
    true 8-byte storage); kernel-side int64-width loads of the narrowed
    storage read garbage slots and drive out-of-bounds state updates. In
    narrowed mode the physical 4-byte words are int32-isomorphic, so a
    zero-copy set_ view with the original shape and strides is exact. With
    true 8-byte storage the tensor passes through untouched: the kernel's
    slot load then reads the native int64 word and the .to(tl.int64)
    base-offset widening is a no-op, which avoids a per-call ATen cast
    chain that added a second device launch (single-kernel contract) and
    ~1.5-2.6us of cast-kernel time inside the timed window (2026-11-12
    fused_recurrent_gated_delta_rule precedent). Pool slots always fit
    either width.
    """
    if tensor is None or tensor.dtype != torch.int64:
        return tensor
    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") in ("1", "true"):
        return tensor
    view = torch.empty(0, dtype=torch.int32, device=tensor.device)
    view.set_(
        tensor.untyped_storage(),
        tensor.storage_offset(),
        tensor.shape,
        tensor.stride(),
    )
    return view


def _pick_block(B, HV, V, K, state):
    """Wave-aware BV dispatch (host metadata only, no device reads).

    Programs = B * HV * ceil(V / BV) each run one serial
    load -> trees -> rank-1 update -> store chain, so wall clock
    ~ ceil(programs / 48) waves. Widening BV folds more V rows into one
    program: the per-program cost grows sub-linearly while the wave count
    drops. The widened arm is taken only when it strictly reduces the wave
    count, the state rows are dense (K-contiguous) and the state tile stays
    within the DSM-healthy bound.
    """
    BK = max(16, triton.next_power_of_2(K))
    BV = min(64, max(16, triton.next_power_of_2(V)))
    wide_bv = min(
        triton.next_power_of_2(V), 128, max(16, _MAX_STATE_TILE_ELEMS // BK)
    )
    if wide_bv > BV:
        items_base = B * HV * triton.cdiv(V, BV)
        items_wide = B * HV * triton.cdiv(V, wide_bv)
        waves_drop = triton.cdiv(items_base, _GCU_NUM_SP) > triton.cdiv(
            items_wide, _GCU_NUM_SP
        )
        dense = state.stride(3) == 1 and state.stride(2) == K
        if waves_drop and dense:
            BV = wide_bv
    return BV, BK


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    scale,
    initial_state,
    out,
    ssm_state_indices,
    use_qk_l2norm_in_kernel=False,
):
    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]

    qk_dim = mixed_qkv.shape[1] - HV * V
    q_dim = qk_dim // 2
    H = q_dim // K

    if B == 0:
        return out, initial_state

    # torch_gcu narrows int64 tensors to int32 storage while the metadata
    # still reports int64, so kernel-side int64-width loads of narrowed
    # storage read garbage slots; normalize with zero device work (see
    # _normalize_index_tensor). Pool slots always fit either width.
    ssm_state_indices = _normalize_index_tensor(ssm_state_indices)

    BV, BK = _pick_block(B, HV, V, K, initial_state)
    NV = triton.cdiv(V, BV)
    total = NV * B * HV
    grid = _decode_grid(total)

    _packed_decode_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        initial_state,
        out,
        ssm_state_indices,
        float(scale),
        grid[0],
        grid[0] * grid[1],
        total,
        int(HV),
        int(H),
        int(mixed_qkv.stride(0)),
        int(initial_state.stride(0)),
        int(initial_state.stride(1)),
        int(initial_state.stride(2)),
        int(initial_state.stride(3)),
        int(out.stride(0)),
        int(out.stride(2)),
        V=int(V),
        K=int(K),
        BV=BV,
        BK=BK,
        NV=NV,
        EVEN_V=(V % BV) == 0,
        EVEN_K=(BK == K),
        USE_L2NORM=bool(use_qk_l2norm_in_kernel),
        num_warps=1,
    )
    return out, initial_state


__all__ = ["fused_recurrent_gated_delta_rule_packed_decode"]
