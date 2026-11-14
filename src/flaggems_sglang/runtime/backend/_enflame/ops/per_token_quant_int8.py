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

import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    offs = rows[:, None] * N + cols[None, :]
    if EVEN_M:
        x = tl.load(x_ptr + offs)
        amax = tl.maximum(tl.max(tl.abs(x), axis=1).to(tl.float32), 1.0e-10)
        scale = amax / 127.0
        inv = 127.0 / amax
        q = tl.clamp(x.to(tl.float32) * inv[:, None], -128.0, 127.0).to(tl.int8)
        tl.store(q_ptr + offs, q)
        tl.store(s_ptr + rows, scale)
    else:
        rmask = rows < M
        x = tl.load(x_ptr + offs, mask=rmask[:, None], other=0.0)
        amax = tl.maximum(tl.max(tl.abs(x), axis=1).to(tl.float32), 1.0e-10)
        scale = amax / 127.0
        inv = 127.0 / amax
        q = tl.clamp(x.to(tl.float32) * inv[:, None], -128.0, 127.0).to(tl.int8)
        tl.store(q_ptr + offs, q, mask=rmask[:, None])
        tl.store(s_ptr + rows, scale, mask=rmask)


@triton.jit
def _per_token_quant_int8_persistent(
    x_ptr,
    q_ptr,
    s_ptr,
    M,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_PID: tl.constexpr,
    EVEN_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(M, BLOCK_M)
    for t in range(pid, num_tiles, NUM_PID):
        rows = t * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        offs = rows[:, None] * N + cols[None, :]
        if EVEN_M:
            x = tl.load(x_ptr + offs)
            amax = tl.maximum(tl.max(tl.abs(x), axis=1).to(tl.float32), 1.0e-10)
            scale = amax / 127.0
            inv = 127.0 / amax
            q = tl.clamp(x.to(tl.float32) * inv[:, None], -128.0, 127.0).to(tl.int8)
            tl.store(q_ptr + offs, q)
            tl.store(s_ptr + rows, scale)
        else:
            rmask = rows < M
            x = tl.load(x_ptr + offs, mask=rmask[:, None], other=0.0)
            amax = tl.maximum(tl.max(tl.abs(x), axis=1).to(tl.float32), 1.0e-10)
            scale = amax / 127.0
            inv = 127.0 / amax
            q = tl.clamp(x.to(tl.float32) * inv[:, None], -128.0, 127.0).to(tl.int8)
            tl.store(q_ptr + offs, q, mask=rmask[:, None])
            tl.store(s_ptr + rows, scale, mask=rmask)


@triton.jit
def _per_token_quant_int8_rows(
    x_ptr,
    q_ptr,
    s_ptr,
    N,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    if EVEN_N:
        x = tl.load(x_ptr + row * N + offs)
    else:
        x = tl.load(x_ptr + row * N + offs, mask=offs < N, other=0.0)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0).to(tl.float32), 1.0e-10)
    scale = amax / 127.0
    inv = 127.0 / amax
    q = tl.clamp(x.to(tl.float32) * inv, -128.0, 127.0).to(tl.int8)
    if EVEN_N:
        tl.store(q_ptr + row * N + offs, q)
    else:
        tl.store(q_ptr + row * N + offs, q, mask=offs < N)
    tl.store(s_ptr + row, scale)


_MAX_TILE_ELEMS = 131072
_PERSIST_PROGRAMS = 48
_PERSIST_MIN_TILES = 64


def _block_m_for(M, N):
    bm_cap = max(1, _MAX_TILE_ELEMS // N)
    bm = min(bm_cap, triton.next_power_of_2(M))
    grid_floor = 16 if M <= 64 else 32
    while triton.cdiv(M, bm) < grid_floor and bm > 1:
        bm //= 2
    while triton.cdiv(M, bm) > 65535 and bm < bm_cap:
        bm *= 2
    return bm


def per_token_quant_int8(x):
    M, N = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    if M == 0:
        return q, scales
    if M <= 32 or N & (N - 1):
        block_n = triton.next_power_of_2(N)
        _per_token_quant_int8_rows[(M,)](
            x,
            q,
            scales,
            N,
            BLOCK_N=block_n,
            EVEN_N=(block_n == N),
            num_warps=4,
            num_stages=1,
        )
        return q, scales
    bm = _block_m_for(M, N)
    num_warps = 4 if bm * N >= _MAX_TILE_ELEMS else 2
    even_m = (M % bm == 0)
    if triton.cdiv(M, bm) >= _PERSIST_MIN_TILES:
        _per_token_quant_int8_persistent[(_PERSIST_PROGRAMS,)](
            x,
            q,
            scales,
            M,
            N=N,
            BLOCK_M=bm,
            BLOCK_N=N,
            NUM_PID=_PERSIST_PROGRAMS,
            EVEN_M=even_m,
            num_warps=num_warps,
            num_stages=1,
        )
        return q, scales
    grid = (triton.cdiv(M, bm),)
    _per_token_quant_int8[grid](
        x,
        q,
        scales,
        M,
        N=N,
        BLOCK_M=bm,
        BLOCK_N=N,
        EVEN_M=even_m,
        num_warps=num_warps,
        num_stages=1,
    )
    return q, scales


def _install_gcuir_text_compat():
    """Scoped fallback-only retry for the process-global GCUIR shims.

    The 2026-11-07 toolchain skew left unconditional dialect shims installed
    by other op modules; they strip the trailing-zero ``operandSegmentSizes``
    segment that the current external gcu-compiler-opt requires, which breaks
    fresh compiles of this op's 2D tile loads (see
    compiler_issues/sglang_l2norm_masked2d_slice_pad_start.md). When the
    wrapped chain rejects our kernels, retry the unmodified IR directly
    against the external tool. The registrar imports op modules
    alphabetically, so this wrapper sits outside the earlier shims and sees
    the original IR. Remove once the toolchain and shims are re-synced.
    """

    try:
        from triton.backends.enflame import toolkit
    except ImportError:
        return
    if getattr(toolkit.gcu_compiler_opt, "_per_token_quant_int8_gcu_compat", False):
        return
    chain = toolkit.gcu_compiler_opt

    def gcu_compiler_opt_compat(content, *args):
        try:
            return chain(content, *args)
        except Exception as first_error:
            text = content if isinstance(content, str) else str(content)
            if "_per_token_quant_int8" not in text:
                raise first_error
            try:
                return toolkit._run_command(
                    "gcu-compiler-opt",
                    text,
                    "-mlir-print-op-generic",
                    *args,
                )
            except Exception:
                raise first_error

    gcu_compiler_opt_compat._per_token_quant_int8_gcu_compat = True
    toolkit.gcu_compiler_opt = gcu_compiler_opt_compat


# _install_gcuir_text_compat()  # DISABLED 2026-11-09: 私有 GCUIR shim 实测引入回归(降级重试路径
# 搞挂合法 kernel,同 conftest shim 事故);详见 commit b333be6/e1acdf6 与
# compiler_issues/vllm_memref_slice_pad_start.md。编译器修复后统一处置。


# ---------------------------------------------------------------------------
# aten::max.dim dispatch guard (torch_gcu P21 defect workaround)
#
# In the int64 regime torch_gcu's native aten::max.dim (topsatenMax) rejects
# the "values + Long indices" dual-output contract (op_aten_max.h:210
# "int64 idx requires int64 data type." -> TOPSATEN_STATUS_NOT_SUPPORT ->
# gcu_utils.cpp:74 Check failed -> whole-process abort rc=134, no Python
# exception).  This op's accuracy suite reference leg
# (per_token_group_quant_int8.py: ``x_.abs().max(dim=-1, keepdim=True)[0]``)
# and torch.testing.assert_close's mismatch formatting (torch.max(abs_diff))
# therefore die before/independent of the FlagGems op under test.  Route the
# aten::max.dim dispatch for the GCU key onto a CPU bridge -- the CPU aten
# kernel is the exact oracle for both outputs, so the reference stays stock.
# Same guard as the FlagGems-enflame min.py / fused chunk_gated_delta_rule
# registrations (compiler_problems/P21).  With the env off the native path
# applies the Long->Int replacement and works, so the guard stays out of the
# way; Long H2D is only taken in the regime that gives Long a true 8-byte
# layout.  Idempotent via a torch-attached flag.
# ---------------------------------------------------------------------------


def _gcu_int64_layout_enabled() -> bool:
    # torch_gcu only gives Long tensors a true 8-byte element layout when
    # TORCH_GCU_ENABLE_INT64_AND_UINT64 selects the int64 regime; with the env
    # off it applies the Long->Int replacement instead.  Accepted values
    # mirror torch_gcu's parse: "1", "true", and nonzero numerics enable
    # int64; unset, "0", "false", "TRUE"/"on"/"yes" keep the replacement.
    val = os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64")
    if val is None:
        return False
    if val == "true":
        return True
    try:
        return int(val) != 0
    except ValueError:
        return False


_MAX_DIM_GUARD_FLAG = "_flaggems_sglang_gcu_max_dim_aten_guard_registered"
_MAX_DIM_GUARD_LIBRARY = None


def _aten_max_dim_cpu_bridge(self, dim, keepdim=False):
    native = torch.max(self.to("cpu"), dim=dim, keepdim=keepdim)
    device = self.device
    return (native.values.to(device), native.indices.to(device))


def _install_native_max_dim_guard():
    global _MAX_DIM_GUARD_LIBRARY
    if not _gcu_int64_layout_enabled():
        return
    if getattr(torch, _MAX_DIM_GUARD_FLAG, False):
        return
    library = torch.library.Library("aten", "IMPL")
    registered = False
    try:
        library.impl(
            "max.dim",
            _aten_max_dim_cpu_bridge,
            "PrivateUse1",
            allow_override=True,
        )
        registered = True
    except TypeError:
        try:
            library.impl("max.dim", _aten_max_dim_cpu_bridge, "PrivateUse1")
            registered = True
        except RuntimeError as error:
            logger.warning(
                "max.dim PrivateUse1 registration skipped: %s", error
            )
    except RuntimeError as error:
        logger.warning("max.dim PrivateUse1 registration skipped: %s", error)
    if registered:
        _MAX_DIM_GUARD_LIBRARY = library
        setattr(torch, _MAX_DIM_GUARD_FLAG, True)


_install_native_max_dim_guard()


__all__ = ["per_token_quant_int8"]
