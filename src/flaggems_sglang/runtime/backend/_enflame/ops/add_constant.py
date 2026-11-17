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

"""add_constant — `dst = src + constant` for a flat int32 tensor, in Triton.

Contract (task statement)
-------------------------
    out = src + constant

`src` is a non-empty, contiguous, one-dimensional int32 tensor and the result has
the same shape and dtype.  Correctness is exact: integer arithmetic, compared
elementwise.

GCU (gcu410, 24 SIP) tuning notes — why this shape
-------------------------------------------------
Adding a constant is pure memory traffic, so the whole cost is 8 bytes per
element plus per-program fixed overhead.  Measurement on this board:

* A single load/store op wider than 8192 int32 lanes fails to compile in
  ``gcu-compiler-opt`` (``memref_ext.slice_start`` operandSegmentSizes error),
  at any num_warps; 8192 needs num_warps >= 4 (per-warp width <= 2048 lanes).
  So per-op width is capped at W=8192 (see
  compiler_issues/sglang_add_constant.md for the full boundary table).
* DSM-free streaming kernels unlock the double-occupancy program slot: the
  board retires two programs per SP, i.e. grid=48 is fully concurrent.  A
  persistent grid-stride kernel capped at 48 programs beats wider flat grids
  everywhere in the large range: n=2^20 measures 7.32us vs 7.49-7.64us for
  32 static-unrolled programs and 10.5us for 128 single-op programs; n=2^24
  measures 48.97us vs 55.73us for the 256-program K=8 form (-12%).
* Mid sizes prefer more, narrower programs over fewer wide ones: at n=2^15 a
  4096-lane block grid (8 programs) beats the 8192-lane grid (4 programs) by
  ~6%; at n=2^16 both are equal (16 vs 8 programs, 3.81-3.82us interleaved).
  W=4096 is never worse in [2^13, 2^17], so the mid arm uses it.

Schedule (host-side, metadata only)
-----------------------------------
* n <= 8192: one full block of next_pow2(n) (floor 1024) — the launch floor
  dominates; nothing to amortise.
* 8192 < n <= 131072 (32 * 4096): W=4096 full blocks, grid = n // 4096
  (2..32 programs), no mask on the hot path.
* n > 131072: persistent grid-stride kernel with W=8192 ops and the grid
  capped at 48; each program strides over the full-block range with a runtime
  loop bound, so arbitrary n needs no K escalation and never exceeds the
  grid.x limit.
* n > 8192 * 16 * 65535 (~2^33): the op falls back to a masked 2D-grid kernel
  that is correct for any n (grid.x bound; keeps the hot paths' offsets in
  the int32-safe domain).
* The remainder below the last full block is handled by a small masked tail
  kernel (a single 8192-lane tile per arm below the gigantic cutoff),
  launched only when the remainder is non-zero.

The constant stays a ``tl.constexpr`` (folded into the instruction stream), the
full-block hot paths carry no predicate, and indexing stays affine and
division-free — block bases are ``program_id * W`` with W a compile-time
constant and the tail base computed on the host.

Strict-backend notes (from PLAYBOOK / AI_NOTES)
-----------------------------------------------
* No `tl.full`, no `tl.where`, no reduction, no scatter store.
* No runtime integer division or modulo; the split is computed on the host.
* One-dimensional grids for the hot path and the persistent kernel; the masked
  fallback uses a rectangular 2D grid with an explicit bound mask
  (memcpy-style), never an out-of-range unmasked access.

There is no `try`/`except`, no device inspection and no PyTorch fallback on the
executed path.
"""

import torch
import triton
import triton.language as tl

__all__ = ["add_constant"]

_W = 8192  # per-op lane width ceiling (compiler limit on this backend)
_W_MID = 4096  # mid-range op width: more programs, never slower than _W
_MID_MAX = 131072  # 32 * _W_MID: top of the W=4096 flat arm
_PERSIST_GRID = 48  # 24 SIP x 2 programs/SP (double occupancy, DSM-free)
_MAX_GRID_X = 65535
_TAIL_BLOCK = 8192


@triton.jit
def _add_constant_full_kernel(
    src_ptr,
    dst_ptr,
    CONST: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
):
    """K unmasked, contiguous W-lane ops per program.

    A single op wider than 8192 lanes does not compile on this backend, so a
    program covers K * W lanes as K plain contiguous runs — still predicate
    free, so the backend vectorises each run as widely as it supports.
    """
    base = tl.program_id(axis=0) * (W * K)
    for i in tl.static_range(K):
        off = base + i * W + tl.arange(0, W)
        v = tl.load(src_ptr + off)
        tl.store(dst_ptr + off, v + CONST)


@triton.jit
def _add_constant_persist_kernel(
    src_ptr,
    dst_ptr,
    n_end,
    CONST: tl.constexpr,
    W: tl.constexpr,
    GRID: tl.constexpr,
):
    """Persistent grid-stride kernel over the full-block range [0, n_end).

    Each of the GRID programs takes W-lane ops at ``pid*W + j*GRID*W``; the
    loop bound is a runtime scalar, so one compiled variant serves any n.
    GRID must equal the launched grid. Ops are unmasked: n_end is a whole
    number of W-lane blocks computed on the host.
    """
    pid = tl.program_id(axis=0)
    o = tl.arange(0, W)
    base = pid * W
    step = GRID * W
    for off in range(base, n_end, step):
        v = tl.load(src_ptr + off + o)
        tl.store(dst_ptr + off + o, v + CONST)


@triton.jit
def _add_constant_tail_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    base_off,
    CONST: tl.constexpr,
    BLOCK: tl.constexpr,
    GRID_X: tl.constexpr,
):
    """Masked remainder blocks; also the whole-range fallback for gigantic n.

    Launched as a rectangular 2D grid ``(GRID_X, cdiv(tiles, GRID_X))``; pids
    past the end are fully masked, so a rectangular grid is always safe here.
    """
    pid = tl.program_id(axis=0) + tl.program_id(axis=1) * GRID_X
    off = base_off + pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n_elements
    v = tl.load(src_ptr + off, mask=mask, other=0)
    tl.store(dst_ptr + off, v + CONST, mask=mask)


def _launch_tail(src2, out, n, base_off, c):
    """Masked tail over [base_off, n); at most one tile below the cutoff."""
    rem = n - base_off
    tiles = (rem + _TAIL_BLOCK - 1) // _TAIL_BLOCK
    _add_constant_tail_kernel[(tiles, 1)](
        src2,
        out,
        n,
        base_off,
        CONST=c,
        BLOCK=_TAIL_BLOCK,
        GRID_X=tiles,
        num_warps=8,
    )


def add_constant(src, constant):
    """Return `src + constant` for a flat int32 tensor."""
    out = torch.empty_like(src)
    n = src.numel()
    if n == 0:
        return out

    src2 = src.contiguous()
    c = int(constant)

    if n > _W * 16 * _MAX_GRID_X:
        # Gigantic n: masked rectangular-grid kernel over the whole range.
        tiles = (n + _TAIL_BLOCK - 1) // _TAIL_BLOCK
        grid_x = min(tiles, _MAX_GRID_X)
        grid_y = (tiles + grid_x - 1) // grid_x
        _add_constant_tail_kernel[(grid_x, grid_y)](
            src2,
            out,
            n,
            0,
            CONST=c,
            BLOCK=_TAIL_BLOCK,
            GRID_X=grid_x,
            num_warps=8,
        )
        return out

    if n <= _W:
        # Launch floor: one full block covers everything.
        block = max(1024, 1 << (n - 1).bit_length())
        _add_constant_full_kernel[(1,)](
            src2,
            out,
            CONST=c,
            W=block,
            K=1,
            num_warps=8,
        )
        return out

    if n <= _MID_MAX:
        # Mid range: W_MID-wide full blocks (2..32 programs) + masked tail.
        n_full = n // _W_MID
        _add_constant_full_kernel[(n_full,)](
            src2,
            out,
            CONST=c,
            W=_W_MID,
            K=1,
            num_warps=8,
        )
        if n_full * _W_MID < n:
            _launch_tail(src2, out, n, n_full * _W_MID, c)
        return out

    # Large range: persistent grid-stride, grid capped at the double-
    # occupancy concurrency limit; the runtime loop bound serves any n.
    n_ops = n // _W
    grid = min(_PERSIST_GRID, n_ops)
    n_end = n_ops * _W
    _add_constant_persist_kernel[(grid,)](
        src2,
        out,
        n_end,
        CONST=c,
        W=_W,
        GRID=grid,
        num_warps=8,
    )
    if n_end < n:
        _launch_tail(src2, out, n, n_end, c)
    return out
