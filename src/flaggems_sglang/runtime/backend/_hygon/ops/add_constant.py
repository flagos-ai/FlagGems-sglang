# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
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

Why this is a vectorisation problem, not a tuning problem
--------------------------------------------------------
Adding a constant is pure memory traffic: the semantics need one read and one
write per element, so the whole cost is 8 bytes per element and every serious
implementation is bandwidth bound.  The statement says the reference switches to
a `kMaxVecBytes`-wide vectorised path above 2^20 elements, i.e. the graded sizes
are in the regime where the baseline is issuing wide accesses.

The first submitted revision produced one masked block per program:

    off  = pid * BLOCK + arange(0, BLOCK)
    mask = off < n_elements
    v    = tl.load(src_ptr + off, mask=mask, ...)
    tl.store(dst_ptr + off, v + CONST, mask=mask)

Every access carried a predicate.  On the two chips where the field is not
competitive that showed up as a hard floor: across the whole leaderboard, 16 of
17 teams sit at roughly 0.31 on Huawei and only one team exceeds parity on
Kunlunxin, regardless of every other chip being at or above parity.  A floor that
no team crosses by tuning, on exactly the operation whose baseline is vectorised,
is the signature of a *predicated* access preventing the wide path rather than of
insufficient parallelism or the wrong block size.

So this revision removes the predicate from the hot path entirely by splitting the
launch into two kernels:

  * **full blocks** — `n_full = n // BLOCK` programs, each handling exactly one
    block with **no mask at all**, so the access is a plain contiguous run and the
    backend is free to emit the widest vector it supports.
  * **the tail** — at most one extra program, handling only the `n % BLOCK`
    remainder, masked.  It runs once per call, so its predicate costs nothing
    measurable.

Every element is still written exactly once, and the tail kernel writes nothing
past `n`.  When `n` divides evenly no tail program is launched at all.

Both kernels keep the constant as a `tl.constexpr`, folding it into the
instruction stream the way the reference's per-constant template instantiation
does, so no value is ever loaded for the addend.

Indexing stays affine and division-free: the block base is `program_id * BLOCK`
with `BLOCK` a compile-time constant, and the tail base is `n_full * BLOCK`, which
is computed on the **host** and passed in.  No runtime integer division or modulo
appears in either kernel.

Strict-backend notes (from PLAYBOOK / AI_NOTES)
-----------------------------------------------
* No `tl.full`, no `tl.where`, no reduction, no scatter store.
* No runtime integer division or modulo; the split is computed on the host.
* No width-changing `bitcast`: values are loaded and stored as int32, so nothing
  is reinterpreted.
* One-dimensional grids only, so no `grid.y` limit applies, and the program count
  is bounded so `grid.x` stays inside its limit.

There is no `try`/`except`, no device inspection and no PyTorch fallback on the
executed path.
"""

import torch
import triton
import triton.language as tl

__all__ = ["add_constant"]

_BLOCK = 1024
_MAX_GRID_X = 65535


@triton.jit
def _add_constant_full_kernel(
    src_ptr,
    dst_ptr,
    CONST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One unmasked, contiguous block per program.

    No predicate is applied, so the load and store are plain contiguous runs and
    the backend can vectorise them as widely as it supports.
    """
    off = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    # Unmasked: `BLOCK` is a compile-time multiple, so the run is contiguous and
    # aligned by construction, which is what lets the backend vectorise it.
    v = tl.load(src_ptr + off)
    tl.store(dst_ptr + off, v + CONST)


@triton.jit
def _add_constant_tail_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    base_off,
    CONST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The single masked remainder block, launched only when `n % BLOCK != 0`."""
    off = base_off + tl.arange(0, BLOCK)
    mask = off < n_elements
    v = tl.load(src_ptr + off, mask=mask, other=0)
    tl.store(dst_ptr + off, v + CONST, mask=mask)


def add_constant(src, constant):
    """Return `src + constant` for a flat int32 tensor."""
    out = torch.empty_like(src)
    n = src.numel()
    if n == 0:
        return out

    src2 = src.contiguous()
    dst2 = out
    c = int(constant)

    # Safety net: widen the block only if the program count would exceed the
    # grid limit.  This changes nothing for the sizes the tuned value covers.
    block = _BLOCK
    while -(-n // block) > _MAX_GRID_X:
        block *= 2

    n_full = n // block
    rem = n - n_full * block

    if n_full:
        _add_constant_full_kernel[(n_full,)](
            src2,
            dst2,
            CONST=c,
            BLOCK=block,
            num_warps=4,
        )
    if rem:
        # One masked program for the remainder; its base is computed here so the
        # kernel needs no division.
        _add_constant_tail_kernel[(1,)](
            src2,
            dst2,
            n,
            n_full * block,
            CONST=c,
            BLOCK=block,
            num_warps=4,
        )
    return out
