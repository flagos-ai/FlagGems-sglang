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

import torch
import triton
import triton.language as tl

# Tile width for the flat kernel. Large tiles amortize per-program launch and
# address-setup cost; past this size the measured throughput stops improving.
_FLAT_BLOCK = 16384

# Smallest tile worth using. Below this the grid gets so wide that per-program
# overhead dominates, and the masked kernel does better.
_MIN_FLAT_BLOCK = 1024

# The flat kernel indexes with int32 offsets.
_MAX_INT32_ELEMS = 1 << 31

# Warps per program for the flat kernel (runtime thread-count tuning did not
# matter measurably; kept explicit for reproducibility).
_FLAT_WARPS = 8

# Tile width of the masked fallback kernel.
_MASKED_BLOCK = 1024


@triton.jit
def _sigmoid_gate_mul_flat(
    x_ptr, g_ptr, y_ptr, D: tl.constexpr, BLOCK: tl.constexpr
):
    """Whole-tensor flat pass; each element gathers its row gate via //D."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    g = tl.sigmoid(tl.load(g_ptr + offs // D).to(tl.float32))
    tl.store(y_ptr + offs, (x * g).to(y_ptr.dtype.element_ty))


@triton.jit
def _sigmoid_gate_mul_masked(x_ptr, g_ptr, y_ptr, D, BLOCK: tl.constexpr):
    """Masked 2-D kernel for shapes the flat tiling cannot divide evenly."""
    pid_d = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)

    g = tl.sigmoid(tl.load(g_ptr + row).to(tl.float32))

    cols = pid_d * BLOCK + tl.arange(0, BLOCK)
    mask = cols < D
    offs = row * D + cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs, (x * g).to(y_ptr.dtype.element_ty), mask=mask)


def _flat_block(numel, D):
    """Largest power-of-two tile that divides ``numel``, or 0 if unusable.

    Any power of two divides evenly, so ``numel % block == 0`` guarantees the
    flat kernel needs no bounds mask. The tile need not divide ``D``: the block
    may straddle row boundaries and ``offs // D`` still resolves each element's
    row exactly.
    """
    if numel <= 0 or numel >= _MAX_INT32_ELEMS or D <= 0:
        return 0
    block = _FLAT_BLOCK
    while block > _MIN_FLAT_BLOCK and numel % block != 0:
        block >>= 1
    return block if numel % block == 0 and block >= _MIN_FLAT_BLOCK else 0


def sigmoid_gate_mul_broadcast(x, gate):
    x = x.contiguous()
    gate = gate.reshape(-1).contiguous()
    numel = x.numel()
    out = torch.empty_like(x)
    if numel == 0:
        return out

    D = x.shape[-1] if x.dim() else 1
    block = _flat_block(numel, D)
    if block:
        _sigmoid_gate_mul_flat[(numel // block,)](
            x, gate, out, D=D, BLOCK=block, num_warps=_FLAT_WARPS
        )
    else:
        rows = numel // D if D else 0
        grid = (triton.cdiv(D, _MASKED_BLOCK), rows)
        _sigmoid_gate_mul_masked[grid](x, gate, out, D, BLOCK=_MASKED_BLOCK)
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]
