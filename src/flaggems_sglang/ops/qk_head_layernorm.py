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

"""Fused affine-free LayerNorm over the head dimension of Q and K."""

import torch
import triton
import triton.language as tl

_ROWS, _WARPS = 4, 4


@triton.jit
def _qk_head_ln_kernel(
    q_ptr,
    k_ptr,
    qo_ptr,
    ko_ptr,
    N_ROWS,
    DH: tl.constexpr,  # true head dim (mean/var divisor, column bound)
    eps,
    BLOCK: tl.constexpr,  # next_pow2(DH), row is padded + masked up to it
    ROWS: tl.constexpr,  # rows of (B*S*H) processed per program
    EVEN: tl.constexpr,  # N_ROWS % ROWS == 0: row mask compiled out
):
    block_id = tl.program_id(0)
    is_k = tl.program_id(1)  # grid axis 1 selects the tensor

    rows = block_id * ROWS + tl.arange(0, ROWS)
    # Contiguity proof for vectorized access; exact for every BLOCK.
    cols = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    offs = rows[:, None] * DH + cols[None, :]
    PADDED: tl.constexpr = DH != BLOCK

    if PADDED:
        if EVEN:
            mask = (cols < DH)[None, :]
        else:
            mask = (rows < N_ROWS)[:, None] & (cols < DH)[None, :]
        if is_k:
            x = tl.load(
                k_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
        else:
            x = tl.load(
                q_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg"
            ).to(tl.float32)
        mean = tl.sum(x, axis=1) / DH
        d = tl.where(mask, x - mean[:, None], 0.0)
        var = tl.sum(d * d, axis=1) / DH
        y = d * tl.rsqrt(var + eps)[:, None]
        y = y.to(qo_ptr.dtype.element_ty)
        if is_k:
            tl.store(ko_ptr + offs, y, mask=mask, cache_modifier=".cg")
        else:
            tl.store(qo_ptr + offs, y, mask=mask, cache_modifier=".cg")
    elif EVEN:
        # Power-of-two head dim and row count divisible by ROWS: no masks
        # anywhere. This is the shape every benchmark case compiles to.
        if is_k:
            x = tl.load(k_ptr + offs, cache_modifier=".cg").to(tl.float32)
        else:
            x = tl.load(q_ptr + offs, cache_modifier=".cg").to(tl.float32)
        mean = tl.sum(x, axis=1) / DH
        d = x - mean[:, None]
        var = tl.sum(d * d, axis=1) / DH
        y = d * tl.rsqrt(var + eps)[:, None]
        y = y.to(qo_ptr.dtype.element_ty)
        if is_k:
            tl.store(ko_ptr + offs, y, cache_modifier=".cg")
        else:
            tl.store(qo_ptr + offs, y, cache_modifier=".cg")
    else:
        rmask = rows < N_ROWS
        if is_k:
            x = tl.load(
                k_ptr + offs,
                mask=rmask[:, None],
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
        else:
            x = tl.load(
                q_ptr + offs,
                mask=rmask[:, None],
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
        mean = tl.sum(x, axis=1) / DH
        d = x - mean[:, None]
        var = tl.sum(d * d, axis=1) / DH
        y = d * tl.rsqrt(var + eps)[:, None]
        y = y.to(qo_ptr.dtype.element_ty)
        if is_k:
            tl.store(
                ko_ptr + offs, y, mask=rmask[:, None], cache_modifier=".cg"
            )
        else:
            tl.store(
                qo_ptr + offs, y, mask=rmask[:, None], cache_modifier=".cg"
            )


def qk_head_layernorm(q, k, eps):
    # Per-head LayerNorm, no affine: the last dim is the head dim.
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    dh = q.shape[-1]
    n_rows = q.numel() // dh
    if n_rows == 0:
        return q_out, k_out

    rows = _ROWS
    n_blocks = triton.cdiv(n_rows, rows)
    block = triton.next_power_of_2(dh)
    # grid axis 0: row blocks; axis 1: Q (0) / K (1).
    _qk_head_ln_kernel[(n_blocks, 2)](
        q,
        k,
        q_out,
        k_out,
        n_rows,
        dh,
        eps,
        BLOCK=block,
        ROWS=rows,
        EVEN=(n_rows % rows == 0),
        num_warps=_WARPS,
    )
    return q_out, k_out


__all__ = ["qk_head_layernorm"]
