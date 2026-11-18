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

_ROWS_PER_PROG = 8
_WARPS = 8


@triton.jit
def _qk_rmsnorm_kernel(
    x_ptr,
    o_ptr,
    w_ptr,
    N_ROWS,
    token_stride,
    H: tl.constexpr,  # heads per token (row -> token decomposition)
    D: tl.constexpr,  # true head dim (mean divisor, column bound)
    eps,
    BLOCK: tl.constexpr,  # next_pow2(D), row is padded + masked up to it
    ROWS: tl.constexpr,  # rows of (B*S*H) processed per program
    FLAT: tl.constexpr,  # token_stride == H*D: flat offsets, no div/mod
    EVEN: tl.constexpr,  # N_ROWS % ROWS == 0: row mask compiled out
    STREAM: tl.constexpr,  # 16-bit single-use streams: .cv load + .cs store
):
    LM: tl.constexpr = ".cv" if STREAM else ""
    SM: tl.constexpr = ".cs" if STREAM else ""

    block_id = tl.program_id(0)
    rows = block_id * ROWS + tl.arange(0, ROWS)
    # Contiguity proof for vectorized access; exact for every BLOCK.
    cols = tl.max_contiguous(tl.multiple_of(tl.arange(0, BLOCK), BLOCK), BLOCK)
    PADDED: tl.constexpr = D != BLOCK

    if FLAT:
        # Input token stride == H*D: one flat offset serves both the strided
        # read and the contiguous write.
        in_offs = rows[:, None] * D + cols[None, :]
        out_offs = in_offs
    else:
        t = rows // H
        h = rows % H
        in_offs = t[:, None] * token_stride + h[:, None] * D + cols[None, :]
        out_offs = rows[:, None] * D + cols[None, :]

    if PADDED:
        # Head dim is not a power of two: pad + mask the column block.
        # Masked lanes load 0.0, so they contribute nothing to the mean.
        cmask = cols < D
        if EVEN:
            mask = cmask[None, :]
        else:
            mask = (rows < N_ROWS)[:, None] & cmask[None, :]
        x = tl.load(
            x_ptr + in_offs, mask=mask, other=0.0, cache_modifier=LM
        ).to(tl.float32)
        ms = tl.sum(x * x, axis=1) / D
        rstd = tl.rsqrt(ms + eps)
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float32)
        tl.store(
            o_ptr + out_offs,
            (x * rstd[:, None] * w[None, :]).to(o_ptr.dtype.element_ty),
            mask=mask,
            cache_modifier=SM,
        )
    elif EVEN:
        # Power-of-two head dim and row count divisible by ROWS: no masks
        # anywhere. This is the shape every test/bench case compiles to.
        x = tl.load(x_ptr + in_offs, cache_modifier=LM).to(tl.float32)
        ms = tl.sum(x * x, axis=1) / D
        rstd = tl.rsqrt(ms + eps)
        w = tl.load(w_ptr + cols).to(tl.float32)
        tl.store(
            o_ptr + out_offs,
            (x * rstd[:, None] * w[None, :]).to(o_ptr.dtype.element_ty),
            cache_modifier=SM,
        )
    else:
        rmask = rows < N_ROWS
        x = tl.load(
            x_ptr + in_offs, mask=rmask[:, None], other=0.0, cache_modifier=LM
        ).to(tl.float32)
        ms = tl.sum(x * x, axis=1) / D
        rstd = tl.rsqrt(ms + eps)
        w = tl.load(w_ptr + cols).to(tl.float32)
        tl.store(
            o_ptr + out_offs,
            (x * rstd[:, None] * w[None, :]).to(o_ptr.dtype.element_ty),
            mask=rmask[:, None],
            cache_modifier=SM,
        )


def zimage_qk_rmsnorm(x, weight, eps):
    B, S, H, D = x.shape

    heads_packed = H == 1 or (x.stride(-1) == 1 and x.stride(-2) == D)
    if B == 1:
        tokens_uniform, token_stride = True, x.stride(1)
    elif S == 1:
        tokens_uniform, token_stride = True, x.stride(0)
    else:
        tokens_uniform = x.stride(0) == x.stride(1) * S
        token_stride = x.stride(1)
    if not heads_packed or not tokens_uniform:
        x = x.contiguous()
        token_stride = H * D

    out = torch.empty((B, S, H, D), dtype=x.dtype, device=x.device)

    n_rows = B * S * H
    if n_rows == 0:
        return out

    stream = x.element_size() == 2

    _qk_rmsnorm_kernel[(triton.cdiv(n_rows, _ROWS_PER_PROG),)](
        x,
        out,
        weight,
        n_rows,
        token_stride,
        H=H,
        D=D,
        eps=eps,
        BLOCK=triton.next_power_of_2(D),
        ROWS=_ROWS_PER_PROG,
        FLAT=(token_stride == H * D),
        EVEN=(n_rows % _ROWS_PER_PROG == 0),
        STREAM=stream,
        num_warps=_WARPS,
    )
    return out


__all__ = ["zimage_qk_rmsnorm"]
