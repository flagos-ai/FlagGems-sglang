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


@triton.jit
def _kernel_2d(
    x_ptr,
    gate_ptr,
    out_ptr,
    hidden_size,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1)
    if WIDE:
        base = row.to(tl.int64) * hidden_size
        offs = base + col * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    else:
        offs = row * hidden_size + col * BLOCK + tl.arange(0, BLOCK)
    g = tl.sigmoid(tl.load(gate_ptr + row).to(tl.float32))
    if EVEN:
        x = tl.load(x_ptr + offs).to(tl.float32)
        tl.store(out_ptr + offs, (x * g).to(out_ptr.dtype.element_ty))
    else:
        if WIDE:
            limit = base + hidden_size
        else:
            limit = (row + 1) * hidden_size
        mask = offs < limit
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + offs, (x * g).to(out_ptr.dtype.element_ty), mask=mask
        )


@triton.jit
def _kernel_flat_rows(
    x_ptr,
    gate_ptr,
    out_ptr,
    numel,
    num_tokens,
    hidden_size,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    EVEN: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    row0 = pid * ROWS
    if WIDE:
        rows = row0.to(tl.int64) + tl.arange(0, ROWS).to(tl.int64)[:, None]
        offs = (
            rows * hidden_size
            + tl.arange(0, BLOCK // ROWS).to(tl.int64)[None, :]
        )
        g = tl.load(gate_ptr + rows, mask=rows < num_tokens, other=0.0).to(
            tl.float32
        )
    else:
        rows = row0 + tl.arange(0, ROWS)[:, None]
        offs = rows * hidden_size + tl.arange(0, BLOCK // ROWS)[None, :]
        g = tl.load(gate_ptr + rows, mask=rows < num_tokens, other=0.0).to(
            tl.float32
        )
    g = tl.sigmoid(g)
    if EVEN:
        x = tl.load(x_ptr + offs).to(tl.float32)
        r = (x * g).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, r)
    else:
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        r = (x * g).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, r, mask=mask)


@triton.jit
def _kernel_flat(
    x_ptr,
    gate_ptr,
    out_ptr,
    numel,
    hidden_size,
    BLOCK: tl.constexpr,
    WIDE: tl.constexpr,
):
    if WIDE:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(
            tl.int64
        )
    else:
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    g = tl.sigmoid(
        tl.load(gate_ptr + offs // hidden_size, mask=mask, other=0.0).to(
            tl.float32
        )
    )
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (x * g).to(out_ptr.dtype.element_ty), mask=mask)


def sigmoid_gate_mul_broadcast(x, gate):
    out = torch.empty_like(x)
    num_tokens, hidden_size = out.shape
    if num_tokens == 0 or hidden_size == 0:
        return out
    if not x.is_contiguous():
        x = x.contiguous()
    gate_flat = gate.reshape(-1)
    if not gate_flat.is_contiguous():
        gate_flat = gate_flat.contiguous()
    numel = num_tokens * hidden_size
    wide = numel >= 2**31

    if hidden_size <= 1024 and numel >= 512 * 1024:
        if hidden_size & (hidden_size - 1) == 0:
            block = 4096
            rows = block // hidden_size
            grid = (triton.cdiv(numel, block),)
            _kernel_flat_rows[grid](
                x,
                gate_flat,
                out,
                numel,
                num_tokens,
                hidden_size,
                BLOCK=block,
                ROWS=rows,
                EVEN=numel % block == 0,
                WIDE=wide,
                num_warps=4,
            )
            return out
        grid = (triton.cdiv(numel, 4096),)
        _kernel_flat[grid](
            x,
            gate_flat,
            out,
            numel,
            hidden_size,
            BLOCK=4096,
            WIDE=wide,
            num_warps=4,
        )
        return out

    if numel >= 4 * 1024 * 1024:
        block = min(triton.next_power_of_2(hidden_size), 8192)
        if block >= 8192:
            num_warps = 2 if num_tokens <= 512 else 16
        else:
            num_warps = 2 if num_tokens <= 512 else 8
        grid = (num_tokens, triton.cdiv(hidden_size, block))
        _kernel_2d[grid](
            x,
            gate_flat,
            out,
            hidden_size,
            BLOCK=block,
            EVEN=hidden_size % block == 0,
            WIDE=wide,
            num_warps=num_warps,
        )
        return out

    if num_tokens <= 64:
        block = min(1024, triton.next_power_of_2(hidden_size))
        num_warps = 2
    else:
        block = min(2048, triton.next_power_of_2(hidden_size))
        num_warps = 2
    grid = (num_tokens, triton.cdiv(hidden_size, block))
    _kernel_2d[grid](
        x,
        gate_flat,
        out,
        hidden_size,
        BLOCK=block,
        EVEN=hidden_size % block == 0,
        WIDE=wide,
        num_warps=num_warps,
    )
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]
