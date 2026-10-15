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
def _sigmoid_gate_mul_broadcast_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    gate_s0: tl.constexpr,
    BLOCK: tl.constexpr,
    P: tl.constexpr,
):
    NC: tl.constexpr = triton.cdiv(hidden_size, BLOCK)
    for t in range(triton.cdiv(num_tokens * NC, P)):
        task = t * P + tl.program_id(0)
        row = task // NC
        col = (task % NC) * BLOCK + tl.arange(0, BLOCK)
        g = tl.sigmoid(
            tl.load(
                gate_ptr + row * gate_s0, mask=row < num_tokens, other=0.0
            ).to(tl.float32)
        )
        mask = (row < num_tokens) & (col < hidden_size)
        x = tl.load(x_ptr + row * x_s0 + col * x_s1, mask=mask, other=0.0).to(
            tl.float32
        )
        tl.store(
            out_ptr + row * hidden_size + col,
            (x * g).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def sigmoid_gate_mul_broadcast(x, gate):
    out = torch.empty_like(x)
    num_tokens, hidden_size = x.shape
    if num_tokens * hidden_size == 0:
        return out
    block = min(triton.next_power_of_2(hidden_size), 1024)
    tasks = num_tokens * triton.cdiv(hidden_size, block)
    grid = (min(tasks, 65535),)
    _sigmoid_gate_mul_broadcast_kernel[grid](
        x,
        gate,
        out,
        num_tokens,
        hidden_size,
        x_s0=x.stride(0),
        x_s1=x.stride(1),
        gate_s0=gate.stride(0),
        BLOCK=block,
        P=grid[0],
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["sigmoid_gate_mul_broadcast"]
