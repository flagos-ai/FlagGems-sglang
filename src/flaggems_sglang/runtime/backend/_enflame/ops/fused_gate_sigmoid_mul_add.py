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

import torch
import triton
import triton.language as tl


@triton.jit
def _direct(
    X,
    W,
    S,
    F,
    Out,
    N: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    WS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    F0: tl.constexpr,
    F1: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    col = tl.arange(0, B)
    w = tl.load(W + col * WS, col < D, other=0).to(tl.float32)
    start = tl.program_id(0) * R
    row = start + tl.arange(0, R)
    mask = (row[:, None] < N) & (col[None, :] < D)
    x = tl.load(X + row[:, None] * X0 + col[None, :] * X1, mask, other=0).to(
        tl.float32
    )
    gate = tl.sum(x * w[None, :], 1)
    scale = 1.0 / (1.0 + tl.exp(-gate))
    s = tl.load(S + row[:, None] * S0 + col[None, :] * S1, mask, other=0).to(
        tl.float32
    )
    f = tl.load(F + row[:, None] * F0 + col[None, :] * F1, mask, other=0).to(
        tl.float32
    )
    out = f + scale[:, None] * s
    tl.store(Out + row[:, None] * D + col[None, :], out, mask)


def fused_gate_sigmoid_mul_add(
    hidden_states, gate_weight, shared_output, final_hidden_states
):
    n, d = hidden_states.shape
    output = torch.empty(
        (n, d),
        dtype=final_hidden_states.dtype,
        device=final_hidden_states.device,
    )
    if not n or not d:
        return output
    b = 1 << (d - 1).bit_length()
    r = max(1, min(1 << (n - 1).bit_length(), 65536 // b))
    warps = 1
    if n == 1:
        r = 1
    elif n <= 8:
        r, warps = (4, 4) if b >= 8192 else (8, 4)
    elif n <= 64:
        r, warps = 16, 4
    elif n <= 128:
        warps = 2
    elif n <= 1024 and b <= 4096:
        warps = 2
    _direct.run(
        hidden_states,
        gate_weight,
        shared_output,
        final_hidden_states,
        output,
        n,
        d,
        hidden_states.stride(0),
        hidden_states.stride(1),
        gate_weight.stride(0),
        shared_output.stride(0),
        shared_output.stride(1),
        final_hidden_states.stride(0),
        final_hidden_states.stride(1),
        r,
        b,
        grid=((n + r - 1) // r,),
        warmup=False,
        num_warps=warps,
        num_stages=1,
        enable_fp_fusion=False,
    )
    return output


__all__ = ["fused_gate_sigmoid_mul_add"]
