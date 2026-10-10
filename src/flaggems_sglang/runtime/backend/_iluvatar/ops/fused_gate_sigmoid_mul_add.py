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
def _contig(X, W, S, F, Out, D: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * D + col * 1, col < D, other=0).to(tl.float32)
    w = tl.load(W + col * 1, col < D, other=0).to(tl.float32)
    gate = tl.sum(x * w, 0)
    scale = 1.0 / (1.0 + tl.exp(-gate))
    s = tl.load(S + row * D + col * 1, col < D, other=0).to(tl.float32)
    f = tl.load(F + row * D + col * 1, col < D, other=0).to(tl.float32)
    tl.store(Out + row * D + col, f + scale * s, col < D)


@triton.jit
def _row(X, W, S, F, Out, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * C[0][0] + col * C[0][1], col < D, other=0).to(
        tl.float32
    )
    w = tl.load(W + col * C[1][0], col < D, other=0).to(tl.float32)
    gate = tl.sum(x * w, 0)
    scale = 1.0 / (1.0 + tl.exp(-gate))
    s = tl.load(S + row * C[2][0] + col * C[2][1], col < D, other=0).to(
        tl.float32
    )
    f = tl.load(F + row * C[3][0] + col * C[3][1], col < D, other=0).to(
        tl.float32
    )
    tl.store(Out + row * D + col, f + scale * s, col < D)


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
    warps = 8 if b <= 1024 else 16 if b <= 2048 else 32
    if (
        hidden_states.is_contiguous()
        and gate_weight.is_contiguous()
        and shared_output.is_contiguous()
        and final_hidden_states.is_contiguous()
    ):
        _contig.run(
            hidden_states,
            gate_weight,
            shared_output,
            final_hidden_states,
            output,
            d,
            b,
            grid=(n,),
            warmup=False,
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
    else:
        _row.run(
            hidden_states,
            gate_weight,
            shared_output,
            final_hidden_states,
            output,
            d,
            (
                hidden_states.stride(),
                gate_weight.stride(),
                shared_output.stride(),
                final_hidden_states.stride(),
            ),
            b,
            grid=(n,),
            warmup=False,
            num_warps=warps,
            num_stages=1,
            enable_fp_fusion=False,
        )
    return output


__all__ = ["fused_gate_sigmoid_mul_add"]
