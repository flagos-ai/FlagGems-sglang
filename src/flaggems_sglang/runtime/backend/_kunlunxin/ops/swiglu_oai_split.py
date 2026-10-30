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
def _kernel(
    X,
    Y,
    N: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    A: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    if N % B == 0 and C == 1:
        gate = tl.load(X + row * S + col, offset_state_policy="continuous").to(
            tl.float32
        )
        up = tl.load(
            X + row * S + N + col, offset_state_policy="continuous"
        ).to(tl.float32)
    else:
        gate = tl.load(X + row * S + col * C, col < N, other=0).to(tl.float32)
        up = tl.load(X + row * S + (col + N) * C, col < N, other=0).to(
            tl.float32
        )
    if LIMIT is not None:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    result = gate * (1.0 / (1.0 + tl.exp(gate * (-A)))) * (up + BETA)
    if N % B == 0 and C == 1:
        tl.store(Y + row * N + col, result, offset_state_policy="continuous")
    else:
        tl.store(Y + row * N + col, result, col < N)


def swiglu_oai_split(gate_up, alpha, beta, limit, out_dtype):
    x = (
        gate_up
        if gate_up.ndim == 2
        else gate_up.reshape(-1, gate_up.shape[-1])
    )
    m, width = x.shape
    n = width // 2
    out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    if m and n:
        b = min(4096, triton.next_power_of_2(n))
        _kernel.run(
            x,
            out,
            n,
            x.stride(0),
            x.stride(1),
            alpha,
            beta,
            limit,
            b,
            grid=(m, (n + b - 1) // b),
            warmup=False,
            num_warps=4,
            num_stages=1,
            isCloseMemoryAsync=False,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["swiglu_oai_split"]
