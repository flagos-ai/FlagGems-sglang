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
    NB: tl.constexpr,
    TOTAL: tl.constexpr,
    R: tl.constexpr,
    B: tl.constexpr,
):
    for tile in range(tl.program_id(0), TOTAL, tl.num_programs(0)):
        row = (tile // NB) * R + tl.arange(0, R)
        col = (tile % NB) * B + tl.arange(0, B)
        address = row[:, None] * S + col[None, :] * C
        if N % B == 0:
            gate = tl.load(X + address).to(tl.float32)
            up = tl.load(X + address + N * C).to(tl.float32)
        else:
            edge = col[None, :] < N
            gate = tl.load(X + address, edge, other=0).to(tl.float32)
            up = tl.load(X + address + N * C, edge, other=0).to(tl.float32)
        if LIMIT is not None:
            gate = tl.minimum(gate, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        result = gate * (1.0 / (1.0 + tl.exp(-(A * gate)))) * (up + BETA)
        if N % B == 0:
            tl.store(Y + row[:, None] * N + col[None, :], result)
        else:
            tl.store(
                Y + row[:, None] * N + col[None, :], result, col[None, :] < N
            )


def swiglu_oai_split(gate_up, alpha, beta, limit, out_dtype):
    x = (
        gate_up
        if gate_up.ndim == 2
        else gate_up.reshape(-1, gate_up.shape[-1])
    )
    m, width = x.shape
    n = width // 2
    out = x.new_empty((m, n), dtype=out_dtype)
    if m and n:
        b = min(1024, triton.next_power_of_2(n))
        r = 4 if m % 4 == 0 else 1
        nb = (n + b - 1) // b
        total = (m // r) * nb
        _kernel.run(
            x,
            out,
            n,
            x.stride(0),
            x.stride(1),
            alpha,
            beta,
            limit,
            nb,
            total,
            r,
            b,
            grid=(min(total, 40),),
            warmup=False,
            num_warps=4,
        )
    return out


__all__ = ["swiglu_oai_split"]
