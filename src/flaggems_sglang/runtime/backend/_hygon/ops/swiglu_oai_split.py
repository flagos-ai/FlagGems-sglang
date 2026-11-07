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
from triton.language.extra import libdevice


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
    B: tl.constexpr,
    Q: tl.constexpr,
):
    for start in range(tl.program_id(0) * Q, TOTAL, tl.num_programs(0) * Q):
        for q in tl.static_range(Q):
            tile = start + q
            if TOTAL % Q == 0 or tile < TOTAL:
                row = tile // NB
                col = (tile % NB) * B + tl.arange(0, B)
                if N % B == 0:
                    gate = tl.load(X + row * S + col * C).to(tl.float32)
                    up = tl.load(X + row * S + (col + N) * C).to(tl.float32)
                else:
                    edge = col < N
                    gate = tl.load(X + row * S + col * C, edge, other=0).to(
                        tl.float32
                    )
                    up = tl.load(
                        X + row * S + (col + N) * C, edge, other=0
                    ).to(tl.float32)
                if LIMIT is not None:
                    gate = tl.minimum(gate, LIMIT)
                    up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
                denom = tl.fma(
                    tl.exp(-(A * gate)),
                    5.960464477539063e-08,
                    5.960464477539063e-08,
                )
                sig = (
                    libdevice.fast_dividef(1.0, denom) * 5.960464477539063e-08
                )
                result = gate * sig * (up + BETA)
                if N % B == 0:
                    tl.store(Y + row * N + col, result)
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
    out = x.new_empty((m, n), dtype=out_dtype)
    if m and n:
        b = min(1024, triton.next_power_of_2(n))
        nb = (n + b - 1) // b
        total = m * nb
        q = 2 if total > 16384 else 1
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
            b,
            q,
            grid=(min((total + q - 1) // q, 4096),),
            warmup=False,
            num_warps=8 if total > 16384 else 4,
        )
    return out


__all__ = ["swiglu_oai_split"]
