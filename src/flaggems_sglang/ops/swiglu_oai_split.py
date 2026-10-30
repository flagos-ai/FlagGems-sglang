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
    SIZE: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    A: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    B: tl.constexpr,
):
    for start in range(tl.program_id(0) * B, SIZE, tl.num_programs(0) * B):
        idx = start + tl.arange(0, B)
        row = idx // N
        col = idx % N
        off = row * S + col * C
        if SIZE % B == 0:
            gate = tl.load(X + off, cache_modifier=".cg").to(tl.float32)
            up = tl.load(X + off + N * C, cache_modifier=".cg").to(tl.float32)
        else:
            gate = tl.load(
                X + off, idx < SIZE, other=0, cache_modifier=".cg"
            ).to(tl.float32)
            up = tl.load(
                X + off + N * C, idx < SIZE, other=0, cache_modifier=".cg"
            ).to(tl.float32)
        if LIMIT is not None:
            gate = tl.minimum(gate, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        sig = 1.0 / (1.0 + tl.exp(-(A * gate)))
        result = gate * sig * (up + BETA)
        if SIZE % B == 0:
            tl.store(Y + idx, result)
        else:
            tl.store(Y + idx, result, idx < SIZE)


def swiglu_oai_split(gate_up, alpha, beta, limit, out_dtype):
    x = (
        gate_up
        if gate_up.ndim == 2
        else gate_up.reshape(-1, gate_up.shape[-1])
    )
    m, width = x.shape
    n = width // 2
    out = x.new_empty((m, n), dtype=out_dtype)
    size = m * n
    if size:
        b = 256 if size <= 65536 else 1024
        _kernel.run(
            x,
            out,
            n,
            size,
            x.stride(0),
            x.stride(1),
            alpha,
            beta,
            limit,
            b,
            grid=(min((size + b - 1) // b, 4096),),
            warmup=False,
            num_warps=4,
        )
    return out


__all__ = ["swiglu_oai_split"]
