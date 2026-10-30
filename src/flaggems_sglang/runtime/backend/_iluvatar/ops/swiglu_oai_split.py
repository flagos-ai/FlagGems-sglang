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
    SIZE: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,
    A: tl.constexpr,
    BETA: tl.constexpr,
    LIMIT: tl.constexpr,
    B: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    for chunk in tl.static_range(CHUNKS):
        idx = (tl.program_id(0) * CHUNKS + chunk) * B + tl.arange(0, B)
        row = idx // N
        col = idx % N
        off = row * S + col * C
        if SIZE % (B * CHUNKS) == 0:
            gate = tl.load(X + off).to(tl.float32)
            up = tl.load(X + off + N * C).to(tl.float32)
        else:
            gate = tl.load(X + off, idx < SIZE, other=0).to(tl.float32)
            up = tl.load(X + off + N * C, idx < SIZE, other=0).to(tl.float32)
        if LIMIT is not None:
            gate = tl.minimum(gate, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        sig = 1.0 / (1.0 + tl.exp(-(A * gate)))
        result = gate * sig * (up + BETA)
        if SIZE % (B * CHUNKS) == 0:
            tl.store(Y + idx, result)
        else:
            tl.store(Y + idx, result, idx < SIZE)


def swiglu_oai_split(gate_up, alpha, beta, limit, out_dtype):
    width = gate_up.shape[-1]
    x = gate_up if gate_up.ndim == 2 else gate_up.reshape(-1, width)
    m = x.shape[0]
    n = width // 2
    out = torch.empty((m, n), device=x.device, dtype=out_dtype or x.dtype)
    size = m * n
    if size:
        b = 512 if size <= 65536 else 1024
        chunks = 2 if size >= 4194304 else 1
        _kernel[((size + b * chunks - 1) // (b * chunks),)](
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
            chunks,
            num_warps=8 if size >= 4194304 else 4,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["swiglu_oai_split"]
