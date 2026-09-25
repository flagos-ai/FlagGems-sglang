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
# implied. See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_gate_add(
    R,
    U,
    G,
    O,
    N: tl.constexpr,
    D: tl.constexpr,
    BROADCAST: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N
    gidx = i % D if BROADCAST else i
    r = tl.load(R + i, valid, 0).to(tl.float32)
    u = tl.load(U + i, valid, 0).to(tl.float32)
    g = tl.load(G + gidx, valid, 0).to(tl.float32)
    product = (u * g).to(O.dtype.element_ty).to(tl.float32)
    result = r + product
    tl.store(O + i, result.to(O.dtype.element_ty), valid)


def residual_gate_add(residual, update, gate):
    output = torch.empty(
        residual.shape, dtype=residual.dtype, device=residual.device
    )
    n = residual.numel()
    if n:
        _residual_gate_add[(triton.cdiv(n, 1024),)](
            residual,
            update,
            gate,
            output,
            n,
            residual.shape[-1],
            gate.numel() != n,
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


__all__ = ["residual_gate_add"]
