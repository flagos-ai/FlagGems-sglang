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
def _scale_flat(
    X, T, Y, N: tl.constexpr, INNER: tl.constexpr, BLOCK: tl.constexpr
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    x = tl.load(X + offset, mask=valid, other=0.0).to(tl.float32)
    tau = tl.load(T + offset // INNER, mask=valid, other=0.0).to(tl.float32)
    tl.store(Y + offset, (x * tau).to(Y.dtype.element_ty), mask=valid)


def log_scaling_tau(x, tau):
    source = x.contiguous()
    out = torch.empty_like(source)
    n = source.numel()
    if n == 0:
        return out
    rows = source.shape[0]
    inner = n // rows
    scale = tau.reshape(rows).contiguous()
    block = min(16384, triton.next_power_of_2(n))
    _scale_flat[(triton.cdiv(n, block),)](
        source,
        scale,
        out,
        N=n,
        INNER=inner,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    return out


__all__ = ["log_scaling_tau"]
