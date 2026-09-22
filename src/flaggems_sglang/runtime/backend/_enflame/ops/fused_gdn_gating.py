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

"""Task 53: Enflame large contiguous vector tiles with grid-strided persistence.

Published float32 gating formula with stable softplus.
Scheduling informed by FlagGems GCU400 softplus/sigmoid at
34bd6d68928c8b0039c42987840929da52aa6a62 (Apache-2.0).
No FlagGems runtime dependency or fallback.
SPDX-License-Identifier: Apache-2.0
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_flat(
    A,
    X,
    B,
    Bias,
    G,
    Y,
    N: tl.constexpr,
    H: tl.constexpr,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    ALS: tl.constexpr,
    DBS: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLOCK: tl.constexpr,
    I64: tl.constexpr,
):
    for block_id in tl.range(
        tl.program_id(0), tl.cdiv(N, BLOCK), tl.num_programs(0)
    ):
        block = block_id.to(tl.int64 if I64 else tl.int32)
        index = block * BLOCK + tl.arange(0, BLOCK)
        row = index // H
        head = index % H
        mask = index < N
        a_log = tl.load(A + head * ALS, mask, other=0).to(tl.float32)
        a = tl.load(X + row * AS0 + head * AS1, mask, other=0).to(tl.float32)
        b = tl.load(B + row * BS0 + head * BS1, mask, other=0).to(tl.float32)
        bias = tl.load(Bias + head * DBS, mask, other=0).to(tl.float32)
        x = a + bias
        softplus = tl.where(
            BETA * x <= min(THRESHOLD, 20.0),
            (1.0 / BETA)
            * (
                tl.maximum(BETA * x, 0.0)
                + tl.log(1.0 + tl.exp(-tl.abs(BETA * x)))
            ),
            x,
        )
        gate = -tl.exp(a_log) * softplus
        output_gate = tl.sigmoid(b)
        tl.store(G + index, gate, mask)
        tl.store(Y + index, output_gate, mask)


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    batch, heads = a.shape
    g = torch.empty((1, batch, heads), dtype=torch.float32, device=a.device)
    output_gate = torch.empty(
        (1, batch, heads), dtype=torch.float32, device=b.device
    )
    n = batch * heads
    if n == 0:
        return g, output_gate
    block = (
        1024
        if n <= 1024
        else triton.next_power_of_2(n) if n <= 32768 else 65536
    )
    grid = min(triton.cdiv(n, block), 48)
    i64 = n >= 2147483648 or any(
        1 + sum((d - 1) * abs(st) for d, st in zip(t.shape, t.stride()))
        >= 2147483648
        for t in (A_log, a, b, dt_bias)
    )
    _gdn_flat[(grid,)](
        A_log,
        a,
        b,
        dt_bias,
        g,
        output_gate,
        n,
        heads,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        A_log.stride(0),
        dt_bias.stride(0),
        beta,
        threshold,
        block,
        i64,
        num_warps=4,
    )
    return g, output_gate


__all__ = ["fused_gdn_gating"]
