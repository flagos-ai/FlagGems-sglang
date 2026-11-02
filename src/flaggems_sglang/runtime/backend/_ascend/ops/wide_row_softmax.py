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

"""wide_row_softmax -- Triton kernel (FlagOS S2 Track-1, Batch 8, Task 126).

Version: qoder r5 (2026-10-05) -- ascend: SINGLE kernel (one launch, no pmax/psum scratch) when the row fits 8192 lanes
the column tiles' (max, sumexp) partials is folded into the normalise program, which
removes one launch and one global write/read round trip.  H2 probe: BLOCK_V=4096 with num_warps=4 (half as many column
    tiles per row) -- this chip is the experiment.
"""
import torch
import triton
import triton.language as tl

OP = "wide_row_softmax"
NEG_INF = tl.constexpr(float("-inf"))


@triton.jit
def _wsr_partial(
    logits_ptr,
    temp_ptr,
    pmax_ptr,
    psum_ptr,
    L_S0: tl.constexpr,
    L_S1: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    cb = pid - b * C
    cols = cb * BLOCK_V + tl.arange(0, BLOCK_V)
    cm = cols < V
    csafe = tl.where(cm, cols, V - 1)
    x = tl.load(logits_ptr + b * L_S0 + csafe * L_S1)
    if HAS_TEMP:
        x = x / tl.load(temp_ptr + b)
    x = tl.where(cm, x, NEG_INF)
    lm = tl.max(x)
    e = tl.exp(x - lm)
    e = tl.where(cm, e, 0.0)
    lse = tl.sum(e)
    tl.store(pmax_ptr + b * C + cb, lm)
    tl.store(psum_ptr + b * C + cb, lse)


@triton.jit
def _wsr_norm(
    logits_ptr,
    temp_ptr,
    pmax_ptr,
    psum_ptr,
    out_ptr,
    L_S0: tl.constexpr,
    L_S1: tl.constexpr,
    O_S0: tl.constexpr,
    O_S1: tl.constexpr,
    V: tl.constexpr,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // C
    cb = pid - b * C

    # ---- merge this row's C partials here (was a separate finalise launch) ----
    offs = tl.arange(0, BLOCK_C)
    m = offs < C
    osafe = tl.where(m, offs, 0)  # lane 0 always in-bounds
    pm = tl.load(pmax_ptr + b * C + osafe)
    pm = tl.where(m, pm, NEG_INF)
    gmax = tl.max(pm)
    ps = tl.load(psum_ptr + b * C + osafe)
    ps = tl.where(m, ps, 0.0)
    gsum = tl.sum(ps * tl.exp(pm - gmax))

    # ---- normalise this tile ----
    cols = cb * BLOCK_V + tl.arange(0, BLOCK_V)
    cm = cols < V
    csafe = tl.where(cm, cols, V - 1)
    x = tl.load(logits_ptr + b * L_S0 + csafe * L_S1)
    if HAS_TEMP:
        x = x / tl.load(temp_ptr + b)
    y = tl.exp(x - gmax) / gsum
    tl.store(out_ptr + b * O_S0 + cols * O_S1, y, mask=cm)


@triton.jit
def _wsr_one(
    logits_ptr,
    temp_ptr,
    out_ptr,
    L_S0: tl.constexpr,
    L_S1: tl.constexpr,
    O_S0: tl.constexpr,
    O_S1: tl.constexpr,
    V: tl.constexpr,
    HAS_TEMP: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Whole row in one program: max, sumexp and the normalised store in ONE launch.

    No pmax/psum scratch and no second kernel -- the point of this file is to stop paying
    per-launch and per-round-trip cost on a dispatch-bound backend.
    """
    b = tl.program_id(0)
    cols = tl.arange(0, BLOCK_V)
    cm = cols < V
    csafe = tl.where(cm, cols, V - 1)  # address always in-bounds
    x = tl.load(logits_ptr + b * L_S0 + csafe * L_S1)
    if HAS_TEMP:
        x = x / tl.load(temp_ptr + b)
    x = tl.where(cm, x, NEG_INF)  # register mask, never other=
    gmax = tl.max(x)
    e = tl.exp(x - gmax)
    e = tl.where(cm, e, 0.0)
    gsum = tl.sum(e)
    tl.store(out_ptr + b * O_S0 + cols * O_S1, e / gsum, mask=cm)


def wide_row_softmax(logits, temperatures, out):
    B = int(logits.shape[0])
    V = int(logits.shape[1])
    dev = logits.device
    if B == 0 or V == 0:
        return out
    base_block = 4096
    block_v = min(base_block, max(1, triton.next_power_of_2(V)))
    C = int(triton.cdiv(V, block_v))
    single_v = min(max(1, triton.next_power_of_2(V)), 8192)
    if single_v >= V:
        # one tile holds the whole row -> one kernel, no scratch, no merge
        L_S0 = int(logits.stride(0))
        L_S1 = int(logits.stride(1))
        O_S0 = int(out.stride(0))
        O_S1 = int(out.stride(1))
        has_temp = temperatures is not None
        if has_temp:
            temp = temperatures.to(torch.float32)
            if not temp.is_contiguous():
                temp = temp.contiguous()
            temp = temp.reshape(-1)
        else:
            temp = logits
        nw_s = 8 if single_v >= 2048 else (4 if single_v >= 512 else 1)
        _wsr_one[(B,)](
            logits,
            temp,
            out,
            L_S0=L_S0,
            L_S1=L_S1,
            O_S0=O_S0,
            O_S1=O_S1,
            V=V,
            HAS_TEMP=has_temp,
            BLOCK_V=single_v,
            num_warps=nw_s,
        )
        return out
    block_c = int(triton.next_power_of_2(max(C, 1)))
    pmax = torch.empty((B, C), device=dev, dtype=torch.float32)
    psum = torch.empty((B, C), device=dev, dtype=torch.float32)
    has_temp = temperatures is not None
    if has_temp:
        temp = temperatures.to(torch.float32)
        if not temp.is_contiguous():
            temp = temp.contiguous()
        temp = temp.reshape(-1)
    else:
        temp = pmax  # dummy, never dereferenced
    L_S0 = int(logits.stride(0))
    L_S1 = int(logits.stride(1))
    O_S0 = int(out.stride(0))
    O_S1 = int(out.stride(1))
    nw = 4 if block_v >= 4096 else (4 if block_v >= 512 else 1)
    grid = (B * C,)
    _wsr_partial[grid](
        logits,
        temp,
        pmax,
        psum,
        L_S0=L_S0,
        L_S1=L_S1,
        V=V,
        C=C,
        HAS_TEMP=has_temp,
        BLOCK_V=block_v,
        num_warps=nw,
    )
    _wsr_norm[grid](
        logits,
        temp,
        pmax,
        psum,
        out,
        L_S0=L_S0,
        L_S1=L_S1,
        O_S0=O_S0,
        O_S1=O_S1,
        V=V,
        C=C,
        BLOCK_C=block_c,
        HAS_TEMP=has_temp,
        BLOCK_V=block_v,
        num_warps=nw,
    )
    return out


__all__ = ["wide_row_softmax"]
