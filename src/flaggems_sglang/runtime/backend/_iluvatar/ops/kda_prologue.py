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


@triton.jit(
    do_not_specialize=["Q", "K", "G", "CU", "CI", "A", "B", "C", "D", "E", "F"]
)
def _kernel(Q, K, G, CU, CI, A, B, C, D, E, F, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    W: tl.constexpr = META[2]
    QS: tl.constexpr = META[3]
    KS: tl.constexpr = META[4]
    GS: tl.constexpr = META[5]
    CS: tl.constexpr = META[6]
    CIS: tl.constexpr = META[7]
    SCALE: tl.constexpr = META[8]
    BC: tl.constexpr = META[9]
    chunk = tl.program_id(0)
    channels = tl.program_id(1) * BC + tl.arange(0, BC)
    head = channels // W
    col = channels % W
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    row = bos + cid * 64 + tl.arange(0, 64)
    valid = (
        (row[:, None] < eos) & (row[:, None] < T) & (channels[None, :] < H * W)
    )
    g = tl.load(
        G
        + row[:, None] * GS[0]
        + head[None, :] * GS[1]
        + col[None, :] * GS[2],
        valid,
        0,
    ).to(tl.float32)
    gc = tl.cumsum(g, 0)
    last = tl.sum(g, 0)
    diff = gc - last[None, :]
    eg = tl.exp(gc)
    ep = tl.exp(diff)
    em = tl.exp(-diff)
    q = tl.load(
        Q
        + row[:, None] * QS[0]
        + head[None, :] * QS[1]
        + col[None, :] * QS[2],
        valid,
        0,
    ).to(tl.float32)
    k = tl.load(
        K
        + row[:, None] * KS[0]
        + head[None, :] * KS[1]
        + col[None, :] * KS[2],
        valid,
        0,
    ).to(tl.float32)
    out = row[:, None] * (H * W) + channels[None, :]
    sq = SCALE * q
    tl.store(A + out, k * ep, valid)
    tl.store(B + out, k * em, valid)
    tl.store(C + out, k * eg, valid)
    tl.store(D + out, sq * eg, valid)
    tl.store(E + out, sq * ep, valid)
    tl.store(F + out, gc, valid)


@triton.jit(
    do_not_specialize=["Q", "K", "G", "CU", "CI", "A", "B", "C", "D", "E", "F"]
)
def _recip(Q, K, G, CU, CI, A, B, C, D, E, F, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    W: tl.constexpr = META[2]
    QS: tl.constexpr = META[3]
    KS: tl.constexpr = META[4]
    GS: tl.constexpr = META[5]
    CS: tl.constexpr = META[6]
    CIS: tl.constexpr = META[7]
    SCALE: tl.constexpr = META[8]
    BC: tl.constexpr = META[9]
    chunk = tl.program_id(0)
    channels = tl.program_id(1) * BC + tl.arange(0, BC)
    head = channels // W
    col = channels % W
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    row = bos + cid * 64 + tl.arange(0, 64)
    valid = (
        (row[:, None] < eos) & (row[:, None] < T) & (channels[None, :] < H * W)
    )
    g = tl.load(
        G
        + row[:, None] * GS[0]
        + head[None, :] * GS[1]
        + col[None, :] * GS[2],
        valid,
        0,
    ).to(tl.float32)
    gc = tl.cumsum(g, 0)
    last = tl.sum(g, 0)
    diff = gc - last[None, :]
    eg = tl.exp(gc)
    ep = tl.exp(diff)
    q = tl.load(
        Q
        + row[:, None] * QS[0]
        + head[None, :] * QS[1]
        + col[None, :] * QS[2],
        valid,
        0,
    ).to(tl.float32)
    k = tl.load(
        K
        + row[:, None] * KS[0]
        + head[None, :] * KS[1]
        + col[None, :] * KS[2],
        valid,
        0,
    ).to(tl.float32)
    out = row[:, None] * (H * W) + channels[None, :]
    sq = SCALE * q
    tl.store(A + out, k * ep, valid)
    tl.store(B + out, k / ep, valid)
    tl.store(C + out, k * eg, valid)
    tl.store(D + out, sq * eg, valid)
    tl.store(E + out, sq * ep, valid)
    tl.store(F + out, gc, valid)


@triton.jit
def _schedule_scan(Q, K, G, CU, CI, A, B, C, D, E, F, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    W: tl.constexpr = META[2]
    QS: tl.constexpr = META[3]
    KS: tl.constexpr = META[4]
    GS: tl.constexpr = META[5]
    CS: tl.constexpr = META[6]
    CIS: tl.constexpr = META[7]
    SCALE: tl.constexpr = META[8]
    BC: tl.constexpr = META[9]
    chunk = tl.program_id(0)
    channels = tl.program_id(1) * BC + tl.arange(0, BC)
    head = channels // W
    col = channels % W
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    row = bos + cid * 64 + tl.arange(0, 64)
    valid = (
        (row[:, None] < eos) & (row[:, None] < T) & (channels[None, :] < H * W)
    )
    g = tl.load(
        G
        + row[:, None] * GS[0]
        + head[None, :] * GS[1]
        + col[None, :] * GS[2],
        valid,
        0,
    ).to(tl.float32)
    gc = tl.cumsum(g, 0)
    last = tl.sum(g, 0)
    diff = gc - last[None, :]
    out = row[:, None] * (H * W) + channels[None, :]
    tl.store(F + out, gc, valid)
    ep = tl.exp(diff)
    k = tl.load(
        K
        + row[:, None] * KS[0]
        + head[None, :] * KS[1]
        + col[None, :] * KS[2],
        valid,
        0,
    ).to(tl.float32)
    tl.store(A + out, k * ep, valid)
    tl.store(B + out, k / ep, valid)
    eg = tl.exp(gc)
    tl.store(C + out, k * eg, valid)
    q = tl.load(
        Q
        + row[:, None] * QS[0]
        + head[None, :] * QS[1]
        + col[None, :] * QS[2],
        valid,
        0,
    ).to(tl.float32)
    sq = SCALE * q
    tl.store(D + out, sq * eg, valid)
    tl.store(E + out, sq * ep, valid)


def kda_prologue(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    shape = q.shape
    t, h, w = shape
    hw = h * w
    outputs = [q.new_empty(shape, dtype=torch.bfloat16) for _ in range(5)]
    prefix = q.new_empty(shape, dtype=torch.float32)
    if t and hw and num_chunks:
        if t >= 1024 or (t >= 128 and hw >= 2048):
            kernel, bc, nw = (_schedule_scan, 64, 1)
        elif t >= 128:
            kernel, bc, nw = (_recip, 32, 1)
        else:
            kernel = _kernel
            bc, nw = (8, 2) if hw <= 512 else (64, 4)
        kernel.run(
            q,
            k,
            g_act,
            cu_seqlens,
            chunk_indices,
            outputs[0],
            outputs[1],
            outputs[2],
            outputs[3],
            outputs[4],
            prefix,
            (
                t,
                h,
                w,
                q.stride(),
                k.stride(),
                g_act.stride(),
                cu_seqlens.stride(0),
                chunk_indices.stride(),
                scale,
                bc,
            ),
            num_warps=nw,
            enable_fp_fusion=False,
            grid=(num_chunks, (hw + bc - 1) // bc),
            warmup=False,
        )
    return (*outputs, prefix)


__all__ = ["kda_prologue"]
