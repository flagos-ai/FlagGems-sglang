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


@triton.jit
def _strip_recip(Q, K, G, CU, CI, A, B, C, D, E, F, META: tl.constexpr):
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
    BT: tl.constexpr = META[10]
    chunk = tl.program_id(2)
    segment = tl.program_id(1)
    channels = tl.program_id(0) * BC + tl.arange(0, BC)
    head = channels // W
    col = channels % W
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    row = bos + cid * 64 + segment * BT + tl.arange(0, BT)
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
    last = tl.full((BC,), 0.0, tl.float32)
    before = tl.full((BC,), 0.0, tl.float32)
    for part in tl.static_range(64 // BT):
        rr = bos + cid * 64 + part * BT + tl.arange(0, BT)
        ok = (
            (rr[:, None] < eos)
            & (rr[:, None] < T)
            & (channels[None, :] < H * W)
        )
        gg = tl.load(
            G
            + rr[:, None] * GS[0]
            + head[None, :] * GS[1]
            + col[None, :] * GS[2],
            ok,
            0,
        ).to(tl.float32)
        ss = tl.sum(gg, 0)
        last += ss
        before += tl.where(part < segment, ss, 0.0)
    gc = tl.cumsum(g, 0) + before[None, :]
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
def _packed_strip(Q, K, G, CU, CI, OUT, F, META: tl.constexpr):
    OSIZE: tl.constexpr = META[0]
    T: tl.constexpr = META[1]
    H: tl.constexpr = META[2]
    W: tl.constexpr = META[3]
    QS: tl.constexpr = META[4]
    KS: tl.constexpr = META[5]
    GS: tl.constexpr = META[6]
    CS: tl.constexpr = META[7]
    CIS: tl.constexpr = META[8]
    SCALE: tl.constexpr = META[9]
    BC: tl.constexpr = META[10]
    BT: tl.constexpr = META[11]
    A = OUT
    B = OUT + 1 * OSIZE
    C = OUT + 2 * OSIZE
    D = OUT + 3 * OSIZE
    E = OUT + 4 * OSIZE
    chunk = tl.program_id(2)
    segment = tl.program_id(1)
    channels = tl.program_id(0) * BC + tl.arange(0, BC)
    head = channels // W
    col = channels % W
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    row = bos + cid * 64 + segment * BT + tl.arange(0, BT)
    valid = (
        (row[:, None] < eos) & (row[:, None] < T) & (channels[None, :] < H * W)
    )
    g = tl.full((BT, BC), 0.0, tl.float32)
    last = tl.full((BC,), 0.0, tl.float32)
    before = tl.full((BC,), 0.0, tl.float32)
    for part in tl.static_range(64 // BT):
        rr = bos + cid * 64 + part * BT + tl.arange(0, BT)
        ok = (
            (rr[:, None] < eos)
            & (rr[:, None] < T)
            & (channels[None, :] < H * W)
        )
        gg = tl.load(
            G
            + rr[:, None] * GS[0]
            + head[None, :] * GS[1]
            + col[None, :] * GS[2],
            ok,
            0,
        ).to(tl.float32)
        g = tl.where(part == segment, gg, g)
        ss = tl.sum(gg, 0)
        last += ss
        before += tl.where(part < segment, ss, 0.0)
    gc = tl.cumsum(g, 0) + before[None, :]
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


@triton.jit(do_not_specialize=["CU", "CI", "Q", "K"])
def _dot_strip(Q, K, G, CU, CI, OUT, F, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    W: tl.constexpr = META[2]
    QS: tl.constexpr = META[3]
    KS: tl.constexpr = META[4]
    GS: tl.constexpr = META[5]
    CS: tl.constexpr = META[6]
    CIS: tl.constexpr = META[7]
    SCALE: tl.constexpr = META[8]
    BT: tl.constexpr = META[9]
    BC: tl.constexpr = META[10]
    OSIZE: tl.constexpr = META[11]
    OST: tl.constexpr = META[12]
    channel = tl.program_id(0) * BC + tl.arange(0, BC)
    part = tl.program_id(1)
    chunk = tl.program_id(2)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    lo = tl.load(CU + seq * CS) + cid * 64
    eos = tl.load(CU + (seq + 1) * CS)
    token = tl.arange(0, 64)
    row = lo + token
    valid = (
        (row[:, None] < eos) & (row[:, None] < T) & (channel[None, :] < H * W)
    )
    g = tl.load(
        G
        + row[:, None] * GS[0]
        + channel[None, :] // W * GS[1]
        + channel[None, :] % W * GS[2],
        valid,
        0,
    )
    step = tl.arange(0, 2 * BT)
    triangle = (
        (step[:, None] >= BT) | (part * BT + step[:, None] >= token[None, :])
    ).to(G.dtype.element_ty)
    acc = tl.dot(triangle, g, out_dtype=tl.float32)
    paired = tl.permute(tl.reshape(acc, (2, BT, BC)), (1, 2, 0))
    prefix, last = tl.split(paired)
    row_out = lo + part * BT + tl.arange(0, BT)
    mask = (
        (row_out[:, None] < eos)
        & (row_out[:, None] < T)
        & (channel[None, :] < H * W)
    )
    q = tl.load(
        Q
        + row_out[:, None] * QS[0]
        + channel[None, :] // W * QS[1]
        + channel[None, :] % W * QS[2],
        mask,
        0,
    ).to(tl.float32)
    k = tl.load(
        K
        + row_out[:, None] * KS[0]
        + channel[None, :] // W * KS[1]
        + channel[None, :] % W * KS[2],
        mask,
        0,
    ).to(tl.float32)
    diff = prefix - last
    eg = tl.exp(prefix)
    ep = tl.exp(diff)
    em = tl.exp(-diff)
    fout = row_out[:, None] * H * W + channel[None, :]
    out = row_out[:, None] * OST + channel[None, :]
    size = OSIZE
    sq = SCALE * q
    tl.store(OUT + out, k * ep, mask)
    tl.store(OUT + size + out, k * em, mask)
    tl.store(OUT + 2 * size + out, k * eg, mask)
    tl.store(OUT + 3 * size + out, sq * eg, mask)
    tl.store(OUT + 4 * size + out, sq * ep, mask)
    tl.store(F + fout, prefix, mask)


@triton.jit
def _sched_kernel(Q, K, G, CU, CI, OUT, F, META: tl.constexpr):
    T: tl.constexpr = META[0]
    H: tl.constexpr = META[1]
    W: tl.constexpr = META[2]
    QS: tl.constexpr = META[3]
    KS: tl.constexpr = META[4]
    GS: tl.constexpr = META[5]
    CS: tl.constexpr = META[6]
    CIS: tl.constexpr = META[7]
    SCALE: tl.constexpr = META[8]
    BT: tl.constexpr = META[9]
    BC: tl.constexpr = META[10]
    MODE: tl.constexpr = META[15]
    channel = tl.program_id(0) * BC + tl.arange(0, BC)
    part = tl.program_id(1)
    chunk = tl.program_id(2)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    lo = tl.load(CU + seq * CS) + cid * 64
    eos = tl.load(CU + (seq + 1) * CS)
    token = tl.arange(0, 64)
    row = lo + token
    valid = (
        (row[:, None] < eos) & (row[:, None] < T) & (channel[None, :] < H * W)
    )
    g = tl.load(
        G
        + row[:, None] * GS[0]
        + channel[None, :] // W * GS[1]
        + channel[None, :] % W * GS[2],
        valid,
        0,
    )
    step = tl.arange(0, 2 * BT)
    triangle = tl.where(
        (step[:, None] >= BT) | (part * BT + step[:, None] >= token[None, :]),
        1.0,
        0.0,
    ).to(G.dtype.element_ty)
    acc = tl.dot(triangle, g, out_dtype=tl.float32)
    paired = tl.permute(tl.reshape(acc, (2, BT, BC)), (1, 2, 0))
    prefix, last = tl.split(paired)
    row_out = lo + part * BT + tl.arange(0, BT)
    mask = (
        (row_out[:, None] < eos)
        & (row_out[:, None] < T)
        & (channel[None, :] < H * W)
    )
    fout = row_out[:, None] * H * W + channel[None, :]
    tl.store(F + fout, prefix, mask)
    if MODE == 1:
        q = tl.load(
            Q
            + row_out[:, None] * QS[0]
            + channel[None, :] // W * QS[1]
            + channel[None, :] % W * QS[2],
            mask,
            0,
        ).to(tl.float32)
    diff = prefix - last
    ep = tl.exp(diff)
    k = tl.load(
        K
        + row_out[:, None] * KS[0]
        + channel[None, :] // W * KS[1]
        + channel[None, :] % W * KS[2],
        mask,
        0,
    ).to(tl.float32)
    if MODE == 3:
        kr = k / ep
    else:
        kr = k * tl.exp(-diff)
    wide = tl.program_id(0) * (2 * BC) + tl.arange(0, 2 * BC)
    wide_mask = (
        (row_out[:, None] < eos)
        & (row_out[:, None] < T)
        & (wide[None, :] < 2 * H * W)
    )
    out = row_out[:, None] * (2 * H * W) + wide[None, :]
    pair0 = tl.reshape(
        tl.join((k * ep).to(tl.bfloat16), kr.to(tl.bfloat16)), (BT, 2 * BC)
    )
    tl.store(OUT + out, pair0, wide_mask)
    eg = tl.exp(prefix)
    if MODE != 1:
        q = tl.load(
            Q
            + row_out[:, None] * QS[0]
            + channel[None, :] // W * QS[1]
            + channel[None, :] % W * QS[2],
            mask,
            0,
        ).to(tl.float32)
    sq = SCALE * q
    pair1 = tl.reshape(
        tl.join((k * eg).to(tl.bfloat16), (sq * eg).to(tl.bfloat16)),
        (BT, 2 * BC),
    )
    tl.store(OUT + 2 * T * H * W + out, pair1, wide_mask)
    tl.store(OUT + 4 * T * H * W + fout, sq * ep, mask)


def kda_prologue(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    shape = q.shape
    t, h, w = shape
    hw = h * w
    low = g_act.dtype in (torch.bfloat16, torch.float16)
    prefix = q.new_empty(shape, dtype=torch.float32)
    if t >= 4096:
        if low:
            bc, nw = (512, 16) if hw >= 512 else (128, 4)
            storage = q.new_empty((5, t, h, w), dtype=torch.bfloat16)
            if t and hw and num_chunks:
                _sched_kernel.run(
                    q,
                    k,
                    g_act,
                    cu_seqlens,
                    chunk_indices,
                    storage,
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
                        8,
                        bc,
                        hw,
                        5 * hw,
                        "",
                        False,
                        2,
                    ),
                    grid=((hw + bc - 1) // bc, 8, num_chunks),
                    warmup=False,
                    num_warps=nw,
                    enable_fp_fusion=False,
                )
            ab = storage[:2].view(t, h, w, 2).unbind(-1)
            cd = storage[2:4].view(t, h, w, 2).unbind(-1)
            return (*ab, *cd, storage[4], prefix)
        osize = t * hw + 256
        storage = q.new_empty((5, osize), dtype=torch.bfloat16)
        if t and hw and num_chunks:
            _packed_strip.run(
                q,
                k,
                g_act,
                cu_seqlens,
                chunk_indices,
                storage,
                prefix,
                (
                    osize,
                    t,
                    h,
                    w,
                    q.stride(),
                    k.stride(),
                    g_act.stride(),
                    cu_seqlens.stride(0),
                    chunk_indices.stride(),
                    scale,
                    128,
                    16,
                ),
                grid=((hw + 127) // 128, 4, num_chunks),
                warmup=False,
                num_warps=2,
                enable_fp_fusion=False,
            )
        return (*storage[:, : t * hw].view(5, t, h, w).unbind(0), prefix)
    if low and t < 1024:
        bt, bc = (8, 64) if t < 128 or hw < 2048 else (16, 64)
        storage = q.new_empty((t, 5, h, w), dtype=torch.bfloat16)
        if t and hw and num_chunks:
            _dot_strip.run(
                q,
                k,
                g_act,
                cu_seqlens,
                chunk_indices,
                storage,
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
                    bt,
                    bc,
                    hw,
                    5 * hw,
                ),
                grid=((hw + bc - 1) // bc, 64 // bt, num_chunks),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return (*storage.unbind(1), prefix)
    if t >= 1024:
        osize = t * hw + 256
        storage = q.new_empty((5, osize), dtype=torch.bfloat16)
        if hw and num_chunks:
            _packed_strip.run(
                q,
                k,
                g_act,
                cu_seqlens,
                chunk_indices,
                storage,
                prefix,
                (
                    osize,
                    t,
                    h,
                    w,
                    q.stride(),
                    k.stride(),
                    g_act.stride(),
                    cu_seqlens.stride(0),
                    chunk_indices.stride(),
                    scale,
                    32,
                    32,
                ),
                grid=((hw + 31) // 32, 2, num_chunks),
                warmup=False,
                num_warps=1,
                enable_fp_fusion=False,
            )
        return (*storage[:, : t * hw].view(5, t, h, w).unbind(0), prefix)
    outputs = [q.new_empty(shape, dtype=torch.bfloat16) for _ in range(5)]
    if t and hw and num_chunks:
        if t >= 1024:
            _strip_recip.run(
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
                    64,
                    32,
                ),
                grid=((hw + 63) // 64, 2, num_chunks),
                warmup=False,
                num_warps=2,
                enable_fp_fusion=False,
            )
        else:
            _kernel.run(
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
                    32,
                ),
                grid=(num_chunks, (hw + 31) // 32),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (*outputs, prefix)


__all__ = ["kda_prologue"]
