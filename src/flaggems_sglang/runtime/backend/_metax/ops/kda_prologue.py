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
    Q,
    K,
    G,
    CU,
    CI,
    OUT,
    F,
    OSIZE: tl.constexpr,
    OST: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    GS: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    SCALE: tl.constexpr,
    BC: tl.constexpr,
):
    A = OUT
    B = OUT + 1 * OSIZE
    C = OUT + 2 * OSIZE
    D = OUT + 3 * OSIZE
    E = OUT + 4 * OSIZE
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
    fout = row[:, None] * (H * W) + channels[None, :]
    out = row[:, None] * OST + channels[None, :]
    sq = SCALE * q
    tl.store(A + out, k * ep, valid)
    tl.store(B + out, k * em, valid)
    tl.store(C + out, k * eg, valid)
    tl.store(D + out, sq * eg, valid)
    tl.store(E + out, sq * ep, valid)
    tl.store(F + fout, gc, valid)


@triton.jit
def _recip(
    Q,
    K,
    G,
    CU,
    CI,
    OUT,
    F,
    OSIZE: tl.constexpr,
    OST: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    GS: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    SCALE: tl.constexpr,
    BC: tl.constexpr,
):
    A = OUT
    B = OUT + 1 * OSIZE
    C = OUT + 2 * OSIZE
    D = OUT + 3 * OSIZE
    E = OUT + 4 * OSIZE
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
    fout = row[:, None] * (H * W) + channels[None, :]
    out = row[:, None] * OST + channels[None, :]
    sq = SCALE * q
    tl.store(A + out, k * ep, valid)
    tl.store(B + out, k / ep, valid)
    tl.store(C + out, k * eg, valid)
    tl.store(D + out, sq * eg, valid)
    tl.store(E + out, sq * ep, valid)
    tl.store(F + fout, gc, valid)


@triton.jit
def _strip_recip(
    Q,
    K,
    G,
    CU,
    CI,
    OUT,
    F,
    OSIZE: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    GS: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    SCALE: tl.constexpr,
    BC: tl.constexpr,
    BT: tl.constexpr,
    OST: tl.constexpr,
):
    A = OUT
    B = OUT + OSIZE
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
    out = row[:, None] * OST + channels[None, :]
    fout = row[:, None] * (H * W) + channels[None, :]
    sq = SCALE * q
    tl.store(A + out, k * ep, valid)
    tl.store(B + out, k / ep, valid)
    tl.store(C + out, k * eg, valid)
    tl.store(D + out, sq * eg, valid)
    tl.store(E + out, sq * ep, valid)
    tl.store(F + fout, gc, valid)


def _baseline_entry(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    t, h, width = q.shape
    storage = torch.empty(
        (t, 5, h, width), device=q.device, dtype=torch.bfloat16
    )
    outputs = storage.unbind(1)
    osize = h * width
    ost = 5 * h * width
    gc = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    if t and h and width and num_chunks:
        if t >= 1024:
            _strip_recip.run(
                q,
                k,
                g_act,
                cu_seqlens,
                chunk_indices,
                storage,
                gc,
                osize,
                t,
                h,
                width,
                q.stride(),
                k.stride(),
                g_act.stride(),
                cu_seqlens.stride(0),
                chunk_indices.stride(),
                scale,
                128,
                16,
                ost,
                grid=(triton.cdiv(h * width, 128), 4, num_chunks),
                warmup=False,
                num_warps=1,
                enable_fp_fusion=False,
            )
        else:
            kernel = _recip if t >= 128 else _kernel
            nw = 1 if t >= 128 else 4
            kernel.run(
                q,
                k,
                g_act,
                cu_seqlens,
                chunk_indices,
                storage,
                gc,
                osize,
                ost,
                t,
                h,
                width,
                q.stride(),
                k.stride(),
                g_act.stride(),
                cu_seqlens.stride(0),
                chunk_indices.stride(),
                scale,
                32,
                num_warps=nw,
                enable_fp_fusion=False,
                grid=(num_chunks, triton.cdiv(h * width, 32)),
                warmup=False,
            )
    return (*outputs, gc)


@triton.jit
def _dot_strip(
    Q,
    K,
    G,
    CU,
    CI,
    OUT,
    F,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    GS: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    SCALE: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    OSIZE: tl.constexpr,
    OST: tl.constexpr,
):
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


def _interleaved_dot(
    q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks
):
    t, h, w = q.shape
    storage = torch.empty((t, 5, h, w), device=q.device, dtype=torch.bfloat16)
    outputs = storage.unbind(1)
    gate_prefix = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    if t and h and w and num_chunks:
        _dot_strip[(triton.cdiv(h * w, 128), 8, num_chunks)](
            q,
            k,
            g_act,
            cu_seqlens,
            chunk_indices,
            storage,
            gate_prefix,
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
            128,
            h * w,
            5 * h * w,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return (*outputs, gate_prefix)


def _medium_entry(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    t, h, w = q.shape
    if (
        g_act.dtype in (torch.bfloat16, torch.float16)
        and 128 <= t < 1024
        and h * w >= 2048
    ):
        return _interleaved_dot(
            q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks
        )
    return _baseline_entry(
        q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks
    )


@triton.jit
def _lastrow_dot(
    Q,
    K,
    G,
    CU,
    CI,
    OUT,
    F,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    GS: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    SCALE: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    OSIZE: tl.constexpr,
    OST: tl.constexpr,
):
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
    step = tl.arange(0, BT)
    triangle = tl.where(
        (step[:, None] == BT - 1)
        | (part * BT + step[:, None] >= token[None, :]),
        1.0,
        0.0,
    ).to(G.dtype.element_ty)
    acc = tl.dot(triangle, g, out_dtype=tl.float32)
    last = tl.sum(tl.where(step[:, None] == BT - 1, acc, 0.0), axis=0)[None, :]
    previous = tl.sum(tl.where(step[:, None] == BT - 2, acc, 0.0), axis=0)[
        None, :
    ]
    final_row = lo + part * BT + BT - 1
    final_g = tl.load(
        G + final_row * GS[0] + channel // W * GS[1] + channel % W * GS[2],
        (final_row < eos) & (final_row < T) & (channel < H * W),
        0,
    ).to(tl.float32)
    prefix = tl.where(
        step[:, None] == BT - 1, previous + final_g[None, :], acc
    )
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


def _lastrow_entry(
    q,
    k,
    g_act,
    scale,
    cu_seqlens,
    chunk_indices,
    num_chunks,
    bt=16,
    bc=128,
    nw=4,
):
    t, h, w = q.shape
    storage = torch.empty((t, 5, h, w), device=q.device, dtype=torch.bfloat16)
    outputs = storage.unbind(1)
    prefix = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    if t and h and w and num_chunks:
        _lastrow_dot[(triton.cdiv(h * w, bc), 64 // bt, num_chunks)](
            q,
            k,
            g_act,
            cu_seqlens,
            chunk_indices,
            storage,
            prefix,
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
            h * w,
            5 * h * w,
            num_warps=nw,
            enable_fp_fusion=False,
        )
    return (*outputs, prefix)


def _previous_entry(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    if q.shape[0] < 128 and g_act.dtype in (torch.bfloat16, torch.float16):
        return _lastrow_entry(
            q,
            k,
            g_act,
            scale,
            cu_seqlens,
            chunk_indices,
            num_chunks,
            16,
            64,
            4,
        )
    return _medium_entry(
        q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks
    )


def _small_entry(
    q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks, mode, bc, nw
):
    if mode == 16:
        return _lastrow_entry(
            q,
            k,
            g_act,
            scale,
            cu_seqlens,
            chunk_indices,
            num_chunks,
            16,
            bc,
            nw,
        )
    t, h, w = q.shape
    storage = torch.empty((t, 5, h, w), device=q.device, dtype=torch.bfloat16)
    prefix = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    if t and h and w and num_chunks:
        _dot_strip[(triton.cdiv(h * w, bc), 8, num_chunks)](
            q,
            k,
            g_act,
            cu_seqlens,
            chunk_indices,
            storage,
            prefix,
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
            h * w,
            5 * h * w,
            num_warps=nw,
            enable_fp_fusion=False,
        )
    return (*storage.unbind(1), prefix)


def _previous_transpose_entry(
    q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks
):
    if q.shape[0] < 128 and g_act.dtype in (torch.bfloat16, torch.float16):
        return _small_entry(
            q,
            k,
            g_act,
            scale,
            cu_seqlens,
            chunk_indices,
            num_chunks,
            8,
            64 if q.shape[1] * q.shape[2] <= 512 else 128,
            4,
        )
    return _previous_entry(
        q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks
    )


@triton.jit
def _transpose_dot(
    Q,
    K,
    G,
    CU,
    CI,
    OUT,
    F,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    GS: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    SCALE: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    OSIZE: tl.constexpr,
    OST: tl.constexpr,
):
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
    acc = tl.trans(
        tl.dot(tl.trans(g), tl.trans(triangle), out_dtype=tl.float32)
    )
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


def _transpose_entry(
    q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks, bt, bc, nw
):
    t, h, w = q.shape
    storage = torch.empty((t, 5, h, w), device=q.device, dtype=torch.bfloat16)
    prefix = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    if t and h and w and num_chunks:
        _transpose_dot[(triton.cdiv(h * w, bc), 64 // bt, num_chunks)](
            q,
            k,
            g_act,
            cu_seqlens,
            chunk_indices,
            storage,
            prefix,
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
            h * w,
            5 * h * w,
            num_warps=nw,
            enable_fp_fusion=False,
        )
    return (*storage.unbind(1), prefix)


def kda_prologue(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    t, h, w = q.shape
    if g_act.dtype in (torch.bfloat16, torch.float16) and t >= 128:
        return _transpose_entry(
            q,
            k,
            g_act,
            scale,
            cu_seqlens,
            chunk_indices,
            num_chunks,
            16,
            128,
            4,
        )
    return _previous_transpose_entry(
        q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks
    )


__all__ = ["kda_prologue"]
