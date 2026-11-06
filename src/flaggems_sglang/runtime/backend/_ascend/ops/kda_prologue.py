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


@triton.jit(do_not_specialize=["Q", "K", "G", "CU", "CI", "OUT", "F"])
def _kernel(Q, K, G, CU, CI, OUT, F, META: tl.constexpr):
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
    A = OUT
    B = OUT + T * H * W
    C = OUT + 2 * T * H * W
    D = OUT + 3 * T * H * W
    E = OUT + 4 * T * H * W
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


@triton.jit(do_not_specialize=["G", "CU", "CI", "TEMP"])
def _direct_dot(G, CU, CI, TEMP, META: tl.constexpr):
    T: tl.constexpr = META[0]
    HW: tl.constexpr = META[1]
    CS: tl.constexpr = META[2]
    CIS: tl.constexpr = META[3]
    BC: tl.constexpr = META[4]
    chunk = tl.program_id(0)
    channel = tl.program_id(1) * BC + tl.arange(0, BC)
    token = tl.arange(0, 64)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    row = bos + cid * 64 + token
    valid = (row[:, None] < eos) & (row[:, None] < T) & (channel[None, :] < HW)
    g = tl.load(G + row[:, None] * HW + channel[None, :], valid, 0)
    lower = (token[:, None] >= token[None, :]).to(G.dtype.element_ty)
    gc = tl.dot(lower, g, out_dtype=tl.float32)
    out = row[:, None] * HW + channel[None, :]
    tl.store(TEMP + out, gc, valid)


@triton.jit(do_not_specialize=["Q", "K", "CU", "CI", "TEMP", "OUT"])
def _flat_finish(Q, K, CU, CI, TEMP, OUT, META: tl.constexpr):
    T: tl.constexpr = META[0]
    HW: tl.constexpr = META[1]
    CS: tl.constexpr = META[2]
    CIS: tl.constexpr = META[3]
    SCALE: tl.constexpr = META[4]
    BLOCK: tl.constexpr = META[5]
    A = OUT
    B = OUT + T * HW
    C = OUT + 2 * T * HW
    D = OUT + 3 * T * HW
    E = OUT + 4 * T * HW
    chunk = tl.program_id(0)
    ix = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    lo = bos + cid * 64
    limit = (tl.minimum(eos, T) - lo) * HW
    out = lo * HW + ix
    gc = tl.load(TEMP + out, ix < limit, 0)
    last_row = tl.minimum(lo + 64, tl.minimum(eos, T)) - 1
    if BLOCK >= HW:
        channel = tl.arange(0, HW)
        totals = tl.load(TEMP + last_row * HW + channel, last_row >= lo, 0)
        last = tl.reshape(
            tl.broadcast_to(totals[None, :], (BLOCK // HW, HW)), (BLOCK,)
        )
    else:
        channel = tl.program_id(1) % (HW // BLOCK) * BLOCK + tl.arange(
            0, BLOCK
        )
        last = tl.load(TEMP + last_row * HW + channel, last_row >= lo, 0)
    q = tl.load(Q + out, ix < limit, 0).to(tl.float32)
    k = tl.load(K + out, ix < limit, 0).to(tl.float32)
    diff = gc - last
    ep = tl.exp(diff)
    em = tl.exp(-diff)
    eg = tl.exp(gc)
    sq = SCALE * q
    tl.store(A + out, k * ep, ix < limit)
    tl.store(B + out, k * em, ix < limit)
    tl.store(C + out, k * eg, ix < limit)
    tl.store(D + out, sq * eg, ix < limit)
    tl.store(E + out, sq * ep, ix < limit)


@triton.jit(do_not_specialize=["Q", "K", "CU", "CI", "TEMP", "OUT"])
def _scheduled_out(Q, K, CU, CI, TEMP, OUT, META: tl.constexpr):
    T: tl.constexpr = META[0]
    HW: tl.constexpr = META[1]
    CS: tl.constexpr = META[2]
    CIS: tl.constexpr = META[3]
    SCALE: tl.constexpr = META[4]
    BLOCK: tl.constexpr = META[5]
    A = OUT
    B = OUT + T * HW
    C = OUT + 2 * T * HW
    D = OUT + 3 * T * HW
    E = OUT + 4 * T * HW
    chunk = tl.program_id(0)
    ix = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    lo = bos + cid * 64
    limit = (tl.minimum(eos, T) - lo) * HW
    out = lo * HW + ix
    gc = tl.load(TEMP + out, ix < limit, 0)
    last_row = tl.minimum(lo + 64, tl.minimum(eos, T)) - 1
    if BLOCK >= HW:
        channel = tl.arange(0, HW)
        totals = tl.load(TEMP + last_row * HW + channel, last_row >= lo, 0)
        last = tl.reshape(
            tl.broadcast_to(totals[None, :], (BLOCK // HW, HW)), (BLOCK,)
        )
    else:
        channel = tl.program_id(1) % (HW // BLOCK) * BLOCK + tl.arange(
            0, BLOCK
        )
        last = tl.load(TEMP + last_row * HW + channel, last_row >= lo, 0)
    diff = gc - last
    ep = tl.exp(diff)
    k = tl.load(K + out, ix < limit, 0).to(tl.float32)
    tl.store(A + out, k * ep, ix < limit)
    tl.store(B + out, k / ep, ix < limit)
    eg = tl.exp(gc)
    tl.store(C + out, k * eg, ix < limit)
    q = tl.load(Q + out, ix < limit, 0).to(tl.float32)
    sq = SCALE * q
    tl.store(D + out, sq * eg, ix < limit)
    tl.store(E + out, sq * ep, ix < limit)


def _baseline(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    shape = q.shape
    t, h, w = shape
    hw = h * w
    storage = q.new_empty((5, t, h, w), dtype=torch.bfloat16)
    gc = q.new_empty(shape, dtype=torch.float32)
    bc = 32 if t < 128 and hw <= 512 else 64
    if t and h and w and num_chunks:
        cs = cu_seqlens.stride(0)
        cis = chunk_indices.stride()
        fast = (
            t > 128
            and hw >= 64
            and (not hw & hw - 1)
            and q.is_contiguous()
            and k.is_contiguous()
            and g_act.is_contiguous()
            and (g_act.dtype in (torch.bfloat16, torch.float16))
        )
        if fast:
            _direct_dot.run(
                g_act,
                cu_seqlens,
                chunk_indices,
                gc,
                (t, hw, cs, cis, 512),
                grid=(num_chunks, (hw + 511) // 512),
                warmup=False,
                num_warps=4,
                num_stages=1,
                multibuffer=False,
                enable_fp_fusion=False,
            )
            if t >= 1024:
                block = min(8192, 64 * hw)
                _scheduled_out.run(
                    q,
                    k,
                    cu_seqlens,
                    chunk_indices,
                    gc,
                    storage,
                    (t, hw, cs, cis, scale, block),
                    grid=(num_chunks, (64 * hw + block - 1) // block),
                    warmup=False,
                    num_warps=4,
                    multibuffer=False,
                    enable_fp_fusion=False,
                )
            else:
                block = min(4096, 64 * hw)
                _flat_finish.run(
                    q,
                    k,
                    cu_seqlens,
                    chunk_indices,
                    gc,
                    storage,
                    (t, hw, cs, cis, scale, block),
                    grid=(num_chunks, (64 * hw + block - 1) // block),
                    warmup=False,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        else:
            _kernel.run(
                q,
                k,
                g_act,
                cu_seqlens,
                chunk_indices,
                storage,
                gc,
                (
                    t,
                    h,
                    w,
                    q.stride(),
                    k.stride(),
                    g_act.stride(),
                    cs,
                    cis,
                    scale,
                    bc,
                ),
                grid=(num_chunks, (hw + bc - 1) // bc),
                warmup=False,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return (*storage.unbind(0), gc)


@triton.jit(do_not_specialize=["Q", "K", "G", "CU", "CI", "OUT", "F"])
def _chunk_roundtrip(Q, K, G, CU, CI, OUT, F, META: tl.constexpr):
    T: tl.constexpr = META[0]
    HW: tl.constexpr = META[1]
    CS: tl.constexpr = META[2]
    CIS: tl.constexpr = META[3]
    SCALE: tl.constexpr = META[4]
    BC: tl.constexpr = META[5]
    BLOCK: tl.constexpr = META[6]
    chunk = tl.program_id(0)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    lo = tl.load(CU + seq * CS) + cid * 64
    eos = tl.load(CU + (seq + 1) * CS)
    token = tl.arange(0, 64)
    row = lo + token
    triangle = (token[:, None] >= token[None, :]).to(G.dtype.element_ty)
    for tile in range((HW + BC - 1) // BC):
        channel = tile * BC + tl.arange(0, BC)
        valid = (
            (row[:, None] < eos) & (row[:, None] < T) & (channel[None, :] < HW)
        )
        gate = tl.load(G + row[:, None] * HW + channel[None, :], valid, 0)
        cumulative = tl.dot(triangle, gate, out_dtype=tl.float32)
        tl.store(F + row[:, None] * HW + channel[None, :], cumulative, valid)
    tl.debug_barrier()
    limit = (tl.minimum(eos, T) - lo) * HW
    lastrow = tl.minimum(lo + 64, tl.minimum(eos, T)) - 1
    if BLOCK >= HW:
        cols = tl.arange(0, HW)
        totals = tl.load(F + lastrow * HW + cols, lastrow >= lo, 0)
        last = tl.reshape(
            tl.broadcast_to(totals[None, :], (BLOCK // HW, HW)), (BLOCK,)
        )
    for tile in range((64 * HW + BLOCK - 1) // BLOCK):
        ix = tile * BLOCK + tl.arange(0, BLOCK)
        out = lo * HW + ix
        valid = (ix < limit) & (ix < 64 * HW)
        prefix = tl.load(F + out, valid, 0)
        if BLOCK >= HW:
            pass
        else:
            cols = tile % (HW // BLOCK) * BLOCK + tl.arange(0, BLOCK)
            last = tl.load(F + lastrow * HW + cols, lastrow >= lo, 0)
        diff = prefix - last
        ep = tl.exp(diff)
        key = tl.load(K + out, valid, 0).to(tl.float32)
        tl.store(OUT + out, key * ep, valid)
        tl.store(OUT + T * HW + out, key * tl.exp(-diff), valid)
        eg = tl.exp(prefix)
        tl.store(OUT + 2 * T * HW + out, key * eg, valid)
        query = tl.load(Q + out, valid, 0).to(tl.float32) * SCALE
        tl.store(OUT + 3 * T * HW + out, query * eg, valid)
        tl.store(OUT + 4 * T * HW + out, query * ep, valid)


@triton.jit
def _partition_roundtrip(Q, K, G, CU, CI, OUT, F, META: tl.constexpr):
    T: tl.constexpr = META[0]
    HW: tl.constexpr = META[1]
    CS: tl.constexpr = META[2]
    CIS: tl.constexpr = META[3]
    SCALE: tl.constexpr = META[4]
    BC: tl.constexpr = META[5]
    GROUP: tl.constexpr = META[6]
    BR: tl.constexpr = META[7]
    chunk = tl.program_id(0)
    offset = tl.program_id(1) * GROUP
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    lo = tl.load(CU + seq * CS) + cid * 64
    eos = tl.minimum(tl.load(CU + (seq + 1) * CS), T)
    token = tl.arange(0, 64)
    row = lo + token
    triangle = (token[:, None] >= token[None, :]).to(G.dtype.element_ty)
    for tile in range((GROUP + BC - 1) // BC):
        channel = offset + tile * BC + tl.arange(0, BC)
        valid = (
            (row[:, None] < eos)
            & (channel[None, :] < HW)
            & (channel[None, :] < offset + GROUP)
        )
        gate = tl.load(G + row[:, None] * HW + channel[None, :], valid, 0)
        cumulative = tl.dot(triangle, gate, out_dtype=tl.float32)
        tl.store(F + row[:, None] * HW + channel[None, :], cumulative, valid)
    tl.debug_barrier()
    lastrow = tl.minimum(lo + 64, eos) - 1
    columns = offset + tl.arange(0, GROUP)
    last = tl.load(
        F + lastrow * HW + columns, (lastrow >= lo) & (columns < HW), 0
    )
    for tile in range(64 // BR):
        rows = lo + tile * BR + tl.arange(0, BR)
        out = rows[:, None] * HW + columns[None, :]
        valid = (rows[:, None] < eos) & (columns[None, :] < HW)
        prefix = tl.load(F + out, valid, 0)
        diff = prefix - last[None, :]
        ep = tl.exp(diff)
        key = tl.load(K + out, valid, 0).to(tl.float32)
        tl.store(OUT + out, key * ep, valid)
        tl.store(OUT + T * HW + out, key * tl.exp(-diff), valid)
        eg = tl.exp(prefix)
        tl.store(OUT + 2 * T * HW + out, key * eg, valid)
        query = tl.load(Q + out, valid, 0).to(tl.float32) * SCALE
        tl.store(OUT + 3 * T * HW + out, query * eg, valid)
        tl.store(OUT + 4 * T * HW + out, query * ep, valid)


def kda_prologue(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    t, h, w = q.shape
    hw = h * w
    if (
        (0 < t < 4096 and (t > 128 or hw > 512))
        and hw >= 64
        and (not hw & hw - 1)
        and q.is_contiguous()
        and k.is_contiguous()
        and g_act.is_contiguous()
        and (g_act.dtype in (torch.bfloat16, torch.float16))
    ):
        storage = q.new_empty((5, t, h, w), dtype=torch.bfloat16)
        prefix = q.new_empty(q.shape, dtype=torch.float32)
        if num_chunks:
            _chunk_roundtrip.run(
                q,
                k,
                g_act,
                cu_seqlens,
                chunk_indices,
                storage,
                prefix,
                (
                    t,
                    hw,
                    cu_seqlens.stride(0),
                    chunk_indices.stride(),
                    scale,
                    512,
                    8192,
                ),
                grid=(num_chunks,),
                warmup=False,
                num_warps=4,
                num_stages=1,
                multibuffer=False,
                enable_fp_fusion=False,
            )
        return (*storage.unbind(0), prefix)
    if (
        t >= 4096
        and hw >= 64
        and (not hw & hw - 1)
        and q.is_contiguous()
        and k.is_contiguous()
        and g_act.is_contiguous()
        and (g_act.dtype in (torch.bfloat16, torch.float16))
    ):
        storage = q.new_empty((5, t, h, w), dtype=torch.bfloat16)
        prefix = q.new_empty(q.shape, dtype=torch.float32)
        if num_chunks:
            _partition_roundtrip.run(
                q,
                k,
                g_act,
                cu_seqlens,
                chunk_indices,
                storage,
                prefix,
                (
                    t,
                    hw,
                    cu_seqlens.stride(0),
                    chunk_indices.stride(),
                    scale,
                    512,
                    min(hw, 1024),
                    16,
                ),
                grid=(num_chunks, (hw + 1024 - 1) // 1024),
                warmup=False,
                num_warps=4,
                num_stages=1,
                multibuffer=False,
                enable_fp_fusion=False,
            )
        return (*storage.unbind(0), prefix)
    return _baseline(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks)


__all__ = ["kda_prologue"]
