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
def _pack(
    G,
    CU,
    CI,
    PACK,
    TRI,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GS: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    BW: tl.constexpr,
):
    if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):
        tri_ix = tl.arange(0, 4096)
        tl.store(
            TRI + tri_ix,
            (tri_ix // 64 >= tri_ix % 64).to(TRI.dtype.element_ty),
        )
    chunk = tl.program_id(0)
    ix = tl.program_id(1) * BW + tl.arange(0, BW)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    lo = bos + cid * 64
    limit = (tl.minimum(eos, T) - lo) * H * W
    if limit >= 64 * H * W:
        g = tl.load(G + lo * H * W + ix)
        tl.store(PACK + chunk * 64 * H * W + ix, g)
    else:
        channels = tl.arange(0, H * W)
        for token in range(BW // (H * W)):
            r = tl.program_id(1) * (BW // (H * W)) + token
            if r * H * W < limit:
                g = tl.load(G + (lo + r) * H * W + channels)
                tl.store(PACK + (chunk * 64 + r) * H * W + channels, g)
            else:
                tl.store(
                    PACK + (chunk * 64 + r) * H * W + channels,
                    tl.full((H * W,), 0, PACK.dtype.element_ty),
                )


@triton.jit
def _dot(PACK, TRI, TEMP, HW: tl.constexpr, BC: tl.constexpr):
    chunk = tl.program_id(0)
    channel = tl.program_id(1) * BC + tl.arange(0, BC)
    token = tl.arange(0, 64)
    a = tl.load(TRI + token[:, None] * 64 + token[None, :])
    ptr = (chunk * 64 + token[:, None]) * HW + channel[None, :]
    g = tl.load(PACK + ptr, channel[None, :] < HW, 0)
    gc = tl.dot(a, g, out_dtype=tl.float32)
    tl.store(TEMP + ptr, gc, channel[None, :] < HW)


@triton.jit
def _finish(
    Q,
    K,
    CU,
    CI,
    TEMP,
    OUT,
    F,
    OSIZE: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    QS: tl.constexpr,
    KS: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    SCALE: tl.constexpr,
    BW: tl.constexpr,
    FAST: tl.constexpr,
):
    A = OUT
    B = OUT + OSIZE
    C = OUT + 2 * OSIZE
    D = OUT + 3 * OSIZE
    E = OUT + 4 * OSIZE
    chunk = tl.program_id(0)
    token = tl.program_id(1)
    c = tl.arange(0, BW)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    row = bos + cid * 64 + token
    safe_row = tl.minimum(row, T - 1)
    col = c if FAST else tl.minimum(c, H * W - 1)
    if FAST:
        gc = tl.load(TEMP + (chunk * 64 + token) * H * W + c)
        last = tl.load(TEMP + (chunk * 64 + 63) * H * W + c)
    else:
        gc = tl.load(TEMP + (chunk * H * W + col) * 64 + token)
        last = tl.load(TEMP + (chunk * H * W + col) * 64 + 63)
    q = tl.load(Q + safe_row * QS[0] + col // W * QS[1] + col % W * QS[2]).to(
        tl.float32
    )
    k = tl.load(K + safe_row * KS[0] + col // W * KS[1] + col % W * KS[2]).to(
        tl.float32
    )
    diff = gc - last
    ep = tl.exp(diff)
    em = tl.exp(-diff)
    eg = tl.exp(gc)
    sq = SCALE * q
    out_row = tl.where((row < eos) & (row < T), row, T + chunk * 64 + token)
    out = out_row * BW + c
    tl.store(A + out, k * ep)
    tl.store(B + out, k * em)
    tl.store(C + out, k * eg)
    tl.store(D + out, sq * eg)
    tl.store(E + out, sq * ep)
    tl.store(F + out, gc)


@triton.jit
def _scalar_prefix(
    G,
    CU,
    CI,
    TEMP,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GS: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
):
    chunk = tl.program_id(0)
    channel = tl.program_id(1)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    token = tl.arange(0, 64)
    row = bos + cid * 64 + token
    safe = tl.minimum(row, T - 1)
    g = tl.load(
        G + safe * GS[0] + channel // W * GS[1] + channel % W * GS[2]
    ).to(tl.float32)
    g = tl.where((row < eos) & (row < T), g, 0.0)
    gc = tl.cumsum(g, 0)
    tl.store(TEMP + (chunk * H * W + channel) * 64 + token, gc)


@triton.jit
def _flat_finish(
    Q,
    K,
    CU,
    CI,
    TEMP,
    OUT,
    F,
    OSIZE: tl.constexpr,
    T: tl.constexpr,
    HW: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    A = OUT
    B = OUT + OSIZE
    C = OUT + 2 * OSIZE
    D = OUT + 3 * OSIZE
    E = OUT + 4 * OSIZE
    chunk = tl.program_id(0)
    ix = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    lo = bos + cid * 64
    limit = (tl.minimum(eos, T) - lo) * HW
    if (tl.program_id(1) + 1) * BLOCK <= limit:
        out = lo * HW + ix
        gc = tl.load(TEMP + chunk * 64 * HW + ix)
        last = tl.load(TEMP + (chunk * 64 + 63) * HW + ix % HW)
        q = tl.load(Q + out).to(tl.float32)
        k = tl.load(K + out).to(tl.float32)
        diff = gc - last
        ep = tl.exp(diff)
        em = tl.exp(last - gc)
        eg = tl.exp(gc)
        sq = SCALE * q
        tl.store(A + out, k * ep)
        tl.store(B + out, k * em)
        tl.store(C + out, k * eg)
        tl.store(D + out, sq * eg)
        tl.store(E + out, sq * ep)
        tl.store(F + out, gc)
    else:
        out = lo * HW + ix
        gc = tl.load(TEMP + chunk * 64 * HW + ix)
        last = tl.load(TEMP + (chunk * 64 + 63) * HW + ix % HW)
        q = tl.load(Q + out, ix < limit, 0).to(tl.float32)
        k = tl.load(K + out, ix < limit, 0).to(tl.float32)
        diff = gc - last
        ep = tl.exp(diff)
        em = tl.exp(last - gc)
        eg = tl.exp(gc)
        sq = SCALE * q
        tl.store(A + out, k * ep, ix < limit)
        tl.store(B + out, k * em, ix < limit)
        tl.store(C + out, k * eg, ix < limit)
        tl.store(D + out, sq * eg, ix < limit)
        tl.store(E + out, sq * ep, ix < limit)
        tl.store(F + out, gc, ix < limit)


def _fallback_entry(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    t, h, w = q.shape
    hw = h * w
    bw = triton.next_power_of_2(hw)
    fast = (
        64 <= hw <= 16384
        and hw == bw
        and q.is_contiguous()
        and k.is_contiguous()
        and g_act.is_contiguous()
        and g_act.dtype == torch.bfloat16
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
    )
    shape = (t, h, w) if fast else (t + num_chunks * 64, bw)
    storage = torch.empty((5, *shape), device=q.device, dtype=torch.bfloat16)
    buffers = storage.unbind(0)
    gc = torch.empty(shape, device=q.device, dtype=torch.float32)
    if t and h and w and num_chunks:
        temp = torch.empty(
            (num_chunks, 64, hw), device=q.device, dtype=torch.float32
        )
        if fast:
            pack = torch.empty(
                (num_chunks, 64, hw), device=q.device, dtype=g_act.dtype
            )
            tri = torch.empty((64, 64), device=q.device, dtype=g_act.dtype)
            block = min(16384, 64 * hw)
            _pack[(num_chunks, triton.cdiv(64 * hw, block))](
                g_act,
                cu_seqlens,
                chunk_indices,
                pack,
                tri,
                t,
                h,
                w,
                g_act.stride(),
                cu_seqlens.stride(0),
                chunk_indices.stride(),
                block,
                num_warps=4,
                isCloseMemoryAsync=False,
                enable_fp_fusion=False,
            )
            _dot[(num_chunks, triton.cdiv(hw, 64))](
                pack,
                tri,
                temp,
                hw,
                64,
                num_warps=4,
                num_stages=1,
                isCloseMemoryAsync=False,
                enable_fp_fusion=False,
            )
        else:
            _scalar_prefix[(num_chunks, hw)](
                g_act,
                cu_seqlens,
                chunk_indices,
                temp,
                t,
                h,
                w,
                g_act.stride(),
                cu_seqlens.stride(0),
                chunk_indices.stride(),
                num_warps=4,
                num_stages=1,
                isCloseMemoryAsync=False,
                enable_fp_fusion=False,
            )
        if fast and q.dtype != torch.float16 and k.dtype != torch.float16:
            block = min(16384, 64 * hw)
            _flat_finish[(num_chunks, triton.cdiv(64 * hw, block))](
                q,
                k,
                cu_seqlens,
                chunk_indices,
                temp,
                storage,
                gc,
                storage.stride(0),
                t,
                hw,
                cu_seqlens.stride(0),
                chunk_indices.stride(),
                scale,
                block,
                num_warps=4,
                isCloseMemoryAsync=False,
                isCloseDtypeConvert=q.dtype == torch.float16
                or k.dtype == torch.float16,
                enable_fp_fusion=False,
            )
        else:
            _finish[(num_chunks, 64)](
                q,
                k,
                cu_seqlens,
                chunk_indices,
                temp,
                storage,
                gc,
                storage.stride(0),
                t,
                h,
                w,
                q.stride(),
                k.stride(),
                cu_seqlens.stride(0),
                chunk_indices.stride(),
                scale,
                bw,
                fast,
                num_warps=4,
                isCloseMemoryAsync=False,
                isCloseDtypeConvert=q.dtype == torch.float16
                or k.dtype == torch.float16,
                enable_fp_fusion=False,
            )
    if fast:
        return (*buffers, gc)
    return tuple(x[:t, :hw].view(t, h, w) for x in (*buffers, gc))


@triton.jit
def _pack_gate(
    G,
    CU,
    CI,
    PACK,
    TRI,
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    GS: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    BW: tl.constexpr,
):
    if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):
        tri_ix = tl.arange(0, 4096)
        tl.store(
            TRI + tri_ix,
            (tri_ix // 64 >= tri_ix % 64).to(TRI.dtype.element_ty),
        )
        tl.store(
            TRI + 4096 + tri_ix, tl.full((4096,), 1.0, TRI.dtype.element_ty)
        )
    chunk = tl.program_id(0)
    ix = tl.program_id(1) * BW + tl.arange(0, BW)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    bos = tl.load(CU + seq * CS)
    eos = tl.load(CU + (seq + 1) * CS)
    lo = bos + cid * 64
    limit = (tl.minimum(eos, T) - lo) * H * W
    if limit >= 64 * H * W:
        g = tl.load(G + lo * H * W + ix)
        tl.store(PACK + chunk * 64 * H * W + ix, g)
    else:
        channels = tl.arange(0, H * W)
        for token in range(BW // (H * W)):
            r = tl.program_id(1) * (BW // (H * W)) + token
            if r * H * W < limit:
                g = tl.load(G + (lo + r) * H * W + channels)
                tl.store(PACK + (chunk * 64 + r) * H * W + channels, g)
            else:
                tl.store(
                    PACK + (chunk * 64 + r) * H * W + channels,
                    tl.full((H * W,), 0, PACK.dtype.element_ty),
                )


@triton.jit
def _direct_all(
    G,
    QP,
    KP,
    CU,
    CI,
    OUT,
    F,
    T: tl.constexpr,
    HW: tl.constexpr,
    CS: tl.constexpr,
    CIS: tl.constexpr,
    SCALE: tl.constexpr,
    BC: tl.constexpr,
):
    chunk = tl.program_id(0)
    channel = tl.program_id(1) * BC + tl.arange(0, BC)
    token = tl.arange(0, 64)
    seq = tl.load(CI + chunk * CIS[0])
    cid = tl.load(CI + chunk * CIS[0] + CIS[1])
    lo = tl.load(CU + seq * CS) + cid * 64
    eos = tl.load(CU + (seq + 1) * CS)
    row = lo + token
    qkptr = row[:, None] * HW + channel[None, :]
    qkvalid = (row[:, None] < eos) & (row[:, None] < T)
    g = tl.load(G + qkptr, qkvalid, 0)
    a = tl.where(token[:, None] >= token[None, :], 1.0, 0.0).to(
        G.dtype.element_ty
    )
    ones = tl.full((64, 64), 1.0, G.dtype.element_ty)
    gc = tl.dot(a, g, out_dtype=tl.float32)
    last = tl.dot(ones, g, out_dtype=tl.float32)
    q = tl.load(QP + qkptr, qkvalid, 0).to(tl.float32)
    k = tl.load(KP + qkptr, qkvalid, 0).to(tl.float32)
    ep = tl.exp(gc - last - 8.0) * 2980.9579870417283
    em = tl.exp(last - gc)
    eg = tl.exp(gc)
    sq = SCALE * q
    out = row[:, None] * HW + channel[None, :]
    valid = (row[:, None] < eos) & (row[:, None] < T)
    tl.store(OUT + out, k * ep, valid)
    tl.store(OUT + T * HW + out, k * em, valid)
    tl.store(OUT + 2 * T * HW + out, k * eg, valid)
    tl.store(OUT + 3 * T * HW + out, sq * eg, valid)
    tl.store(OUT + 4 * T * HW + out, sq * ep, valid)
    tl.store(F + out, gc, valid)


def kda_prologue(q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks):
    t, h, w = q.shape
    hw = h * w
    fast = (
        64 <= hw <= 16384
        and hw == triton.next_power_of_2(hw)
        and q.is_contiguous()
        and k.is_contiguous()
        and g_act.is_contiguous()
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and g_act.dtype == torch.bfloat16
    )
    if not fast:
        return _fallback_entry(
            q, k, g_act, scale, cu_seqlens, chunk_indices, num_chunks
        )
    storage = torch.empty((5, t, h, w), device=q.device, dtype=torch.bfloat16)
    prefix = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    if t and h and w and num_chunks:
        bc = min(256, hw)
        _direct_all[(num_chunks, triton.cdiv(hw, bc))](
            g_act,
            q,
            k,
            cu_seqlens,
            chunk_indices,
            storage,
            prefix,
            t,
            hw,
            cu_seqlens.stride(0),
            chunk_indices.stride(),
            scale,
            bc,
            num_warps=4,
            num_stages=1,
            isCloseMemoryAsync=False,
            exp_range="-112:80.72",
            enable_fp_fusion=False,
        )
    return (*storage.unbind(0), prefix)


__all__ = ["kda_prologue"]
