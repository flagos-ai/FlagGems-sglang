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

"""DeepEP low-latency dispatch scatter -- Enflame GCU specialization.

Single-launch deterministic rewrite for the hot shapes (E <= 16, contiguous
recv_topk, t * k <= 32768): every program owns a contiguous block of tokens,
computes each slot's destination as the canonical row-major rank with a
serial walk over named per-expert counters (no atomics, no 2D tiles, no
loop-carried vectors), and writes the payload rows inline. The e-part
(starts, m_indices, expert_start_loc = start + live) runs in the same launch,
strided over experts. Output is bitwise identical to the reference order.

The E <= 8 arm (the production hot path) is latency-tuned: the per-token
expert ids load as one back-to-back hoisted group (pipelines instead of
eight store-interleaved round trips), the eight named-counter seeds come
from four packed 16-bit-field reductions, and fully-covering row/scale
masks fold away when the widths are exact powers of two.

Out-of-gate shapes fall back to the upstream two-kernel implementation.
"""

import triton
import triton.language as tl

# Programs for the serial-walk kernels. 24 wins at small token counts (less
# redundant prefix scanning), 48 from ~128 tokens up (saturates the SPs).
_SMALL_T_TOKENS = 128
_SMALL_PROGRAMS = 24
_LARGE_PROGRAMS = 48

# Serial-walk gate: per-walk scan volume stays linear and the (BE, CHUNK)
# count matrix stays inside the DSM budget (BE <= 32 x 2048 x 4B = 256KB).
_MAX_SERIAL_ENTRIES = 32768
_MAX_SERIAL_EXPERTS = 32


@triton.jit
def _serial_scatter8(
    C,
    V,
    S,
    M,
    X,
    XS,
    K,
    Y,
    YS,
    O,
    T,
    NK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CS: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    MS: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    Q: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    Y0: tl.constexpr,
    Y1: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HB: tl.constexpr,
    BS: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    CHUNK: tl.constexpr,
    FLAT_IDS: tl.constexpr,
    ROW_FULL: tl.constexpr,
    SCALE_FULL: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    es = tl.arange(0, BE)
    cols = tl.arange(0, HB)
    cmask = cols < H * Q
    scales = tl.arange(0, BS)
    smask = scales < D
    counts = tl.load(C + es * CS, es < E, 0)
    # ---- e-part: strided over experts ----
    for ei in range(pid, E, nprog):
        start_e = tl.sum(tl.where(es < ei, counts, 0), 0)
        count_e = tl.load(C + ei * CS)
        valid_e = tl.load(V + ei * VS)
        live_e = 0
        for c0 in range(0, T * NK, CHUNK):
            cp = c0 + tl.arange(0, CHUNK)
            cm2 = cp < T * NK
            if FLAT_IDS:
                idc = tl.load(K + cp * K1, cm2, -1)
            else:
                idc = tl.load(K + (cp // NK) * K0 + (cp % NK) * K1, cm2, -1)
            live_e += tl.sum((idc == ei).to(tl.int32), 0)
        tl.store(S + ei * SS, start_e + live_e)
        m = tl.arange(0, 128)
        for base in range(0, count_e, 128):
            r = base + m
            tl.store(
                M + (start_e + r) * MS, tl.where(r < valid_e, ei, -1), r < count_e
            )
    # ---- block prefix scan: live entries before this program's block ----
    tpp = (T + nprog - 1) // nprog
    tb0 = pid * tpp
    tb1 = tl.minimum(tb0 + tpp, T)
    scan_end = tl.where(tb0 < T, tb0 * NK, 0)
    starts_vec = tl.sum(tl.where(es[None, :] < es[:, None], counts[None, :], 0), 1)
    scan_cnt = tl.zeros([BE], dtype=tl.int32)
    for c0 in range(0, scan_end, CHUNK):
        cp = c0 + tl.arange(0, CHUNK)
        cm2 = cp < scan_end
        if FLAT_IDS:
            idc = tl.load(K + cp * K1, cm2, -1)
        else:
            idc = tl.load(K + (cp // NK) * K0 + (cp % NK) * K1, cm2, -1)
        scan_cnt += tl.sum((idc[None, :] == es[:, None]).to(tl.int32), 1)
    nxt = starts_vec + scan_cnt
    # Packed extraction: 4 int32 reductions instead of 8 (two 16-bit fields
    # each). Row ranks stay below 65536 inside the serial gate (t*k <= 32768
    # plus per-expert 128-row padding), so the fields never collide; the
    # two-term packed add wraps in int32 bits, which the shift/mask unpack
    # below recovers exactly.
    sel = es // 2
    sh = (es % 2) * 16
    pk0 = tl.sum(tl.where(sel == 0, nxt << sh, 0))
    pk1 = tl.sum(tl.where(sel == 1, nxt << sh, 0))
    pk2 = tl.sum(tl.where(sel == 2, nxt << sh, 0))
    pk3 = tl.sum(tl.where(sel == 3, nxt << sh, 0))
    d0 = pk0 & 0xFFFF
    d1 = (pk0 >> 16) & 0xFFFF
    d2 = pk1 & 0xFFFF
    d3 = (pk1 >> 16) & 0xFFFF
    d4 = pk2 & 0xFFFF
    d5 = (pk2 >> 16) & 0xFFFF
    d6 = pk3 & 0xFFFF
    d7 = (pk3 >> 16) & 0xFFFF
    # ---- serial token walk: canonical row-major ranks ----
    for ti in range(tb0, tb1):
        token = ti.to(tl.int64)
        if ROW_FULL:
            row = tl.load(X + token * X0 + (cols // Q) * X1 + cols % Q)
        else:
            row = tl.load(X + token * X0 + (cols // Q) * X1 + cols % Q, cmask, 0)
        if HAS_SCALE:
            if SCALE_FULL:
                srow = tl.load(XS + token * XS0 + scales * XS1)
            else:
                srow = tl.load(XS + token * XS0 + scales * XS1, smask, 0)
        # Hoist the token's expert ids: one back-to-back load group pipelines
        # instead of eight latency-exposed loads interleaved with stores.
        if NK <= 8:
            ke = ()
            for jj in tl.static_range(0, NK):
                ke = ke + (tl.load(K + token * K0 + jj * K1),)
        for j in tl.static_range(0, NK):
            if NK <= 8:
                e = ke[j]
            else:
                e = tl.load(K + token * K0 + j * K1)
            if e >= 0:
                dest = tl.where(
                    e == 0,
                    d0,
                    tl.where(
                        e == 1,
                        d1,
                        tl.where(
                            e == 2,
                            d2,
                            tl.where(
                                e == 3,
                                d3,
                                tl.where(
                                    e == 4,
                                    d4,
                                    tl.where(e == 5, d5, tl.where(e == 6, d6, d7)),
                                ),
                            ),
                        ),
                    ),
                )
                d64 = dest.to(tl.int64)
                if ROW_FULL:
                    tl.store(Y + d64 * Y0 + (cols // Q) * Y1 + cols % Q, row)
                else:
                    tl.store(Y + d64 * Y0 + (cols // Q) * Y1 + cols % Q, row, cmask)
                if HAS_SCALE:
                    if SCALE_FULL:
                        tl.store(YS + d64 * YS0 + scales * YS1, srow)
                    else:
                        tl.store(YS + d64 * YS0 + scales * YS1, srow, smask)
                tl.store(O + token * O0 + j * O1, dest)
            d0 = d0 + (e == 0).to(tl.int32)
            d1 = d1 + (e == 1).to(tl.int32)
            d2 = d2 + (e == 2).to(tl.int32)
            d3 = d3 + (e == 3).to(tl.int32)
            d4 = d4 + (e == 4).to(tl.int32)
            d5 = d5 + (e == 5).to(tl.int32)
            d6 = d6 + (e == 6).to(tl.int32)
            d7 = d7 + (e == 7).to(tl.int32)


@triton.jit
def _serial_scatter(
    C,
    V,
    S,
    M,
    X,
    XS,
    K,
    Y,
    YS,
    O,
    T,
    NK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    CS: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    MS: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    Q: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    Y0: tl.constexpr,
    Y1: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HB: tl.constexpr,
    BS: tl.constexpr,
    E: tl.constexpr,
    BE: tl.constexpr,
    CHUNK: tl.constexpr,
    FLAT_IDS: tl.constexpr,
    ROW_FULL: tl.constexpr,
    SCALE_FULL: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    es = tl.arange(0, BE)
    cols = tl.arange(0, HB)
    cmask = cols < H * Q
    scales = tl.arange(0, BS)
    smask = scales < D
    counts = tl.load(C + es * CS, es < E, 0)
    for ei in range(pid, E, nprog):
        start_e = tl.sum(tl.where(es < ei, counts, 0), 0)
        count_e = tl.load(C + ei * CS)
        valid_e = tl.load(V + ei * VS)
        live_e = 0
        for c0 in range(0, T * NK, CHUNK):
            cp = c0 + tl.arange(0, CHUNK)
            cm2 = cp < T * NK
            if FLAT_IDS:
                idc = tl.load(K + cp * K1, cm2, -1)
            else:
                idc = tl.load(K + (cp // NK) * K0 + (cp % NK) * K1, cm2, -1)
            live_e += tl.sum((idc == ei).to(tl.int32), 0)
        tl.store(S + ei * SS, start_e + live_e)
        m = tl.arange(0, 128)
        for base in range(0, count_e, 128):
            r = base + m
            tl.store(
                M + (start_e + r) * MS, tl.where(r < valid_e, ei, -1), r < count_e
            )
    tpp = (T + nprog - 1) // nprog
    tb0 = pid * tpp
    tb1 = tl.minimum(tb0 + tpp, T)
    scan_end = tl.where(tb0 < T, tb0 * NK, 0)
    starts_vec = tl.sum(tl.where(es[None, :] < es[:, None], counts[None, :], 0), 1)
    scan_cnt = tl.zeros([BE], dtype=tl.int32)
    for c0 in range(0, scan_end, CHUNK):
        cp = c0 + tl.arange(0, CHUNK)
        cm2 = cp < scan_end
        if FLAT_IDS:
            idc = tl.load(K + cp * K1, cm2, -1)
        else:
            idc = tl.load(K + (cp // NK) * K0 + (cp % NK) * K1, cm2, -1)
        scan_cnt += tl.sum((idc[None, :] == es[:, None]).to(tl.int32), 1)
    nxt = starts_vec + scan_cnt
    for ti in range(tb0, tb1):
        token = ti.to(tl.int64)
        row = tl.load(X + token * X0 + (cols // Q) * X1 + cols % Q, cmask, 0)
        if HAS_SCALE:
            srow = tl.load(XS + token * XS0 + scales * XS1, smask, 0)
        for j in tl.static_range(0, NK):
            e = tl.load(K + token * K0 + j * K1)
            if e >= 0:
                dest = tl.sum(tl.where(es == e, nxt, 0))
                d64 = dest.to(tl.int64)
                tl.store(Y + d64 * Y0 + (cols // Q) * Y1 + cols % Q, row, cmask)
                if HAS_SCALE:
                    tl.store(YS + d64 * YS0 + scales * YS1, srow, smask)
                tl.store(O + token * O0 + j * O1, dest)
            nxt = nxt + (es == e).to(tl.int32)


@triton.jit
def _prepare(
    C,
    V,
    S,
    M,
    E: tl.constexpr,
    CS: tl.constexpr,
    VS: tl.constexpr,
    SS: tl.constexpr,
    MS: tl.constexpr,
    BE: tl.constexpr,
    B: tl.constexpr,
):
    e = tl.program_id(0)
    es = tl.arange(0, BE)
    counts = tl.load(C + es * CS, es < E, other=0)
    start = tl.sum(tl.where(es < e, counts, 0), 0)
    count = tl.load(C + e * CS)
    valid = tl.load(V + e * VS)
    tl.store(S + e * SS, start)
    offs = tl.arange(0, B)
    for base in range(0, count, B):
        r = base + offs
        tl.store(M + (start + r) * MS, tl.where(r < valid, e, -1), r < count)


@triton.jit
def _dispatch(
    X,
    XS,
    K,
    S,
    Y,
    YS,
    O,
    T: tl.constexpr,
    NK: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    X0: tl.constexpr,
    X1: tl.constexpr,
    Q: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    S0: tl.constexpr,
    Y0: tl.constexpr,
    Y1: tl.constexpr,
    YS0: tl.constexpr,
    YS1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HB: tl.constexpr,
    BS: tl.constexpr,
):
    cols = tl.arange(0, HB)
    scales = tl.arange(0, BS)
    for ti in range(tl.program_id(0), T, tl.num_programs(0)):
        token = ti.to(tl.int64)
        row = tl.load(
            X + token * X0 + (cols // Q) * X1 + cols % Q, cols < H * Q, other=0
        )
        if HAS_SCALE:
            srow = tl.load(
                XS + token * XS0 + scales * XS1, scales < D, other=0
            )
        for slot in range(NK):
            e = tl.load(K + token * K0 + slot * K1)
            if e >= 0:
                dest = tl.atomic_add(S + e * S0, 1, sem="relaxed").to(tl.int64)
                tl.store(
                    Y + dest * Y0 + (cols // Q) * Y1 + cols % Q,
                    row,
                    cols < H * Q,
                )
                if HAS_SCALE:
                    tl.store(YS + dest * YS0 + scales * YS1, srow, scales < D)
                tl.store(O + token * O0 + slot * O1, dest)


def ep_scatter(
    recv_x,
    recv_x_scale,
    recv_topk,
    num_recv_tokens_per_expert,
    num_valid_tokens_per_expert,
    expert_start_loc,
    output_tensor,
    output_tensor_scale,
    m_indices,
    output_index,
):
    e = num_recv_tokens_per_expert.numel()
    t, k = recv_topk.shape
    h = recv_x.shape[1]
    q = recv_x.element_size()
    has_scale = recv_x_scale is not None
    d = recv_x_scale.shape[1] if has_scale else 0
    if (
        e
        and t
        and k
        and e <= _MAX_SERIAL_EXPERTS
        and t * k <= _MAX_SERIAL_ENTRIES
    ):
        programs = _SMALL_PROGRAMS if t <= _SMALL_T_TOKENS else _LARGE_PROGRAMS
        flat_ids = (
            recv_topk.stride(1) == 1
            and recv_topk.stride(0) == k * recv_topk.stride(1)
        )
        row_full = triton.next_power_of_2(h * q) == h * q
        scale_full = has_scale and triton.next_power_of_2(max(d, 1)) == d
        _serial_args = (
            num_recv_tokens_per_expert,
            num_valid_tokens_per_expert,
            expert_start_loc,
            m_indices,
            triton.reinterpret(recv_x, tl.uint8),
            recv_x_scale,
            recv_topk,
            triton.reinterpret(output_tensor, tl.uint8),
            output_tensor_scale,
            output_index,
            t,
            k,
            h,
            d,
            num_recv_tokens_per_expert.stride(0),
            num_valid_tokens_per_expert.stride(0),
            expert_start_loc.stride(0),
            m_indices.stride(0),
            recv_x.stride(0) * q,
            recv_x.stride(1) * q,
            q,
            recv_x_scale.stride(0) if has_scale else 0,
            recv_x_scale.stride(1) if has_scale else 0,
            recv_topk.stride(0),
            recv_topk.stride(1),
            output_tensor.stride(0) * q,
            output_tensor.stride(1) * q,
            output_tensor_scale.stride(0) if has_scale else 0,
            output_tensor_scale.stride(1) if has_scale else 0,
            output_index.stride(0),
            output_index.stride(1),
            has_scale,
            triton.next_power_of_2(h * q),
            triton.next_power_of_2(max(d, 1)),
            e,
            triton.next_power_of_2(e),
            min(2048, triton.next_power_of_2(t * k)),
            flat_ids,
            row_full,
            scale_full,
        )
        kernel = _serial_scatter8 if e <= 8 else _serial_scatter
        kernel[(min(programs, t),)](*_serial_args, num_warps=1)
        return (
            expert_start_loc,
            output_tensor,
            output_tensor_scale,
            m_indices,
            output_index,
        )
    if e:
        _prepare.run(
            num_recv_tokens_per_expert,
            num_valid_tokens_per_expert,
            expert_start_loc,
            m_indices,
            e,
            num_recv_tokens_per_expert.stride(0),
            num_valid_tokens_per_expert.stride(0),
            expert_start_loc.stride(0),
            m_indices.stride(0),
            triton.next_power_of_2(e),
            1024,
            grid=(e,),
            warmup=False,
        )
    if t and k:
        _dispatch.run(
            triton.reinterpret(recv_x, tl.uint8),
            recv_x_scale,
            recv_topk,
            expert_start_loc,
            triton.reinterpret(output_tensor, tl.uint8),
            output_tensor_scale,
            output_index,
            t,
            k,
            h,
            d,
            recv_x.stride(0) * q,
            recv_x.stride(1) * q,
            q,
            recv_x_scale.stride(0) if has_scale else 0,
            recv_x_scale.stride(1) if has_scale else 0,
            recv_topk.stride(0),
            recv_topk.stride(1),
            expert_start_loc.stride(0),
            output_tensor.stride(0) * q,
            output_tensor.stride(1) * q,
            output_tensor_scale.stride(0) if has_scale else 0,
            output_tensor_scale.stride(1) if has_scale else 0,
            output_index.stride(0),
            output_index.stride(1),
            has_scale,
            triton.next_power_of_2(h * q),
            triton.next_power_of_2(max(d, 1)),
            num_warps=1,
            grid=(min(t, 8192),),
            warmup=False,
        )
    return (
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_index,
    )


__all__ = ["ep_scatter"]
