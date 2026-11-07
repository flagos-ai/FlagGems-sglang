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
def _fill_zero_output(OUT, META: tl.constexpr):
    COUNT: tl.constexpr = META[0]
    B: tl.constexpr = META[1]
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(OUT + i, 0.0, i < COUNT)


@triton.jit
def _step_a_v_kernel(X, A, OUT, SEG, WIDX, RANKS, PERM, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    SC: tl.constexpr = META[5]
    XS0: tl.constexpr = META[6]
    XS1: tl.constexpr = META[7]
    XS2: tl.constexpr = META[8]
    AS0: tl.constexpr = META[9]
    AS1: tl.constexpr = META[10]
    AS2: tl.constexpr = META[11]
    BM: tl.constexpr = META[12]
    BN: tl.constexpr = META[13]
    BK: tl.constexpr = META[14]
    EVEN_K: tl.constexpr = META[15]
    FLAT: tl.constexpr = META[16]
    HAS_PERM: tl.constexpr = META[17]
    PERM_SIZE: tl.constexpr = META[18]
    HIP: tl.constexpr = META[19]
    m0 = tl.program_id(0) * BM
    m = m0 + tl.arange(0, BM)
    keep = m < M
    if FLAT:
        pos = m
        p0 = m0
        p1 = m0 + BM - 1
        xoff = m * XS1
        ooff = m * N
    else:
        idx = m // H
        head = m - idx * H
        if HAS_PERM:
            keep = keep & (idx < PERM_SIZE)
            rows = tl.load(PERM + idx, mask=keep, other=0).to(tl.int32)
        else:
            rows = idx
        pos = idx
        p0 = m0 // H
        p1 = (m0 + BM - 1) // H
        xoff = rows * XS0 + head * XS1
        ooff = (rows * H + head) * N
    n0 = tl.program_id(1) * BN if N > BN else 0
    offs_n = n0 + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for b in range(BS):
        lo = tl.load(SEG + b) * SC
        hi = tl.load(SEG + b + 1) * SC
        if (lo < hi) & (lo <= p1) & (hi > p0):
            w = tl.load(WIDX + b)
            if tl.load(RANKS + w) > 0:
                rmask = keep & (pos >= lo) & (pos < hi)
                if H % BM == 0:
                    partial = acc
                else:
                    partial = tl.zeros((BM, BN), dtype=tl.float32)
                xp = X + xoff[:, None] + offs_k[None, :] * XS2
                ap = (
                    A + w * AS0 + offs_n[None, :] * AS1 + offs_k[:, None] * AS2
                )
                for k_start in range(0, K, BK):
                    if EVEN_K:
                        x = tl.load(
                            xp,
                            mask=rmask[:, None],
                            other=0.0,
                            cache_modifier=".cg" if HIP else "",
                            eviction_policy="" if HIP else "evict_first",
                        )
                        a = tl.load(ap, mask=nmask[None, :], other=0.0)
                    else:
                        x = tl.load(
                            xp,
                            mask=rmask[:, None]
                            & (k_start + offs_k[None, :] < K),
                            other=0.0,
                            cache_modifier=".cg" if HIP else "",
                            eviction_policy="" if HIP else "evict_first",
                        )
                        a = tl.load(
                            ap,
                            mask=nmask[None, :]
                            & (k_start + offs_k[:, None] < K),
                            other=0.0,
                        )
                    if X.dtype.element_ty == tl.float32:
                        partial = tl.dot(
                            x,
                            a,
                            partial,
                            input_precision="ieee" if HIP else "tf32x3",
                        )
                    else:
                        partial = tl.dot(x, a, partial)
                    xp += BK * XS2
                    ap += BK * AS2
                if H % BM == 0:
                    acc = partial
                else:
                    acc = tl.where(rmask[:, None], partial, acc)
    tl.store(
        OUT + ooff[:, None] + offs_n[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=keep[:, None] & nmask[None, :],
    )


@triton.jit
def _step_block_kernel(X, A, OUT, SEG, WIDX, RANKS, PERM, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    SC: tl.constexpr = META[5]
    XS1: tl.constexpr = META[7]
    XS2: tl.constexpr = META[8]
    AS0: tl.constexpr = META[9]
    AS1: tl.constexpr = META[10]
    AS2: tl.constexpr = META[11]
    BM: tl.constexpr = META[12]
    BN: tl.constexpr = META[13]
    BK: tl.constexpr = META[14]
    FLAT: tl.constexpr = META[16]
    HAS_PERM: tl.constexpr = META[17]
    PERM_SIZE: tl.constexpr = META[18]
    HIP: tl.constexpr = META[19]
    m0 = tl.program_id(0) * BM
    m = m0 + tl.arange(0, BM)
    keep = m < M
    if FLAT:
        pos = m
        p0 = m0
        p1 = m0 + BM - 1
        ooff = m * N
    else:
        idx = m // H
        head = m - idx * H
        if HAS_PERM:
            keep = keep & (idx < PERM_SIZE)
            rows = tl.load(PERM + idx, mask=keep, other=0).to(tl.int32)
        else:
            rows = idx
        pos = idx
        p0 = m0 // H
        p1 = (m0 + BM - 1) // H
        ooff = (rows * H + head) * N
    n0 = tl.program_id(1) * BN if N > BN else 0
    offs_n = n0 + tl.arange(0, BN)
    nmask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for b in range(BS):
        lo = tl.load(SEG + b) * SC
        hi = tl.load(SEG + b + 1) * SC
        if (lo < hi) & (lo <= p1) & (hi > p0):
            w = tl.load(WIDX + b)
            if tl.load(RANKS + w) > 0:
                rmask = keep & (pos >= lo) & (pos < hi)
                if H % BM == 0:
                    partial = acc
                else:
                    partial = tl.zeros((BM, BN), dtype=tl.float32)
                xp = tl.make_block_ptr(
                    base=X + lo * XS1,
                    shape=(hi - lo, K),
                    strides=(XS1, XS2),
                    offsets=(m0 - lo, 0),
                    block_shape=(BM, BK),
                    order=(1, 0),
                )
                ap = tl.make_block_ptr(
                    base=A + w * AS0,
                    shape=(K, N),
                    strides=(AS2, AS1),
                    offsets=(0, n0),
                    block_shape=(BK, BN),
                    order=(0, 1),
                )
                for k_start in range(0, K, BK):
                    x = tl.load(
                        xp,
                        boundary_check=(0, 1),
                        padding_option="zero",
                        cache_modifier=".cg" if HIP else "",
                        eviction_policy="" if HIP else "evict_first",
                    )
                    a = tl.load(
                        ap, boundary_check=(0, 1), padding_option="zero"
                    )
                    if X.dtype.element_ty == tl.float32:
                        partial = tl.dot(
                            x,
                            a,
                            partial,
                            input_precision="ieee" if HIP else "tf32x3",
                        )
                    else:
                        partial = tl.dot(x, a, partial)
                    xp = tl.advance(xp, (0, BK))
                    ap = tl.advance(ap, (BK, 0))
                if H % BM == 0:
                    acc = partial
                else:
                    acc = tl.where(rmask[:, None], partial, acc)
    tl.store(
        OUT + ooff[:, None] + offs_n[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=keep[:, None] & nmask[None, :],
    )


@triton.jit
def _segment_kernel(X, A, OUT, SEG, WIDX, RANKS, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    TILES: tl.constexpr = META[5]
    XS1: tl.constexpr = META[6]
    XS2: tl.constexpr = META[7]
    AS0: tl.constexpr = META[8]
    AS1: tl.constexpr = META[9]
    AS2: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    pid = tl.program_id(0)
    b = pid // TILES
    tile = pid % TILES
    if b == BS:
        if N > BN:
            if tl.program_id(1) > 0:
                return
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        z = tl.arange(0, 1024)
        for start in range(0, first, 1024):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < first)
        for start in range(last, M * N, 1024):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < M * N)
        return
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    m0 = lo + tile * BM
    if m0 >= hi:
        return
    m = m0 + tl.arange(0, BM)
    keep = m < hi
    n0 = tl.program_id(1) * BN if N > BN else 0
    ns = n0 + tl.arange(0, BN)
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        tl.store(
            OUT + m[:, None] * N + ns[None, :],
            0.0,
            mask=keep[:, None] & (ns[None, :] < N),
        )
        return
    xb = tl.make_block_ptr(
        base=X + lo * XS1,
        shape=(hi - lo, K),
        strides=(XS1, XS2),
        offsets=(tile * BM, 0),
        block_shape=(BM, BK),
        order=(1, 0),
    )
    ab = tl.make_block_ptr(
        base=A + w * AS0,
        shape=(N, K),
        strides=(AS1, AS2),
        offsets=(n0, 0),
        block_shape=(BN, BK),
        order=(1, 0),
    )
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(
            xb,
            boundary_check=(0, 1),
            padding_option="zero",
            eviction_policy="evict_first",
        )
        a = tl.load(ab, boundary_check=(0, 1), padding_option="zero")
        if X.dtype.element_ty == tl.float32:
            acc = tl.dot(x, tl.trans(a), acc, input_precision="tf32x3")
        else:
            acc = tl.dot(x, tl.trans(a), acc)
        xb = tl.advance(xb, (0, BK))
        ab = tl.advance(ab, (0, BK))
    tl.store(
        OUT + m[:, None] * N + ns[None, :],
        acc,
        mask=keep[:, None] & (ns[None, :] < N),
    )


@triton.jit
def _hip_segment_kernel(X, A, OUT, SEG, WIDX, RANKS, META: tl.constexpr):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    TILES: tl.constexpr = META[5]
    XS1: tl.constexpr = META[6]
    XS2: tl.constexpr = META[7]
    AS0: tl.constexpr = META[8]
    AS1: tl.constexpr = META[9]
    AS2: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    pid = tl.program_id(0)
    b = pid // TILES
    tile = pid % TILES
    if b == BS:
        if N > BN:
            if tl.program_id(1) > 0:
                return
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        z = tl.arange(0, 1024)
        for start in range(0, first, 1024):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < first)
        for start in range(last, M * N, 1024):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < M * N)
        return
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    m0 = lo + tile * BM
    if m0 >= hi:
        return
    m = m0 + tl.arange(0, BM)
    keep = m < hi
    n0 = tl.program_id(1) * BN if N > BN else 0
    ns = n0 + tl.arange(0, BN)
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        tl.store(
            OUT + m[:, None] * N + ns[None, :],
            0.0,
            mask=keep[:, None] & (ns[None, :] < N),
        )
        return
    xb = tl.make_block_ptr(
        base=X + lo * XS1,
        shape=(hi - lo, K),
        strides=(XS1, XS2),
        offsets=(tile * BM, 0),
        block_shape=(BM, BK),
        order=(1, 0),
    )
    ab = tl.make_block_ptr(
        base=A + w * AS0,
        shape=(K, N),
        strides=(AS2, AS1),
        offsets=(0, n0),
        block_shape=(BK, BN),
        order=(0, 1),
    )
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(
            xb,
            boundary_check=(0, 1),
            padding_option="zero",
            cache_modifier=".cg",
        )
        a = tl.load(ab, boundary_check=(0, 1), padding_option="zero")
        if X.dtype.element_ty == tl.float32:
            acc = tl.dot(x, a, acc, input_precision="ieee")
        else:
            acc = tl.dot(x, a, acc)
        xb = tl.advance(xb, (0, BK))
        ab = tl.advance(ab, (BK, 0))
    tl.store(
        OUT + m[:, None] * N + ns[None, :],
        acc,
        mask=keep[:, None] & (ns[None, :] < N),
    )


@triton.jit
def _step_direct_reverse_kernel(
    X, A, OUT, SEG, WIDX, RANKS, PERM, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    SC: tl.constexpr = META[5]
    XS1: tl.constexpr = META[7]
    XS2: tl.constexpr = META[8]
    AS0: tl.constexpr = META[9]
    AS1: tl.constexpr = META[10]
    AS2: tl.constexpr = META[11]
    BM: tl.constexpr = META[12]
    BN: tl.constexpr = META[13]
    BK: tl.constexpr = META[14]
    FLAT: tl.constexpr = META[16]
    HAS_PERM: tl.constexpr = META[17]
    PERM_SIZE: tl.constexpr = META[18]
    HIP: tl.constexpr = META[19]
    m0 = tl.program_id(0) * BM
    m = m0 + tl.arange(0, BM)
    keep = m < M
    if FLAT:
        pos = m
        p0 = m0
        p1 = m0 + BM - 1
        ooff = m * N
    else:
        idx = m // H
        head = m - idx * H
        if HAS_PERM:
            keep = keep & (idx < PERM_SIZE)
            rows = tl.load(PERM + idx, mask=keep, other=0).to(tl.int32)
        else:
            rows = idx
        pos = idx
        p0 = m0 // H
        p1 = (m0 + BM - 1) // H
        ooff = (rows * H + head) * N
    n0 = tl.program_id(1) * BN if N > BN else 0
    offs_n = n0 + tl.arange(0, BN)
    nmask = offs_n < N
    acc = tl.zeros((BN, BM), dtype=tl.float32)
    for b in range(BS):
        lo = tl.load(SEG + b) * SC
        hi = tl.load(SEG + b + 1) * SC
        if (lo < hi) & (lo <= p1) & (hi > p0):
            w = tl.load(WIDX + b)
            if tl.load(RANKS + w) > 0:
                rmask = keep & (pos >= lo) & (pos < hi)
                if H % BM == 0:
                    partial = acc
                else:
                    partial = tl.zeros((BN, BM), dtype=tl.float32)
                xp = tl.make_block_ptr(
                    base=X + lo * XS1,
                    shape=(K, hi - lo),
                    strides=(XS2, XS1),
                    offsets=(0, m0 - lo),
                    block_shape=(BK, BM),
                    order=(0, 1),
                )
                ap = tl.make_block_ptr(
                    base=A + w * AS0,
                    shape=(N, K),
                    strides=(AS1, AS2),
                    offsets=(n0, 0),
                    block_shape=(BN, BK),
                    order=(1, 0),
                )
                for k_start in range(0, K, BK):
                    x = tl.load(
                        xp,
                        boundary_check=(0, 1),
                        padding_option="zero",
                        cache_modifier=".cg" if HIP else "",
                        eviction_policy="" if HIP else "evict_first",
                    )
                    a = tl.load(
                        ap, boundary_check=(0, 1), padding_option="zero"
                    )
                    if X.dtype.element_ty == tl.float32:
                        partial = tl.dot(
                            a,
                            x,
                            partial,
                            input_precision="ieee" if HIP else "tf32x3",
                        )
                    else:
                        partial = tl.dot(a, x, partial)
                    xp = tl.advance(xp, (BK, 0))
                    ap = tl.advance(ap, (0, BK))
                if H % BM == 0:
                    acc = partial
                else:
                    acc = tl.where(rmask[None, :], partial, acc)
    tl.store(
        OUT + ooff[:, None] + offs_n[None, :],
        tl.trans(acc).to(OUT.dtype.element_ty),
        mask=keep[:, None] & nmask[None, :],
    )


@triton.jit
def _hip_reverse_segment_kernel(
    X, A, OUT, SEG, WIDX, RANKS, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    TILES: tl.constexpr = META[5]
    XS1: tl.constexpr = META[6]
    XS2: tl.constexpr = META[7]
    AS0: tl.constexpr = META[8]
    AS1: tl.constexpr = META[9]
    AS2: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    pid = tl.program_id(0)
    b = pid // TILES
    tile = pid % TILES
    if b == BS:
        if N > BN:
            if tl.program_id(1) > 0:
                return
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        z = tl.arange(0, 1024)
        for start in range(0, first, 1024):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < first)
        for start in range(last, M * N, 1024):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < M * N)
        return
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    m0 = lo + tile * BM
    if m0 >= hi:
        return
    m = m0 + tl.arange(0, BM)
    keep = m < hi
    n0 = tl.program_id(1) * BN if N > BN else 0
    ns = n0 + tl.arange(0, BN)
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        tl.store(
            OUT + m[:, None] * N + ns[None, :],
            0.0,
            mask=keep[:, None] & (ns[None, :] < N),
        )
        return
    xb = tl.make_block_ptr(
        base=X + lo * XS1,
        shape=(K, hi - lo),
        strides=(XS2, XS1),
        offsets=(0, tile * BM),
        block_shape=(BK, BM),
        order=(0, 1),
    )
    ab = tl.make_block_ptr(
        base=A + w * AS0,
        shape=(N, K),
        strides=(AS1, AS2),
        offsets=(n0, 0),
        block_shape=(BN, BK),
        order=(1, 0),
    )
    acc = tl.zeros((BN, BM), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(
            xb,
            boundary_check=(0, 1),
            padding_option="zero",
            cache_modifier=".cg",
        )
        a = tl.load(ab, boundary_check=(0, 1), padding_option="zero")
        if X.dtype.element_ty == tl.float32:
            acc = tl.dot(a, x, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, x, acc)
        xb = tl.advance(xb, (BK, 0))
        ab = tl.advance(ab, (0, BK))
    tl.store(
        OUT + m[:, None] * N + ns[None, :],
        tl.trans(acc),
        mask=keep[:, None] & (ns[None, :] < N),
    )


@triton.jit
def _hip_reverse_segment_kernel_storecs(
    X, A, OUT, SEG, WIDX, RANKS, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    TILES: tl.constexpr = META[5]
    XS1: tl.constexpr = META[6]
    XS2: tl.constexpr = META[7]
    AS0: tl.constexpr = META[8]
    AS1: tl.constexpr = META[9]
    AS2: tl.constexpr = META[10]
    BM: tl.constexpr = META[11]
    BN: tl.constexpr = META[12]
    BK: tl.constexpr = META[13]
    pid = tl.program_id(0)
    b = pid // TILES
    tile = pid % TILES
    if b == BS:
        if N > BN:
            if tl.program_id(1) > 0:
                return
        first = tl.load(SEG) * H * N
        last = tl.load(SEG + BS) * H * N
        z = tl.arange(0, 1024)
        for start in range(0, first, 1024):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < first, cache_modifier=".cs")
        for start in range(last, M * N, 1024):
            p = start + z
            tl.store(OUT + p, 0.0, mask=p < M * N, cache_modifier=".cs")
        return
    lo = tl.load(SEG + b) * H
    hi = tl.load(SEG + b + 1) * H
    m0 = lo + tile * BM
    if m0 >= hi:
        return
    m = m0 + tl.arange(0, BM)
    keep = m < hi
    n0 = tl.program_id(1) * BN if N > BN else 0
    ns = n0 + tl.arange(0, BN)
    w = tl.load(WIDX + b)
    if tl.load(RANKS + w) == 0:
        tl.store(
            OUT + m[:, None] * N + ns[None, :],
            0.0,
            mask=keep[:, None] & (ns[None, :] < N),
            cache_modifier=".cs",
        )
        return
    xb = tl.make_block_ptr(
        base=X + lo * XS1,
        shape=(K, hi - lo),
        strides=(XS2, XS1),
        offsets=(0, tile * BM),
        block_shape=(BK, BM),
        order=(0, 1),
    )
    ab = tl.make_block_ptr(
        base=A + w * AS0,
        shape=(N, K),
        strides=(AS1, AS2),
        offsets=(n0, 0),
        block_shape=(BN, BK),
        order=(1, 0),
    )
    acc = tl.zeros((BN, BM), tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(
            xb,
            boundary_check=(0, 1),
            padding_option="zero",
            cache_modifier=".cg",
        )
        a = tl.load(ab, boundary_check=(0, 1), padding_option="zero")
        if X.dtype.element_ty == tl.float32:
            acc = tl.dot(a, x, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, x, acc)
        xb = tl.advance(xb, (BK, 0))
        ab = tl.advance(ab, (0, BK))
    tl.store(
        OUT + m[:, None] * N + ns[None, :],
        tl.trans(acc),
        mask=keep[:, None] & (ns[None, :] < N),
        cache_modifier=".cs",
    )


@triton.jit
def _step_block_kernel_storecs(
    X, A, OUT, SEG, WIDX, RANKS, PERM, META: tl.constexpr
):
    H: tl.constexpr = META[0]
    N: tl.constexpr = META[1]
    M: tl.constexpr = META[2]
    K: tl.constexpr = META[3]
    BS: tl.constexpr = META[4]
    SC: tl.constexpr = META[5]
    XS1: tl.constexpr = META[7]
    XS2: tl.constexpr = META[8]
    AS0: tl.constexpr = META[9]
    AS1: tl.constexpr = META[10]
    AS2: tl.constexpr = META[11]
    BM: tl.constexpr = META[12]
    BN: tl.constexpr = META[13]
    BK: tl.constexpr = META[14]
    FLAT: tl.constexpr = META[16]
    HAS_PERM: tl.constexpr = META[17]
    PERM_SIZE: tl.constexpr = META[18]
    HIP: tl.constexpr = META[19]
    m0 = tl.program_id(0) * BM
    m = m0 + tl.arange(0, BM)
    keep = m < M
    if FLAT:
        pos = m
        p0 = m0
        p1 = m0 + BM - 1
        ooff = m * N
    else:
        idx = m // H
        head = m - idx * H
        if HAS_PERM:
            keep = keep & (idx < PERM_SIZE)
            rows = tl.load(PERM + idx, mask=keep, other=0).to(tl.int32)
        else:
            rows = idx
        pos = idx
        p0 = m0 // H
        p1 = (m0 + BM - 1) // H
        ooff = (rows * H + head) * N
    n0 = tl.program_id(1) * BN if N > BN else 0
    offs_n = n0 + tl.arange(0, BN)
    nmask = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for b in range(BS):
        lo = tl.load(SEG + b) * SC
        hi = tl.load(SEG + b + 1) * SC
        if (lo < hi) & (lo <= p1) & (hi > p0):
            w = tl.load(WIDX + b)
            if tl.load(RANKS + w) > 0:
                rmask = keep & (pos >= lo) & (pos < hi)
                if H % BM == 0:
                    partial = acc
                else:
                    partial = tl.zeros((BM, BN), dtype=tl.float32)
                xp = tl.make_block_ptr(
                    base=X + lo * XS1,
                    shape=(hi - lo, K),
                    strides=(XS1, XS2),
                    offsets=(m0 - lo, 0),
                    block_shape=(BM, BK),
                    order=(1, 0),
                )
                ap = tl.make_block_ptr(
                    base=A + w * AS0,
                    shape=(K, N),
                    strides=(AS2, AS1),
                    offsets=(0, n0),
                    block_shape=(BK, BN),
                    order=(0, 1),
                )
                for k_start in range(0, K, BK):
                    x = tl.load(
                        xp,
                        boundary_check=(0, 1),
                        padding_option="zero",
                        cache_modifier=".cg" if HIP else "",
                        eviction_policy="" if HIP else "evict_first",
                    )
                    a = tl.load(
                        ap, boundary_check=(0, 1), padding_option="zero"
                    )
                    if X.dtype.element_ty == tl.float32:
                        partial = tl.dot(
                            x,
                            a,
                            partial,
                            input_precision="ieee" if HIP else "tf32x3",
                        )
                    else:
                        partial = tl.dot(x, a, partial)
                    xp = tl.advance(xp, (0, BK))
                    ap = tl.advance(ap, (BK, 0))
                if H % BM == 0:
                    acc = partial
                else:
                    acc = tl.where(rmask[:, None], partial, acc)
    tl.store(
        OUT + ooff[:, None] + offs_n[None, :],
        acc.to(OUT.dtype.element_ty),
        mask=keep[:, None] & nmask[None, :],
        cache_modifier=".cs",
    )


def step_a_v(attn_output, A_buf, batch_info):
    S, H, K = attn_output.shape
    N = A_buf.shape[1]
    xs0, xs1, xs2 = attn_output.stride()
    seg = batch_info.seg_indptr
    widx = batch_info.weight_indices
    ranks = batch_info.lora_ranks
    perm = batch_info.permutation
    if seg.stride(0) != 1:
        seg = seg.contiguous()
    if widx.stride(0) != 1:
        widx = widx.contiguous()
    if ranks.stride(0) != 1:
        ranks = ranks.contiguous()
    if perm is not None and perm.stride(0) != 1:
        perm = perm.contiguous()
    has_perm = perm is not None
    if has_perm and perm.numel() == 0:
        out = attn_output.new_empty((S, H, N))
        if S * H * N > 0:
            _fill_zero_output.run(
                out,
                (S * H * N, 256),
                grid=((S * H * N + 256 - 1) // 256,),
                warmup=False,
            )
        return out
    flat = not has_perm and xs0 == H * xs1
    if has_perm and perm.numel() != S:
        out = torch.zeros(
            (S, H, N), dtype=attn_output.dtype, device=attn_output.device
        )
    else:
        out = attn_output.new_empty((S, H, N))
        if not has_perm:
            perm = None
    m_total = S * H
    bn = min(32, max(16, 0 if N == 0 else 1 << (N - 1).bit_length()))
    kp = max(16, 0 if K == 0 else 1 << (K - 1).bit_length())
    hip = torch.version.hip is not None
    if m_total <= 4096:
        bm = 16
        bk = min(512, kp)
        stages = 2
    elif hip and N >= 32:
        bm = 128
        bk = min(128, kp)
        stages = 2
    else:
        bm = 64
        bk = min(128, kp)
        stages = 3
    if attn_output.dtype == torch.float32:
        bk = min(64, bk)
    if not hip and flat and (m_total > 4096):
        ml = batch_info.max_len
        if not isinstance(ml, int) or ml <= 0 or ml > S:
            ml = S
        tiles = max(1, (ml * H + bm - 1) // bm)
        _segment_kernel.run(
            attn_output,
            A_buf,
            out,
            seg,
            widx,
            ranks,
            (
                H,
                N,
                m_total,
                K,
                batch_info.bs,
                tiles,
                xs1,
                xs2,
                *A_buf.stride(),
                bm,
                bn,
                bk,
                K % bk == 0,
            ),
            grid=(tiles * batch_info.bs + 1, (N + bn - 1) // bn),
            warmup=False,
            num_warps=4,
            num_stages=stages,
        )
        return out
    if (
        hip
        and flat
        and (m_total > 4096)
        and (16 < N <= 32)
        and (attn_output.dtype != torch.float32)
    ):
        bm = 64
        stages = 1
        ml = batch_info.max_len
        if not isinstance(ml, int) or ml <= 0 or ml > S:
            ml = S
        tiles = max(1, (ml * H + bm - 1) // bm)
        _hip_reverse_segment_kernel_storecs.run(
            attn_output,
            A_buf,
            out,
            seg,
            widx,
            ranks,
            (
                H,
                N,
                m_total,
                K,
                batch_info.bs,
                tiles,
                xs1,
                xs2,
                *A_buf.stride(),
                bm,
                bn,
                bk,
                K % bk == 0,
            ),
            grid=(tiles * batch_info.bs + 1, (N + bn - 1) // bn),
            warmup=False,
            num_warps=4,
            num_stages=stages,
        )
        return out
    if (
        hip
        and flat
        and (m_total > 4096)
        and (N <= 16)
        and (K > 128)
        and (attn_output.dtype != torch.float32)
    ):
        bk = min(256, kp)
        stages = 1
    grid = ((m_total + bm - 1) // bm, (N + bn - 1) // bn)
    if (
        hip
        and flat
        and (m_total <= 4096)
        and (K > 128)
        and (N <= 32)
        and (attn_output.dtype != torch.float32)
    ):
        bk = min(128, bk)
        bn = min(16, bn)
        ml = batch_info.max_len
        if not isinstance(ml, int) or ml <= 0 or ml > S:
            ml = S
        tiles = max(1, (ml * H + bm - 1) // bm)
        _hip_segment_kernel.run(
            attn_output,
            A_buf,
            out,
            seg,
            widx,
            ranks,
            (
                H,
                N,
                m_total,
                K,
                batch_info.bs,
                tiles,
                xs1,
                xs2,
                *A_buf.stride(),
                bm,
                bn,
                bk,
                K % bk == 0,
            ),
            grid=(tiles * batch_info.bs + 1, (N + bn - 1) // bn),
            warmup=False,
            num_warps=4,
            num_stages=stages,
        )
        return out
    kernel = (
        _step_direct_reverse_kernel
        if (
            hip
            and flat
            and (attn_output.dtype != torch.float32)
            and (N <= 32)
            and (m_total > 4096 or K > 128)
        )
        and N > 16
        else (
            _step_block_kernel_storecs
            if hip
            and flat
            and (attn_output.dtype != torch.float32)
            and (N <= 32)
            and (m_total > 4096 or K > 128)
            else _step_a_v_kernel
        )
    )
    kernel.run(
        attn_output,
        A_buf,
        out,
        seg,
        widx,
        ranks,
        perm,
        (
            H,
            N,
            m_total,
            K,
            batch_info.bs,
            H if flat else 1,
            xs0,
            xs1,
            xs2,
            *A_buf.stride(),
            bm,
            bn,
            bk,
            K % bk == 0,
            flat,
            has_perm,
            perm.numel() if has_perm else 0,
            hip,
        ),
        grid=grid,
        warmup=False,
        num_warps=4,
        num_stages=stages,
    )
    return out


__all__ = ["step_a_v"]
