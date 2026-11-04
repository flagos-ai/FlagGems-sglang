# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton implementation of SGLang apply_rope_with_cos_sin_cache_inplace.

Rotary position embedding applied in place to ``query`` and ``key``.

Rotary dimension ``D`` is taken from ``cos_sin_cache.shape[-1]``; the first half
of the last cache dim holds cos, the second half holds sin.  Lanes in
``[D, head_size)`` of every head are passed through unrotated.  The rotation is
computed in float32 and written back in the tensor's original dtype.

Structure
---------
Each program owns a ``[BN, HP, L]`` tile of one tensor: ``BN`` consecutive
tokens, ``HP`` consecutive heads, and ``L`` innermost lanes that are *contiguous
in memory*, so every operand access is unit-stride and vectorizable.

* neox      : ``a = x[..., 0:HALF]`` and ``b = x[..., HALF:D]`` are two
              contiguous loads; ``o1``/``o2`` go back to the same lane ranges.
* non-neox  : the ``D`` rotary lanes are loaded as one contiguous tile, viewed as
              ``[..., HALF, 2]`` and split into the even lanes ``a`` and the odd
              lanes ``b``; re-joining is a no-op relabeling.

``cos``/``sin`` are gathered once per token (head invariant) and broadcast over
``HP``.

Both tensors share **one** launch.  The grid's second axis is split into a query
range (``YQ`` blocks) and a key range (``YK`` blocks), so the two per-token
problems that are numerically independent no longer serialize as two
under-filled kernels.  ``HP`` always divides both head counts, so no tile is
ragged; ``EQ``/``EK`` suppress all masking when the tile divides the problem
exactly.  A single-tensor launch (``YK == 0``) is used as the fallback when the
two tensors cannot share a launch.

Grid shaping
------------
The tile is grown to ``TILE_LANES`` and then shrunk one tile step at a time
until the grid has at least 256 programs, because these shapes are limited by
resident block count rather than by bytes or instructions.  Two plans share that
machinery and differ only in shape constants:

* the streaming plan is used when the in-place traffic fits in L2.  Here the
  host keeps the grid very wide (``SMALL_MIN_PROGRAMS``) and gives the most
  lanes to registers (``SMALL_LANES_PER_WARP``), because the measured optimum on
  these shapes is many small blocks rather than a few deep ones;
* the saturated plan is used once the traffic exceeds L2.  There the grid is
  already far past one wave, so the tile is allowed two token rows and the head
  block is capped at ``LARGE_HEAD_BLOCK``, which measured 5% faster on the one
  workload big enough to reach DRAM.
"""

import math

import triton
import triton.language as tl

TILE_LANES = 256
SMALL_MIN_PROGRAMS = 256
SMALL_LANES_PER_WARP = 64
LARGE_MIN_PROGRAMS = 1024
LARGE_LANES_PER_WARP = 128
LARGE_HEAD_BLOCK = 4
L2_CAPACITY_BYTES = 64 * 1024 * 1024


@triton.jit
def _rot_tile(
    x_ptr,
    pos_ptr,
    cache_ptr,
    offs_n,
    offs_h,
    N,
    RW,
    RC,
    H: tl.constexpr,  # heads in this tensor
    S: tl.constexpr,  # head_size
    D: tl.constexpr,  # rotary_dim
    BN: tl.constexpr,  # tokens per program
    HP: tl.constexpr,  # heads per program
    PN: tl.constexpr,  # next pow2 of (S - D)
    IS_NEOX: tl.constexpr,
    EVEN: tl.constexpr,  # N % BN == 0 and H % HP == 0
):
    HALF: tl.constexpr = D // 2
    PASS: tl.constexpr = S - D

    row = offs_n[:, None, None] * RW + offs_h[None, :, None] * S

    if EVEN:
        pos = tl.load(pos_ptr + offs_n)
    else:
        pos = tl.load(pos_ptr + offs_n, mask=offs_n < N, other=0)
    cp = (pos.to(tl.int32) * RC)[:, None, None]

    if IS_NEOX:
        j = tl.arange(0, HALF)[None, None, :]
        if EVEN:
            a = tl.load(x_ptr + row + j).to(tl.float32)
            b = tl.load(x_ptr + row + HALF + j).to(tl.float32)
            cos = tl.load(cache_ptr + cp + j).to(tl.float32)
            sin = tl.load(cache_ptr + cp + HALF + j).to(tl.float32)
            tl.store(
                x_ptr + row + j, (a * cos - b * sin).to(x_ptr.dtype.element_ty)
            )
            tl.store(
                x_ptr + row + HALF + j,
                (b * cos + a * sin).to(x_ptr.dtype.element_ty),
            )
        else:
            m3 = (offs_n < N)[:, None, None] & (offs_h < H)[None, :, None]
            mc = (offs_n < N)[:, None, None]
            a = tl.load(x_ptr + row + j, mask=m3, other=0.0).to(tl.float32)
            b = tl.load(x_ptr + row + HALF + j, mask=m3, other=0.0).to(
                tl.float32
            )
            cos = tl.load(cache_ptr + cp + j, mask=mc, other=0.0).to(
                tl.float32
            )
            sin = tl.load(cache_ptr + cp + HALF + j, mask=mc, other=0.0).to(
                tl.float32
            )
            tl.store(
                x_ptr + row + j,
                (a * cos - b * sin).to(x_ptr.dtype.element_ty),
                mask=m3,
            )
            tl.store(
                x_ptr + row + HALF + j,
                (b * cos + a * sin).to(x_ptr.dtype.element_ty),
                mask=m3,
            )
    else:
        d = tl.arange(0, D)[None, None, :]
        j = tl.arange(0, HALF)[None, None, :]
        if EVEN:
            t = tl.load(x_ptr + row + d).to(tl.float32)
            cos = tl.load(cache_ptr + cp + j).to(tl.float32)
            sin = tl.load(cache_ptr + cp + HALF + j).to(tl.float32)
            a, b = tl.split(tl.reshape(t, (BN, HP, HALF, 2)))
            o = tl.reshape(
                tl.join(a * cos - b * sin, b * cos + a * sin), (BN, HP, D)
            )
            tl.store(x_ptr + row + d, o.to(x_ptr.dtype.element_ty))
        else:
            m3 = (offs_n < N)[:, None, None] & (offs_h < H)[None, :, None]
            mc = (offs_n < N)[:, None, None]
            t = tl.load(x_ptr + row + d, mask=m3, other=0.0).to(tl.float32)
            cos = tl.load(cache_ptr + cp + j, mask=mc, other=0.0).to(
                tl.float32
            )
            sin = tl.load(cache_ptr + cp + HALF + j, mask=mc, other=0.0).to(
                tl.float32
            )
            a, b = tl.split(tl.reshape(t, (BN, HP, HALF, 2)))
            o = tl.reshape(
                tl.join(a * cos - b * sin, b * cos + a * sin), (BN, HP, D)
            )
            tl.store(x_ptr + row + d, o.to(x_ptr.dtype.element_ty), mask=m3)

    if PASS > 0:
        k = tl.arange(0, PN)[None, None, :]
        km = k < PASS
        if EVEN:
            v = tl.load(x_ptr + row + D + k, mask=km, other=0.0)
            tl.store(x_ptr + row + D + k, v, mask=km)
        else:
            mk = (
                (offs_n < N)[:, None, None] & (offs_h < H)[None, :, None]
            ) & km
            v = tl.load(x_ptr + row + D + k, mask=mk, other=0.0)
            tl.store(x_ptr + row + D + k, v, mask=mk)


@triton.jit
def _rope_kernel(
    q_ptr,
    k_ptr,
    pos_ptr,
    cache_ptr,
    N,
    YQ,
    RQ,
    RK,
    RC,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    HP: tl.constexpr,
    PN: tl.constexpr,
    IS_NEOX: tl.constexpr,
    EQ: tl.constexpr,
    EK: tl.constexpr,
    HAS_K: tl.constexpr,
):
    pid_y = tl.program_id(1)
    offs_n = tl.program_id(0) * BN + tl.arange(0, BN)
    if HAS_K:
        if pid_y < YQ:
            _rot_tile(
                q_ptr,
                pos_ptr,
                cache_ptr,
                offs_n,
                pid_y * HP + tl.arange(0, HP),
                N,
                RQ,
                RC,
                HQ,
                S,
                D,
                BN,
                HP,
                PN,
                IS_NEOX,
                EQ,
            )
        else:
            _rot_tile(
                k_ptr,
                pos_ptr,
                cache_ptr,
                offs_n,
                (pid_y - YQ) * HP + tl.arange(0, HP),
                N,
                RK,
                RC,
                HK,
                S,
                D,
                BN,
                HP,
                PN,
                IS_NEOX,
                EK,
            )
    else:
        _rot_tile(
            q_ptr,
            pos_ptr,
            cache_ptr,
            offs_n,
            pid_y * HP + tl.arange(0, HP),
            N,
            RQ,
            RC,
            HQ,
            S,
            D,
            BN,
            HP,
            PN,
            IS_NEOX,
            EQ,
        )


def _tile_plan(N, HQA, HKB, rot, cap, min_programs, head_cap):
    """Grow ``[BN, HP, rot]`` up to ``cap`` lanes, then shrink until the grid has
    at least ``min_programs`` programs.  ``HP`` divides both head counts and
    ``BN`` divides ``N``, so every tile is full and ``N % BN == 0`` holds."""
    g = math.gcd(HQA, HKB)
    hp = 1
    while hp * 2 <= g and hp * 2 <= head_cap and (hp * 2) * rot <= cap:
        hp *= 2
    bn = 1
    while (
        bn * 2 <= N
        and N % (bn * 2) == 0
        and bn * 2 <= 128
        and (bn * 2) * hp * rot <= cap
    ):
        bn *= 2

    def grid(bn, hp):
        nb = (N + bn - 1) // bn
        nq = (HQA + hp - 1) // hp
        nk = 0 if HKB <= 0 else (HKB + hp - 1) // hp
        return nb * (nq + nk), nq

    total, yq = grid(bn, hp)
    while total < min_programs and (hp > 1 or bn > 1):
        if hp > 1:
            hp //= 2
        else:
            bn //= 2
        total, yq = grid(bn, hp)
    return bn, hp, yq


def _plan(N, HQ, HK, S, D, is_neox, bytes_moved):
    """Pick (BN, HP, YQ, YK) and the warps-per-block divisor for one launch."""
    rot = max(1, (D // 2) if is_neox else D)
    if bytes_moved > L2_CAPACITY_BYTES:
        bn, hp, yq = _tile_plan(
            N, HQ, HK, rot, TILE_LANES, LARGE_MIN_PROGRAMS, LARGE_HEAD_BLOCK
        )
        wpw = LARGE_LANES_PER_WARP
    else:
        bn, hp, yq = _tile_plan(
            N, HQ, HK, rot, TILE_LANES, SMALL_MIN_PROGRAMS, 16
        )
        wpw = SMALL_LANES_PER_WARP
    yk = 0 if HK <= 0 else (HK + hp - 1) // hp
    return bn, hp, yq, yk, wpw


def _launch(q, k, pos, cache, S, D, is_neox):
    N = q.shape[0]
    HQ = q.shape[1] // S
    HK = (k.shape[1] // S) if k is not None else 0
    if N == 0 or HQ == 0:
        return
    itemsize = q.element_size()
    bytes_moved = (
        2 * N * (q.shape[1] + (k.shape[1] if k is not None else 0)) * itemsize
    )
    bn, hp, yq, yk, wpw = _plan(N, HQ, HK, S, D, is_neox, bytes_moved)
    rot = max(1, (D // 2) if is_neox else D)
    num_warps = max(1, min(8, (bn * hp * rot) // wpw))
    _rope_kernel[(triton.cdiv(N, bn), yq + yk)](
        q,
        k if k is not None else q,
        pos,
        cache,
        N,
        yq,
        q.stride(0),
        (k.stride(0) if k is not None else q.stride(0)),
        cache.stride(0),
        HQ=HQ,
        HK=(HK if HK > 0 else 1),
        S=S,
        D=D,
        BN=bn,
        HP=hp,
        PN=(triton.next_power_of_2(S - D) if S > D else 1),
        IS_NEOX=bool(is_neox),
        EQ=(N % bn == 0) and (HQ % hp == 0),
        EK=((N % bn == 0) and (HK % hp == 0)) if HK > 0 else False,
        HAS_K=HK > 0,
        num_warps=num_warps,
    )


def sglang_fl_rotary_embedding(
    positions, query, key, head_size, cos_sin_cache, is_neox=True
):
    S = int(head_size)
    D = int(cos_sin_cache.shape[-1])
    pos = positions.reshape(-1)
    nx = bool(is_neox)
    n = pos.numel()
    fused = (
        query.shape[0] == n
        and key.shape[0] == n
        and query.dtype == key.dtype
        and query.stride(-1) == 1
        and key.stride(-1) == 1
        and cos_sin_cache.stride(-1) == 1
    )
    if fused:
        _launch(query, key, pos, cos_sin_cache, S, D, nx)
    else:
        _launch(query, None, pos, cos_sin_cache, S, D, nx)
        _launch(key, None, pos, cos_sin_cache, S, D, nx)
    return query, key


__all__ = ["sglang_fl_rotary_embedding"]
