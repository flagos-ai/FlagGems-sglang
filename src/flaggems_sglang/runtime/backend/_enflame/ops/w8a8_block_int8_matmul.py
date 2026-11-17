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
def _w8a8_bk_matmul(
    a_ptr,
    b_ptr,
    as_ptr,
    bs_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ASM: tl.constexpr,
    ASK: tl.constexpr,
    BSN: tl.constexpr,
    BSK: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    NK: tl.constexpr,
    K_PER_SCALE: tl.constexpr,
    N_PER_SCALE: tl.constexpr,
    GRID_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile = tl.program_id(0)
    pid_m = tile // GRID_N
    pid_n = tile - pid_m * GRID_N
    nblk = pid_n // N_PER_SCALE

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    in_m = rm < M
    in_n = rn < N

    a_ptrs = a_ptr + rm[:, None] * AM + rk[None, :] * AK
    b_ptrs = b_ptr + rn[None, :] * BN + rk[:, None] * BK
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(NK):
        ik = i // K_PER_SCALE
        if EVEN_K:
            a = tl.load(a_ptrs, mask=in_m[:, None], other=0)
            b = tl.load(b_ptrs, mask=in_n[None, :], other=0)
        else:
            in_k = (i * BLOCK_K + rk) < K
            a = tl.load(a_ptrs, mask=in_m[:, None] & in_k[None, :], other=0)
            b = tl.load(b_ptrs, mask=in_k[:, None] & in_n[None, :], other=0)
        sa = tl.load(as_ptr + rm * ASM + ik * ASK, mask=in_m, other=0.0)
        sb = tl.load(bs_ptr + nblk * BSN + ik * BSK)
        prod = tl.dot(a, b, out_dtype=tl.int32)
        acc += prod.to(tl.float32) * (sa * sb)[:, None]
        a_ptrs += BLOCK_K * AK
        b_ptrs += BLOCK_K * BK

    c_ptrs = c_ptr + rm[:, None] * CM + rn[None, :] * CN
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
    else:
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=in_m[:, None] & in_n[None, :],
        )


@triton.jit
def _w8a8_bk_matmul_unroll(
    a_ptr,
    b_ptr,
    as_ptr,
    bs_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    AM: tl.constexpr,
    AK: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ASM: tl.constexpr,
    ASK: tl.constexpr,
    BSN: tl.constexpr,
    BSK: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    NK: tl.constexpr,
    K_PER_SCALE: tl.constexpr,
    N_PER_SCALE: tl.constexpr,
    GRID_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SUBK: tl.constexpr,
):
    # Same math/tile decomposition as _w8a8_bk_matmul with the k loop unrolled
    # SUBK scale blocks per outer iteration (tl.static_range): the GCU backend
    # ignores num_stages/tl.range pipelining for this loop, so the unroll is
    # what creates load ILP.  k-order and per-block fp32 epilogue sequence are
    # identical to the plain kernel (bitwise-equal outputs).
    tile = tl.program_id(0)
    pid_m = tile // GRID_N
    pid_n = tile - pid_m * GRID_N
    nblk = pid_n // N_PER_SCALE

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    in_m = rm < M
    in_n = rn < N

    a_ptrs = a_ptr + rm[:, None] * AM + rk[None, :] * AK
    b_ptrs = b_ptr + rn[None, :] * BN + rk[:, None] * BK
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, NK, SUBK):
        for u in tl.static_range(SUBK):
            it = i + u
            ik = it // K_PER_SCALE
            if EVEN_K:
                a = tl.load(a_ptrs, mask=in_m[:, None], other=0)
                b = tl.load(b_ptrs, mask=in_n[None, :], other=0)
            else:
                in_k = (it * BLOCK_K + rk) < K
                a = tl.load(
                    a_ptrs, mask=in_m[:, None] & in_k[None, :], other=0
                )
                b = tl.load(
                    b_ptrs, mask=in_k[:, None] & in_n[None, :], other=0
                )
            sa = tl.load(as_ptr + rm * ASM + ik * ASK, mask=in_m, other=0.0)
            sb = tl.load(bs_ptr + nblk * BSN + ik * BSK)
            prod = tl.dot(a, b, out_dtype=tl.int32)
            acc += prod.to(tl.float32) * (sa * sb)[:, None]
            a_ptrs += BLOCK_K * AK
            b_ptrs += BLOCK_K * BK

    c_ptrs = c_ptr + rm[:, None] * CM + rn[None, :] * CN
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))
    else:
        tl.store(
            c_ptrs,
            acc.to(c_ptr.dtype.element_ty),
            mask=in_m[:, None] & in_n[None, :],
        )


def _pow2_at_most(value, cap):
    size = 16
    while size * 2 <= value and size * 2 <= cap:
        size = size * 2
    return size


def w8a8_block_int8_matmul(A, B, As, Bs, block_size, output_dtype):
    A = A.contiguous()
    B = B.contiguous()
    As = As.contiguous()
    Bs = Bs.contiguous()
    m, k = A.shape
    n = B.shape[0]
    block_n, block_k = int(block_size[0]), int(block_size[1])
    out = torch.empty((m, n), dtype=output_dtype, device=A.device)
    if m == 0 or n == 0:
        return out

    # GCU400/gcu410 tile dispatch, swept over the benchmark shape set
    # (kernel-mode do_bench median, 2026-11-08): the per-k-block fp32
    # epilogue (cvt + fma on the (BM, BN) tile) is the throughput floor, so
    # wider BM amortizes it — BM=128/num_warps=4 takes m>=512 shapes to
    # 0.66-0.75x, but only while the flattened grid still fills the 24 SIPs
    # (BM=128 below ~32 programs regresses to underfill, e.g. m=256 n=1024:
    # 1.35x); every shape outside the arm keeps the production
    # tiles/num_warps bit-for-bit (BM=64 stays nw=2: nw4 there regresses
    # (64,4096) by +9% in the official bench).
    tile_n = _pow2_at_most(block_n, 128)
    grid_n = (n + tile_n - 1) // tile_n
    if tile_n == 128 and m >= 128 and ((m + 127) // 128) * grid_n >= 32:
        block_m = 128
    else:
        block_m = _pow2_at_most(m, 64)
    grid_m = (m + block_m - 1) // block_m
    tile_k = _pow2_at_most(block_k, 128)
    nk = (k + tile_k - 1) // tile_k

    # BM=64 arm only: unroll the k loop SUBK scale blocks per iteration for
    # load ILP (GCU400, kernel-mode do_bench median, 5-round paired sweep,
    # 2026-11-13): SUBK=8 -20.6~-22.2% (m=64/256, K=4096), SUBK=2 -12.8% at
    # NK=4; SUBK=16 regresses (+15~47%), and the BM=16 / BM=128 arms are
    # neutral-to-worse with the unroll -> gated to this arm.  Requires the
    # scale block to equal the k tile (block_k <= 128) and NK % SUBK == 0;
    # everything else keeps the plain kernel bit-for-bit.
    subk = 0
    if block_m == 64 and tile_n == 128 and block_k == tile_k:
        if nk % 8 == 0:
            subk = 8
        elif nk % 2 == 0:
            subk = 2

    kwargs = dict(
        M=m,
        N=n,
        K=k,
        AM=A.stride(0),
        AK=A.stride(1),
        BN=B.stride(0),
        BK=B.stride(1),
        ASM=As.stride(0),
        ASK=As.stride(1),
        BSN=Bs.stride(0),
        BSK=Bs.stride(1),
        CM=out.stride(0),
        CN=out.stride(1),
        NK=nk,
        K_PER_SCALE=block_k // tile_k,
        N_PER_SCALE=block_n // tile_n,
        GRID_N=grid_n,
        EVEN_M=(m % block_m == 0),
        EVEN_N=(n % tile_n == 0),
        EVEN_K=(k % tile_k == 0),
        BLOCK_M=block_m,
        BLOCK_N=tile_n,
        BLOCK_K=tile_k,
    )
    if subk:
        kwargs["SUBK"] = subk
        _w8a8_bk_matmul_unroll[(grid_m * grid_n,)](
            A,
            B,
            As,
            Bs,
            out,
            num_warps=4 if block_m == 128 else 2,
            num_stages=2,
            **kwargs,
        )
    else:
        _w8a8_bk_matmul[(grid_m * grid_n,)](
            A,
            B,
            As,
            Bs,
            out,
            num_warps=4 if block_m == 128 else 2,
            num_stages=2,
            **kwargs,
        )
    return out


__all__ = ["w8a8_block_int8_matmul"]
