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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_f16(raw):
    """Bit-exact E4M3FN -> FP16 decode (all 256 codes, subnormals included).

    Exponent field 0xF (inf/nan magnitudes) passes through the rebias intact.
    """
    v = raw.to(tl.int16)
    e = (v >> 3) & 0xF
    m = v & 0x7
    mag_bits = ((e + 8) << 10) | (m << 7)
    norm = mag_bits.to(tl.float16, bitcast=True)
    sub = m.to(tl.float16) * 0.001953125  # 2**-9
    mag = tl.where(e == 0, sub, norm)
    return tl.where(v >= 128, -mag, mag)


@triton.jit
def _split_f16(prod):
    """Split an fp32 value into an fp16 hi/lo pair (prod ~= hi + lo)."""
    hi = prod.to(tl.float16)
    lo = (prod - hi.to(tl.float32)).to(tl.float16)
    return hi, lo


@triton.jit
def _decode_ab_hilo_kernel(
    a_ptr,  # uint8 [M, K]  (raw e4m3 bytes)
    b_ptr,  # uint8 [N, K]  (raw e4m3 bytes)
    as_ptr,  # fp32 [M, K // block_k]
    bs_ptr,  # fp32 [N // block_n, K // block_k]
    ahi_ptr,  # fp16 [M, K]
    alo_ptr,  # fp16 [M, K]
    bhi_ptr,  # fp16 [K, N]  (transposed)
    blo_ptr,  # fp16 [K, N]  (transposed)
    M,
    N,
    K,
    n_k,  # K // block_k
    BLOCK_N_SCALE,  # block_n
    A_TILES: tl.constexpr,  # number of A row-tiles on axis 0
    HAS_MASK_A: tl.constexpr,
    HAS_MASK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One-launch decode of both operands, with ``As``/``Bs`` folded."""
    pid_r = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    if pid_r < A_TILES:
        offs_m = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        m_ok = offs_m < M
        ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
        if HAS_MASK_A:
            raw = tl.load(ptrs, mask=m_ok[:, None], other=0)
            s = tl.load(as_ptr + offs_m * n_k + pid_k, mask=m_ok, other=0.0)
        else:
            raw = tl.load(ptrs)
            s = tl.load(as_ptr + offs_m * n_k + pid_k)
        prod = _e4m3_to_f16(raw).to(tl.float32) * s[:, None]
        hi, lo = _split_f16(prod)
        if HAS_MASK_A:
            tl.store(
                ahi_ptr + offs_m[:, None] * K + offs_k[None, :],
                hi,
                mask=m_ok[:, None],
            )
            tl.store(
                alo_ptr + offs_m[:, None] * K + offs_k[None, :],
                lo,
                mask=m_ok[:, None],
            )
        else:
            tl.store(ahi_ptr + offs_m[:, None] * K + offs_k[None, :], hi)
            tl.store(alo_ptr + offs_m[:, None] * K + offs_k[None, :], lo)
    else:
        offs_n = (pid_r - A_TILES) * BLOCK_R + tl.arange(0, BLOCK_R)
        n_ok = offs_n < N
        ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]
        if HAS_MASK_B:
            raw = tl.load(ptrs, mask=n_ok[:, None], other=0)
            s = tl.load(
                bs_ptr + (offs_n // BLOCK_N_SCALE) * n_k + pid_k,
                mask=n_ok,
                other=0.0,
            )
        else:
            raw = tl.load(ptrs)
            s = tl.load(bs_ptr + (offs_n // BLOCK_N_SCALE) * n_k + pid_k)
        prod = _e4m3_to_f16(raw).to(tl.float32) * s[:, None]
        hi, lo = _split_f16(prod)
        if HAS_MASK_B:
            tl.store(
                bhi_ptr + offs_k[:, None] * N + offs_n[None, :],
                tl.trans(hi),
                mask=n_ok[None, :],
            )
            tl.store(
                blo_ptr + offs_k[:, None] * N + offs_n[None, :],
                tl.trans(lo),
                mask=n_ok[None, :],
            )
        else:
            tl.store(
                bhi_ptr + offs_k[:, None] * N + offs_n[None, :], tl.trans(hi)
            )
            tl.store(
                blo_ptr + offs_k[:, None] * N + offs_n[None, :], tl.trans(lo)
            )


@triton.jit
def _gemm_hilo_kernel(
    a_hi,
    a_lo,  # fp16 [M, K] contiguous (A * As, hi + lo)
    b_hi,
    b_lo,  # fp16 [K, N] contiguous (B * Bs, hi + lo, transposed)
    c_ptr,  # output_dtype [M, N]
    M,
    N,
    K,
    stride_cm,
    n_k,  # K // block_k  (== number of GEMM k iterations)
    HAS_MASK_M: tl.constexpr,
    HAS_MASK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """3-dot hi/lo GEMM with fp32 accumulation via chained ``tl.dot``."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M > 1:
        group_size = GROUP_M * num_n
        gid = pid // group_size
        first_m = gid * GROUP_M
        gsize = tl.minimum(num_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % group_size) % gsize)
        pid_n = (pid % group_size) // gsize
    else:
        pid_m = pid // num_n
        pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    msk_m = offs_m < M
    msk_n = offs_n < N

    a_base = a_hi + offs_m[:, None] * K + offs_k[None, :]
    alo_base = a_lo + offs_m[:, None] * K + offs_k[None, :]
    b_base = b_hi + offs_k[:, None] * N + offs_n[None, :]
    blo_base = b_lo + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, n_k):
        koff = k * BLOCK_K
        if HAS_MASK_M:
            ah = tl.load(a_base + koff, mask=msk_m[:, None], other=0.0)
            al = tl.load(alo_base + koff, mask=msk_m[:, None], other=0.0)
        else:
            ah = tl.load(a_base + koff)
            al = tl.load(alo_base + koff)
        if HAS_MASK_N:
            bh = tl.load(b_base + koff * N, mask=msk_n[None, :], other=0.0)
            bl = tl.load(blo_base + koff * N, mask=msk_n[None, :], other=0.0)
        else:
            bh = tl.load(b_base + koff * N)
            bl = tl.load(blo_base + koff * N)
        acc = tl.dot(ah, bh, acc)
        acc = tl.dot(al, bh, acc)
        acc = tl.dot(ah, bl, acc)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :],
        acc.to(c_ptr.dtype.element_ty),
        mask=msk_m[:, None] & msk_n[None, :],
    )


@triton.jit
def _fused_hilo_kernel(
    a_ptr,  # uint8 [M, K]  (raw e4m3 bytes)
    b_ptr,  # uint8 [N, K]  (raw e4m3 bytes)
    c_ptr,  # output_dtype [M, N]
    as_ptr,  # fp32 [M, K // block_k]
    bs_ptr,  # fp32 [N // block_n, K // block_k]
    M,
    N,
    K,
    stride_cm,
    n_k,  # K // block_k  (== number of GEMM k iterations)
    BLOCK_N_SCALE,  # block_n
    HAS_MASK_M: tl.constexpr,
    HAS_MASK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fully fused decode + scale fold + hi/lo GEMM for small M (one."""
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    msk_m = offs_m < M
    msk_n = offs_n < N
    n_blk = pid_n * BLOCK_N // BLOCK_N_SCALE

    a_base = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_base = b_ptr + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, n_k):
        koff = k * BLOCK_K
        if HAS_MASK_M:
            a_raw = tl.load(a_base + koff, mask=msk_m[:, None], other=0)
            sa = tl.load(as_ptr + offs_m * n_k + k, mask=msk_m, other=0.0)
        else:
            a_raw = tl.load(a_base + koff)
            sa = tl.load(as_ptr + offs_m * n_k + k)
        apr = _e4m3_to_f16(a_raw).to(tl.float32) * sa[:, None]
        ah, al = _split_f16(apr)
        if HAS_MASK_N:
            b_raw = tl.load(b_base + koff, mask=msk_n[:, None], other=0)
            sb = tl.load(bs_ptr + n_blk * n_k + k)
        else:
            b_raw = tl.load(b_base + koff)
            sb = tl.load(bs_ptr + n_blk * n_k + k)
        bpr = _e4m3_to_f16(b_raw).to(tl.float32) * sb
        bh, bl = _split_f16(bpr)
        bt = tl.trans(bh)
        acc = tl.dot(ah, bt, acc)
        acc = tl.dot(al, bt, acc)
        acc = tl.dot(ah, tl.trans(bl), acc)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :],
        acc.to(c_ptr.dtype.element_ty),
        mask=msk_m[:, None] & msk_n[None, :],
    )


_DECODE_BR = 64
_DECODE_WARPS = 4

# Small-M path: fully fused, one launch (launch overhead dominates here).
_FUSED_MAX_M = 8
_FUSED_BM = 16
_FUSED_BN = 128
_FUSED_WARPS = 4

_NAT_MAX_M = 8
_NAT_MIN_N = 4096
_NAT_DECODE_BR = 64
_NAT_DECODE_WARPS = 4
_NAT_BM = 16
_NAT_BN = 256
_NAT_WARPS = 8


@triton.jit
def _decode_ab_nat_kernel(
    a_ptr,  # uint8 [M, K]  (raw e4m3 bytes)
    b_ptr,  # uint8 [N, K]  (raw e4m3 bytes)
    a16_ptr,  # fp16 [M, K]  (exact e4m3 values, no scale fold)
    b16_ptr,  # fp16 [N, K]  (natural layout, no transpose)
    M,
    N,
    K,
    A_TILES: tl.constexpr,
    MASK_A: tl.constexpr,
    MASK_B: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One-launch decode of both operands into plain fp16 (no scale."""
    pid_r = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    if pid_r < A_TILES:
        offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        ok = offs_r < M
        ptrs = a_ptr + offs_r[:, None] * K + offs_k[None, :]
        if MASK_A:
            raw = tl.load(ptrs, mask=ok[:, None], other=0)
        else:
            raw = tl.load(ptrs)
        val = _e4m3_to_f16(raw)
        dst = a16_ptr + offs_r[:, None] * K + offs_k[None, :]
        if MASK_A:
            tl.store(dst, val, mask=ok[:, None])
        else:
            tl.store(dst, val)
    else:
        offs_r = (pid_r - A_TILES) * BLOCK_R + tl.arange(0, BLOCK_R)
        ok = offs_r < N
        ptrs = b_ptr + offs_r[:, None] * K + offs_k[None, :]
        if MASK_B:
            raw = tl.load(ptrs, mask=ok[:, None], other=0)
        else:
            raw = tl.load(ptrs)
        val = _e4m3_to_f16(raw)
        dst = b16_ptr + offs_r[:, None] * K + offs_k[None, :]
        if MASK_B:
            tl.store(dst, val, mask=ok[:, None])
        else:
            tl.store(dst, val)


@triton.jit
def _gemm_nat_fold_kernel(
    a16_ptr,  # fp16 [M, K]  (exact e4m3 values)
    b16_ptr,  # fp16 [N, K]  (natural layout)
    c_ptr,  # output_dtype [M, N]
    as_ptr,  # fp32 [M, K // block_k]
    bs_ptr,  # fp32 [N // block_n, K // block_k]
    M,
    N,
    K,
    stride_cm,
    n_k,  # K // block_k  (== number of GEMM k iterations)
    BLOCK_N_SCALE,  # block_n
    MASK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """1-dot GEMM on exact-fp16 operands with exact post-dot scale."""
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    msk_m = offs_m < M
    msk_n = offs_n < N

    a_base = a16_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_base = b16_ptr + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, n_k):
        koff = k * BLOCK_K
        a = tl.load(a_base + koff)
        if MASK_N:
            b = tl.load(b_base + koff, mask=msk_n[:, None], other=0.0)
            sb = tl.load(
                bs_ptr + (offs_n // BLOCK_N_SCALE) * n_k + k,
                mask=msk_n,
                other=0.0,
            )
        else:
            b = tl.load(b_base + koff)
            sb = tl.load(bs_ptr + (offs_n // BLOCK_N_SCALE) * n_k + k)
        sa = tl.load(as_ptr + offs_m * n_k + k)
        d = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
        acc += d * (sa[:, None] * sb)

    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :],
        acc.to(c_ptr.dtype.element_ty),
        mask=msk_m[:, None] & msk_n[None, :],
    )


def _matmul_small_m_wide_n(A, B, As, Bs, out, M, N, K, block_n, block_k, n_k):
    """v4 path for ``M <= 8 and N >= 4096``: decode (1 launch) + 1-dot GEMM."""
    a16 = torch.empty((M, K), device=A.device, dtype=torch.float16)
    b16 = torch.empty((N, K), device=B.device, dtype=torch.float16)
    a_tiles = triton.cdiv(M, _NAT_DECODE_BR)
    b_tiles = triton.cdiv(N, _NAT_DECODE_BR)
    _decode_ab_nat_kernel[(a_tiles + b_tiles, triton.cdiv(K, block_k))](
        A.view(torch.uint8),
        B.view(torch.uint8),
        a16,
        b16,
        M,
        N,
        K,
        A_TILES=a_tiles,
        MASK_A=(M % _NAT_DECODE_BR) != 0,
        MASK_B=(N % _NAT_DECODE_BR) != 0,
        BLOCK_R=_NAT_DECODE_BR,
        BLOCK_K=block_k,
        num_warps=_NAT_DECODE_WARPS,
    )
    grid = (triton.cdiv(M, _NAT_BM) * triton.cdiv(N, _NAT_BN),)
    _gemm_nat_fold_kernel[grid](
        a16,
        b16,
        out,
        As,
        Bs,
        M,
        N,
        K,
        N,  # stride_cm (out contiguous)
        n_k,
        block_n,
        MASK_N=(N % _NAT_BN) != 0,
        BLOCK_M=_NAT_BM,
        BLOCK_N=_NAT_BN,
        BLOCK_K=block_k,
        num_warps=_NAT_WARPS,
        num_stages=2,
    )


def w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype):
    """Block-wise fp8 GEMM: ``(A_deq @ B_deq.T).to(output_dtype)``."""
    block_n, block_k = block_size
    M, K = A.shape
    N = B.shape[0]

    A = A.contiguous()
    B = B.contiguous()
    As = As.contiguous()
    Bs = Bs.contiguous()

    n_k = K // block_k

    if M <= _NAT_MAX_M and N >= _NAT_MIN_N:
        out = torch.empty((M, N), device=A.device, dtype=output_dtype)
        _matmul_small_m_wide_n(
            A, B, As, Bs, out, M, N, K, block_n, block_k, n_k
        )
        return out

    if M <= _FUSED_MAX_M:
        # One fused launch: decode + scale fold + hi/lo GEMM.
        out = torch.empty((M, N), device=A.device, dtype=output_dtype)
        grid = (triton.cdiv(M, _FUSED_BM) * triton.cdiv(N, _FUSED_BN),)
        _fused_hilo_kernel[grid](
            A.view(torch.uint8),
            B.view(torch.uint8),
            out,
            As,
            Bs,
            M,
            N,
            K,
            N,  # stride_cm (out contiguous)
            n_k,
            block_n,
            HAS_MASK_M=(M % _FUSED_BM) != 0,
            HAS_MASK_N=(N % _FUSED_BN) != 0,
            BLOCK_M=_FUSED_BM,
            BLOCK_N=_FUSED_BN,
            BLOCK_K=block_k,
            num_warps=_FUSED_WARPS,
            num_stages=1,
        )
        return out

    # Single-launch streaming decode of both operands into fp16 hi/lo
    # pairs (scales folded, B transposed to [K, N]).
    a_hi = torch.empty((M, K), device=A.device, dtype=torch.float16)
    a_lo = torch.empty((M, K), device=A.device, dtype=torch.float16)
    b_hi = torch.empty((K, N), device=B.device, dtype=torch.float16)
    b_lo = torch.empty((K, N), device=B.device, dtype=torch.float16)

    a_tiles = triton.cdiv(M, _DECODE_BR)
    b_tiles = triton.cdiv(N, _DECODE_BR)
    _decode_ab_hilo_kernel[(a_tiles + b_tiles, triton.cdiv(K, block_k))](
        A.view(torch.uint8),
        B.view(torch.uint8),
        As,
        Bs,
        a_hi,
        a_lo,
        b_hi,
        b_lo,
        M,
        N,
        K,
        n_k,
        block_n,
        A_TILES=a_tiles,
        HAS_MASK_A=(M % _DECODE_BR) != 0,
        HAS_MASK_B=(N % _DECODE_BR) != 0,
        BLOCK_R=_DECODE_BR,
        BLOCK_K=block_k,
        num_warps=_DECODE_WARPS,
    )

    # Pure 3-dot GEMM.  Small M uses a short tile (more CTAs in flight);
    # large M uses a [256, 128] tile with a GROUP_M swizzle for L2 reuse.
    out = torch.empty((M, N), device=A.device, dtype=output_dtype)
    if M <= 64:
        block_m, block_n_tile, group_m, warps = 64, 256, 1, 8
    else:
        block_m, block_n_tile, group_m, warps = 256, 128, 8, 8
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n_tile),)
    _gemm_hilo_kernel[grid](
        a_hi,
        a_lo,
        b_hi,
        b_lo,
        out,
        M,
        N,
        K,
        N,  # stride_cm (out contiguous)
        n_k,
        HAS_MASK_M=(M % block_m) != 0,
        HAS_MASK_N=(N % block_n_tile) != 0,
        GROUP_M=group_m,
        BLOCK_M=block_m,
        BLOCK_N=block_n_tile,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=2,
    )
    return out


__all__ = ["w8a8_block_fp8_matmul"]
