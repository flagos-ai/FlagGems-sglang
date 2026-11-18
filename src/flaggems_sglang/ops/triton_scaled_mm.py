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
def _e4m3_to_fp16(b32):
    lo = b32 & 0x7F
    e = lo >> 3
    den = ((lo << 7).to(tl.uint16)).to(tl.float16, bitcast=True) * tl.full(
        (1,), 256.0, tl.float16
    )
    nrm = (((lo << 7) + 0x2000).to(tl.uint16)).to(tl.float16, bitcast=True)
    y = tl.where(e == 0, den, nrm)
    isnan = lo == 0x7F
    y = tl.where(isnan, float("nan"), y.to(tl.float32)).to(tl.float16)
    return tl.where(b32 >= 0x80, -y, y)


@triton.jit
def _unpack4(w):
    b0 = _e4m3_to_fp16(w & 0xFF)
    b1 = _e4m3_to_fp16((w >> 8) & 0xFF)
    b2 = _e4m3_to_fp16((w >> 16) & 0xFF)
    b3 = _e4m3_to_fp16((w >> 24) & 0xFF)
    return tl.interleave(tl.interleave(b0, b2), tl.interleave(b1, b3))


@triton.jit
def _gemm_fp16a_packedb_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    scale_a_ptr,
    scale_b_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk4,
    stride_cm,
    stride_cn,
    stride_bia,
    stride_bib,
    OUT_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SA_BCAST: tl.constexpr,
    SB_BCAST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn4 = pid_n * (BLOCK_N // 4) + tl.arange(0, BLOCK_N // 4)
    rk = tl.arange(0, BLOCK_K)
    m_mask = rm < M

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk4 + rn4[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a16 = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        bw = tl.load(b_ptrs)
        b16 = _unpack4(bw)
        acc = tl.dot(a16, b16, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk4

    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N
    if SA_BCAST:
        scale_a = tl.load(scale_a_ptr).to(tl.float32)
    else:
        scale_a = tl.load(scale_a_ptr + rm, mask=m_mask, other=1.0).to(
            tl.float32
        )
    if SB_BCAST:
        scale_b = tl.load(scale_b_ptr).to(tl.float32)
    else:
        scale_b = tl.load(scale_b_ptr + rn, mask=n_mask, other=1.0).to(
            tl.float32
        )
    acc = acc * scale_a[:, None] * scale_b[None, :]

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + rm[:, None] * stride_bia + rn[None, :] * stride_bib,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc = acc + bias

    tl.store(
        c_ptr + rm[:, None] * stride_cm + rn[None, :],
        acc.to(OUT_DTYPE),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _widen_fp8_vec_kernel(x_ptr, y_ptr, numel4, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel4
    w = tl.load(x_ptr + offs, mask=mask, other=0)
    b0 = w & 0xFF
    b1 = (w >> 8) & 0xFF
    b2 = (w >> 16) & 0xFF
    b3 = (w >> 24) & 0xFF
    y0 = _e4m3_to_fp16(b0)
    y1 = _e4m3_to_fp16(b1)
    y2 = _e4m3_to_fp16(b2)
    y3 = _e4m3_to_fp16(b3)
    y02 = tl.interleave(y0, y2)
    y13 = tl.interleave(y1, y3)
    y = tl.interleave(y02, y13)
    tl.store(
        y_ptr + (offs[:, None] * 4 + tl.arange(0, 4)[None, :]),
        tl.reshape(y, (BLOCK, 4)),
        mask=mask[:, None],
    )


@triton.jit
def _widen_fp8_kernel(
    x_ptr, y_ptr, numel, BLOCK: tl.constexpr, IS_FP8: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    if IS_FP8:
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, x.to(tl.float16), mask=mask)
    else:
        c = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.uint32)
        tl.store(y_ptr + offs, _e4m3_to_fp16(c), mask=mask)


@triton.jit
def _gemv_packed_kernel(
    a_ptr,
    b_ptr,
    part_ptr,
    N,
    K,
    KSPLIT,
    stride_ak,
    stride_bk4,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    ks = pid % KSPLIT
    pnr = pid // KSPLIT
    rn4 = pnr * (BN // 4) + tl.arange(0, BN // 4)
    rk = tl.arange(0, BK)
    kchunk = K // KSPLIT
    a_ptrs = a_ptr + (ks * kchunk + rk) * stride_ak
    b_ptrs = b_ptr + (ks * kchunk + rk)[:, None] * stride_bk4 + rn4[None, :]
    acc0 = tl.zeros((BN // 4,), dtype=tl.float32)
    acc1 = tl.zeros((BN // 4,), dtype=tl.float32)
    acc2 = tl.zeros((BN // 4,), dtype=tl.float32)
    acc3 = tl.zeros((BN // 4,), dtype=tl.float32)
    for _ in range(0, kchunk // BK):
        au = tl.load(a_ptrs).to(tl.int32)
        av = _e4m3_to_fp16(au).to(tl.float32)
        w = tl.load(b_ptrs)
        b0 = _e4m3_to_fp16(w & 0xFF).to(tl.float32)
        b1 = _e4m3_to_fp16((w >> 8) & 0xFF).to(tl.float32)
        b2 = _e4m3_to_fp16((w >> 16) & 0xFF).to(tl.float32)
        b3 = _e4m3_to_fp16((w >> 24) & 0xFF).to(tl.float32)
        acc0 += tl.sum(av[:, None] * b0, 0)
        acc1 += tl.sum(av[:, None] * b1, 0)
        acc2 += tl.sum(av[:, None] * b2, 0)
        acc3 += tl.sum(av[:, None] * b3, 0)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk4
    part_ptrs = part_ptr + ks * N + rn4 * 4
    tl.store(part_ptrs + 0, acc0)
    tl.store(part_ptrs + 1, acc1)
    tl.store(part_ptrs + 2, acc2)
    tl.store(part_ptrs + 3, acc3)


@triton.jit
def _gemv_reduce_kernel(
    part_ptr,
    scale_a_ptr,
    scale_b_ptr,
    bias_ptr,
    c_ptr,
    N,
    KSPLIT,
    stride_bib,
    OUT_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SB_BCAST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BLOCK + tl.arange(0, BLOCK)
    n_mask = rn < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(0, KSPLIT):
        acc += tl.load(part_ptr + s * N + rn, mask=n_mask, other=0.0)
    sa = tl.load(scale_a_ptr).to(tl.float32)
    if SB_BCAST:
        sb = tl.load(scale_b_ptr).to(tl.float32)
    else:
        sb = tl.load(scale_b_ptr + rn, mask=n_mask, other=1.0).to(tl.float32)
    acc = acc * sa * sb
    if HAS_BIAS:
        acc = acc + tl.load(
            bias_ptr + rn * stride_bib, mask=n_mask, other=0.0
        ).to(tl.float32)
    tl.store(c_ptr + rn, acc.to(OUT_DTYPE), mask=n_mask)


@triton.jit
def _gemv_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    scale_a_ptr,
    scale_b_ptr,
    N,
    K,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cn,
    stride_bib,
    OUT_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SB_BCAST: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    DECODE_FP8: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    a_ptrs = a_ptr + rk * stride_ak
    acc = tl.zeros((BN,), dtype=tl.float32)
    if EVEN_N:
        if EVEN_K:
            for _ in range(0, tl.cdiv(K, BK)):
                if not DECODE_FP8:
                    a = tl.load(a_ptrs).to(tl.float32)
                    b = tl.load(b_ptrs).to(tl.float32)
                else:
                    a = _e4m3_to_fp16(tl.load(a_ptrs).to(tl.uint32)).to(
                        tl.float32
                    )
                    b = _e4m3_to_fp16(tl.load(b_ptrs).to(tl.uint32)).to(
                        tl.float32
                    )
                acc += tl.sum(a[:, None] * b, 0)
                a_ptrs += BK * stride_ak
                b_ptrs += BK * stride_bk
        else:
            for k in range(0, tl.cdiv(K, BK)):
                k_mask = k * BK + rk < K
                if not DECODE_FP8:
                    a = tl.load(a_ptrs, mask=k_mask, other=0.0).to(tl.float32)
                    b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0).to(
                        tl.float32
                    )
                else:
                    a = _e4m3_to_fp16(
                        tl.load(a_ptrs, mask=k_mask, other=0).to(tl.uint32)
                    ).to(tl.float32)
                    b = _e4m3_to_fp16(
                        tl.load(b_ptrs, mask=k_mask[:, None], other=0).to(
                            tl.uint32
                        )
                    ).to(tl.float32)
                acc += tl.sum(a[:, None] * b, 0)
                a_ptrs += BK * stride_ak
                b_ptrs += BK * stride_bk
        sa = tl.load(scale_a_ptr).to(tl.float32)
        sb = (
            tl.load(scale_b_ptr).to(tl.float32)
            if SB_BCAST
            else tl.load(scale_b_ptr + rn).to(tl.float32)
        )
        acc = acc * sa * sb
        if HAS_BIAS:
            acc = acc + tl.load(bias_ptr + rn * stride_bib).to(tl.float32)
        tl.store(c_ptr + rn * stride_cn, acc.to(OUT_DTYPE))
    else:
        n_mask = rn < N
        for k in range(0, tl.cdiv(K, BK)):
            k_mask = k * BK + rk < K
            m_b = k_mask[:, None] & n_mask[None, :]
            if not DECODE_FP8:
                a = tl.load(a_ptrs, mask=k_mask, other=0.0).to(tl.float32)
                b = tl.load(b_ptrs, mask=m_b, other=0.0).to(tl.float32)
            else:
                a = _e4m3_to_fp16(
                    tl.load(a_ptrs, mask=k_mask, other=0).to(tl.uint32)
                ).to(tl.float32)
                b = _e4m3_to_fp16(
                    tl.load(b_ptrs, mask=m_b, other=0).to(tl.uint32)
                ).to(tl.float32)
            acc += tl.sum(a[:, None] * b, 0)
            a_ptrs += BK * stride_ak
            b_ptrs += BK * stride_bk

        sa = tl.load(scale_a_ptr).to(tl.float32)
        if SB_BCAST:
            sb = tl.load(scale_b_ptr).to(tl.float32)
        else:
            sb = tl.load(scale_b_ptr + rn, mask=n_mask, other=1.0).to(
                tl.float32
            )
        acc = acc * sa * sb

        if HAS_BIAS:
            bias = tl.load(
                bias_ptr + rn * stride_bib, mask=n_mask, other=0.0
            ).to(tl.float32)
            acc = acc + bias

        tl.store(c_ptr + rn * stride_cn, acc.to(OUT_DTYPE), mask=n_mask)


@triton.jit
def _dot_i32(a, b, acc):
    return tl.dot(a, b, acc, out_dtype=tl.int32)


@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    scale_a_ptr,
    scale_b_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bia,
    stride_bib,
    OUT_DTYPE: tl.constexpr,
    IS_INT8: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SA_BCAST: tl.constexpr,
    SB_BCAST: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    m_mask = rm < M
    n_mask = rn < N

    if IS_INT8:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K & EVEN_M & EVEN_N:
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            if IS_INT8:
                acc = _dot_i32(a, b, acc)
            else:
                acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    elif EVEN_K:
        if EVEN_N:
            for _ in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs, mask=m_mask[:, None], other=0)
                b = tl.load(b_ptrs)
                if IS_INT8:
                    acc = _dot_i32(a, b, acc)
                else:
                    acc = tl.dot(a, b, acc)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for _ in range(0, tl.cdiv(K, BLOCK_K)):
                a = tl.load(a_ptrs, mask=m_mask[:, None], other=0)
                b = tl.load(b_ptrs, mask=n_mask[None, :], other=0)
                if IS_INT8:
                    acc = _dot_i32(a, b, acc)
                else:
                    acc = tl.dot(a, b, acc)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = k * BLOCK_K + rk < K
            a = tl.load(
                a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0
            )
            b = tl.load(
                b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0
            )
            if IS_INT8:
                acc = _dot_i32(a, b, acc)
            else:
                acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    accf = acc.to(tl.float32)

    if SA_BCAST:
        scale_a = tl.load(scale_a_ptr).to(tl.float32)
    else:
        scale_a = tl.load(scale_a_ptr + rm, mask=m_mask, other=1.0).to(
            tl.float32
        )
    if SB_BCAST:
        scale_b = tl.load(scale_b_ptr).to(tl.float32)
    else:
        scale_b = tl.load(scale_b_ptr + rn, mask=n_mask, other=1.0).to(
            tl.float32
        )
    accf = accf * scale_a[:, None] * scale_b[None, :]

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + rm[:, None] * stride_bia + rn[None, :] * stride_bib,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        accf = accf + bias

    tl.store(
        c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn,
        accf.to(OUT_DTYPE),
        mask=m_mask[:, None] & n_mask[None, :],
    )


def _tl_dtype(out_dtype):
    if out_dtype == torch.float16:
        return tl.float16
    if out_dtype == torch.bfloat16:
        return tl.bfloat16
    return tl.float32


def triton_scaled_mm(input, weight, scale_a, scale_b, out_dtype, bias=None):
    assert input.ndim == 2 and weight.ndim == 2
    M, K = input.shape
    K2, N = weight.shape
    assert K == K2

    is_int8 = input.dtype == torch.int8
    assert is_int8 or input.dtype == torch.float8_e4m3fn

    a = input.contiguous()
    b = weight.contiguous()
    scale_a = scale_a.contiguous()
    scale_b = scale_b.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    out = torch.empty((M, N), device=input.device, dtype=out_dtype)
    out_tl = _tl_dtype(out_dtype)

    sa_bcast = 1 if scale_a.numel() == 1 else 0
    sb_bcast = 1 if scale_b.numel() == 1 else 0
    has_bias = bias is not None
    bias_arg = bias if has_bias else out
    bia = bias.stride(0) if has_bias else 0
    bib = bias.stride(1) if has_bias else 0

    if M == 1:
        if not is_int8 and K % 256 == 0 and N % 64 == 0:
            if K % 1024 == 0:
                ksplit = 4
            elif K % 512 == 0:
                ksplit = 2
            else:
                ksplit = 1
            BN, BK, nw = 64, 256, 4
            part = torch.empty(
                (ksplit, N), device=input.device, dtype=torch.float32
            )
            grid = (ksplit * triton.cdiv(N, BN),)
            a_u8 = a.view(torch.uint8)
            b_u32 = b.view(torch.uint8).view(torch.uint32)
            _gemv_packed_kernel[grid](
                a_u8,
                b_u32,
                part,
                N,
                K,
                ksplit,
                a_u8.stride(1),
                b_u32.stride(0),
                BN=BN,
                BK=BK,
                num_warps=nw,
            )
            _gemv_reduce_kernel[(triton.cdiv(N, 1024),)](
                part,
                scale_a,
                scale_b,
                bias_arg,
                out,
                N,
                ksplit,
                bib,
                OUT_DTYPE=out_tl,
                HAS_BIAS=has_bias,
                SB_BCAST=sb_bcast,
                BLOCK=1024,
                num_warps=4,
            )
            return out

        BN, BK, nw, ns = 64, 256, 8, 2
        grid = (triton.cdiv(N, BN),)
        _gemv_kernel[grid](
            a if is_int8 else a.view(torch.uint8),
            b if is_int8 else b.view(torch.uint8),
            bias_arg,
            out,
            scale_a,
            scale_b,
            N,
            K,
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(1),
            bib,
            OUT_DTYPE=out_tl,
            HAS_BIAS=has_bias,
            SB_BCAST=sb_bcast,
            EVEN_K=(K % BK == 0),
            EVEN_N=(N % BN == 0),
            DECODE_FP8=(not is_int8),
            BN=BN,
            BK=BK,
            num_warps=nw,
            num_stages=ns,
        )
        return out

    if not is_int8:
        if 1 < M <= 64 and K % 4 == 0:
            if M <= 16:
                if N <= 4096:
                    BM, BN, BK, GM, nw = 16, 64, 128, 32, 8
                else:
                    BM, BN, BK, GM, nw = 16, 64, 64, 16, 4
            else:
                BM, BN, BK, GM, nw = 64, 64, 64, 32, 8
            if N % BN == 0 and K % BK == 0:
                a16 = torch.empty(
                    (M, K), device=input.device, dtype=torch.float16
                )
                _widen_fp8_vec_kernel[(triton.cdiv((M * K) // 4, 1024),)](
                    a.view(torch.uint8).view(torch.uint32),
                    a16,
                    (M * K) // 4,
                    BLOCK=1024,
                    num_warps=4,
                )
                grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
                b_u32 = b.view(torch.uint8).view(torch.uint32)
                _gemm_fp16a_packedb_kernel[grid](
                    a16,
                    b_u32,
                    bias_arg,
                    out,
                    scale_a,
                    scale_b,
                    M,
                    N,
                    K,
                    a16.stride(0),
                    a16.stride(1),
                    b_u32.stride(0),
                    out.stride(0),
                    out.stride(1),
                    bia,
                    bib,
                    OUT_DTYPE=out_tl,
                    HAS_BIAS=has_bias,
                    SA_BCAST=sa_bcast,
                    SB_BCAST=sb_bcast,
                    BLOCK_M=BM,
                    BLOCK_N=BN,
                    BLOCK_K=BK,
                    GROUP_M=GM,
                    num_warps=nw,
                    num_stages=1,
                )
                return out

        a16 = torch.empty((M, K), device=input.device, dtype=torch.float16)
        b16 = torch.empty((K, N), device=input.device, dtype=torch.float16)
        if K % 4 == 0:
            na4, nb4 = (M * K) // 4, (K * N) // 4
            _widen_fp8_vec_kernel[(triton.cdiv(na4, 1024),)](
                a.view(torch.uint8).view(torch.uint32),
                a16,
                na4,
                BLOCK=1024,
                num_warps=4,
            )
            _widen_fp8_vec_kernel[(triton.cdiv(nb4, 1024),)](
                b.view(torch.uint8).view(torch.uint32),
                b16,
                nb4,
                BLOCK=1024,
                num_warps=4,
            )
        else:
            na, nb = a.numel(), b.numel()
            a8 = a.view(torch.uint8)
            b8 = b.view(torch.uint8)
            _widen_fp8_kernel[(triton.cdiv(na, 2048),)](
                a8, a16, na, BLOCK=2048, IS_FP8=False, num_warps=4
            )
            _widen_fp8_kernel[(triton.cdiv(nb, 2048),)](
                b8, b16, nb, BLOCK=2048, IS_FP8=False, num_warps=4
            )
        a, b = a16, b16

        if M <= 128:
            BM, BN, BK, GM, nw = 64, 64, 64, 16, 8
        else:
            # BK=32 with num_warps=8 miscompiles on Hopper (sm_90) and reads
            # out of bounds; BK=64/num_warps=4 is correct and also faster here.
            BM, BN, BK, GM, nw = 128, 128, 64, 8, 4
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        _gemm_kernel[grid](
            a,
            b,
            bias_arg,
            out,
            scale_a,
            scale_b,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            bia,
            bib,
            OUT_DTYPE=out_tl,
            IS_INT8=False,
            HAS_BIAS=has_bias,
            SA_BCAST=sa_bcast,
            SB_BCAST=sb_bcast,
            EVEN_K=(K % BK == 0),
            EVEN_M=(M % BM == 0),
            EVEN_N=(N % BN == 0),
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=GM,
            num_warps=nw,
            num_stages=1,
        )
        return out

    if M >= 256:
        BM, BN, BK, GM, nw, ns = 256, 256, 32, 8, 8, 1
    else:
        BM, BN, BK, GM, nw, ns = 64, 128, 64, 8, 4, 1
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _gemm_kernel[grid](
        a,
        b,
        bias_arg,
        out,
        scale_a,
        scale_b,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        bia,
        bib,
        OUT_DTYPE=out_tl,
        IS_INT8=is_int8,
        HAS_BIAS=has_bias,
        SA_BCAST=sa_bcast,
        SB_BCAST=sb_bcast,
        EVEN_K=(K % BK == 0),
        EVEN_M=(M % BM == 0),
        EVEN_N=(N % BN == 0),
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        GROUP_M=GM,
        num_warps=nw,
        num_stages=ns,
    )
    return out


__all__ = ["triton_scaled_mm"]
