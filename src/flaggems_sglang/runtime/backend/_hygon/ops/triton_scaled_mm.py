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
def _widen_kernel(
    xa_ptr,
    xb_ptr,
    oa_ptr,
    ob_ptr,
    NA,
    NB,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    na_blocks = tl.cdiv(NA, BLOCK)
    if pid < na_blocks:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < NA
        x = tl.load(xa_ptr + offs, mask=mask, other=0.0)
        tl.store(oa_ptr + offs, x.to(tl.bfloat16), mask=mask)
    else:
        offs = (pid - na_blocks) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < NB
        x = tl.load(xb_ptr + offs, mask=mask, other=0.0)
        tl.store(ob_ptr + offs, x.to(tl.bfloat16), mask=mask)


@triton.autotune(
    configs=[
        # M <= 8: one M tile; fill the device with N tiles x splits.
        triton.Config(
            {"BM": 16, "BN": 256, "BK": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BM": 16, "BN": 128, "BK": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BM": 16, "BN": 256, "BK": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BM": 16, "BN": 128, "BK": 64}, num_warps=4, num_stages=2
        ),
        # M = 64: bigger M tiles reduce the fp8 widening traffic 4x per tile.
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BM": 32, "BN": 128, "BK": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BM": 64, "BN": 256, "BK": 32}, num_warps=8, num_stages=2
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_splitk_kernel(
    a_ptr,
    b_ptr,
    part_ptr,  # [SK, M, N] partial sums
    M,
    N,
    K,
    K_PER,  # k elements handled by one split (cdiv(K, SK))
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    pid = tl.program_id(0)
    sk_id = pid % SK
    pid_mn = pid // SK
    nn = tl.cdiv(N, BN)
    pid_m = pid_mn // nn
    pid_n = pid_mn % nn

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    am = (rm < M)[:, None]
    bn = (rn < N)[None, :]

    k_start = sk_id * K_PER
    k_end = tl.minimum(k_start + K_PER, K)

    a_ptrs = a_ptr + rm[:, None] * K + k_start + rk[None, :]
    b_ptrs = b_ptr + (k_start + rk[:, None]) * N + rn[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(k_start, k_end, BK):
        k_mask = (k + rk) < k_end
        av = tl.load(a_ptrs, mask=am & k_mask[None, :], other=0.0)
        bv = tl.load(b_ptrs, mask=bn & k_mask[:, None], other=0.0)
        acc = tl.dot(av.to(tl.bfloat16), bv.to(tl.bfloat16), acc)
        a_ptrs += BK
        b_ptrs += BK * N

    part_ptrs = part_ptr + sk_id * M * N + rm[:, None] * N + rn[None, :]
    tl.store(part_ptrs, acc, mask=am & bn)


@triton.autotune(
    configs=[
        triton.Config(
            {"BM": 16, "BN": 256, "BK": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BM": 16, "BN": 128, "BK": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BM": 16, "BN": 256, "BK": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BM": 16, "BN": 128, "BK": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 128}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BM": 32, "BN": 128, "BK": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 64}, num_warps=8, num_stages=2
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_splitk_i8_kernel(
    a_ptr,
    b_ptr,
    part_ptr,  # [SK, M, N] int32 partial sums
    M,
    N,
    K,
    K_PER,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    pid = tl.program_id(0)
    sk_id = pid % SK
    pid_mn = pid // SK
    nn = tl.cdiv(N, BN)
    pid_m = pid_mn // nn
    pid_n = pid_mn % nn

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    am = (rm < M)[:, None]
    bn = (rn < N)[None, :]

    k_start = sk_id * K_PER
    k_end = tl.minimum(k_start + K_PER, K)

    a_ptrs = a_ptr + rm[:, None] * K + k_start + rk[None, :]
    b_ptrs = b_ptr + (k_start + rk[:, None]) * N + rn[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(k_start, k_end, BK):
        k_mask = (k + rk) < k_end
        av = tl.load(a_ptrs, mask=am & k_mask[None, :], other=0)
        bv = tl.load(b_ptrs, mask=bn & k_mask[:, None], other=0)
        acc = tl.dot(av, bv, acc, out_dtype=tl.int32)
        a_ptrs += BK
        b_ptrs += BK * N

    part_ptrs = part_ptr + sk_id * M * N + rm[:, None] * N + rn[None, :]
    tl.store(part_ptrs, acc, mask=am & bn)


@triton.autotune(
    configs=[
        triton.Config({"BM": 16, "BN": 128}, num_warps=4, num_stages=1),
        triton.Config({"BM": 16, "BN": 256}, num_warps=4, num_stages=1),
        triton.Config({"BM": 32, "BN": 256}, num_warps=4, num_stages=1),
        triton.Config({"BM": 16, "BN": 512}, num_warps=8, num_stages=1),
    ],
    key=["M", "N", "SK"],
)
@triton.jit
def _splitk_epilogue_kernel(
    part_ptr,  # [SK, M, N]
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    M,
    N,
    SK,
    BM: tl.constexpr,
    BN: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    PART_I32: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SA_BCAST: tl.constexpr,
    SB_BCAST: tl.constexpr,
):
    pid = tl.program_id(0)
    nn = tl.cdiv(N, BN)
    pid_m = pid // nn
    pid_n = pid % nn

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)

    mask_m = rm < M
    mask_n = rn < N
    m2 = mask_m[:, None] & mask_n[None, :]

    if PART_I32:
        acc = tl.zeros((BM, BN), dtype=tl.int32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
    part_ptrs = part_ptr + rm[:, None] * N + rn[None, :]
    for s in range(SK):
        acc += tl.load(part_ptrs + s * M * N, mask=m2, other=0)

    accf = acc.to(tl.float32)
    if SA_BCAST:
        sa = tl.load(sa_ptr)
    else:
        sa = tl.load(sa_ptr + rm, mask=mask_m, other=1.0)
    if SB_BCAST:
        sb = tl.load(sb_ptr)
    else:
        sb = tl.load(sb_ptr + rn, mask=mask_n, other=1.0)
    accf = accf * sa[:, None] * sb[None, :]

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + rm[:, None] * N + rn[None, :], mask=m2, other=0.0
        )
        accf = accf + bias

    tl.store(
        c_ptr + rm[:, None] * N + rn[None, :], accf.to(OUT_DTYPE), mask=m2
    )


@triton.autotune(
    configs=[
        # Deep-pipe big tiles: best at large M/N (M=4096).
        triton.Config(
            {"BM": 256, "BN": 512, "BK": 32, "GROUP": 4},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 128, "BN": 512, "BK": 32, "GROUP": 4},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 256, "BN": 256, "BK": 32, "GROUP": 4},
            num_warps=8,
            num_stages=2,
        ),
        # Classic 64KB shmem tiles: best at mid M (M=512).
        triton.Config(
            {"BM": 128, "BN": 128, "BK": 64, "GROUP": 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 128, "BN": 128, "BK": 64, "GROUP": 8},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 128, "BN": 256, "BK": 64, "GROUP": 8},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 256, "BN": 128, "BK": 64, "GROUP": 8},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 256, "BN": 256, "BK": 64, "GROUP": 4},
            num_warps=8,
            num_stages=2,
        ),
        # Small-M coverage (M=64 tail of the two-pass range).
        triton.Config(
            {"BM": 64, "BN": 64, "BK": 64, "GROUP": 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 64, "BN": 128, "BK": 64, "GROUP": 8},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _scaled_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    EVEN_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SA_BCAST: tl.constexpr,
    SB_BCAST: tl.constexpr,
):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)

    num_in_group = GROUP * nn
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(nm - first_pid_m, GROUP)
    pid_m = first_pid_m + ((pid % num_in_group) % group_size_m)
    pid_n = (pid % num_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    am = (rm < M)[:, None]
    bn = (rn < N)[None, :]

    # A tile [BM, BK] (contiguous along k); B tile [BK, BN] (contiguous
    # along n).
    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        if EVEN_K:
            av = tl.load(a_ptrs, mask=am, other=0.0)
            bv = tl.load(b_ptrs, mask=bn, other=0.0)
        else:
            k_mask = (k + rk) < K
            av = tl.load(a_ptrs, mask=am & k_mask[None, :], other=0.0)
            bv = tl.load(b_ptrs, mask=bn & k_mask[:, None], other=0.0)
        acc = tl.dot(av, bv, acc)
        a_ptrs += BK
        b_ptrs += BK * N

    if SA_BCAST:
        sa = tl.load(sa_ptr)
    else:
        sa = tl.load(sa_ptr + rm, mask=rm < M, other=1.0)
    if SB_BCAST:
        sb = tl.load(sb_ptr)
    else:
        sb = tl.load(sb_ptr + rn, mask=rn < N, other=1.0)
    acc = acc * sa[:, None] * sb[None, :]

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + rm[:, None] * N + rn[None, :], mask=am & bn, other=0.0
        )
        acc = acc + bias

    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, acc.to(OUT_DTYPE), mask=am & bn)


@triton.autotune(
    configs=[
        triton.Config(
            {"BM": 256, "BN": 512, "BK": 64, "GROUP": 4},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 128, "BN": 512, "BK": 64, "GROUP": 4},
            num_warps=16,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 256, "BN": 256, "BK": 64, "GROUP": 4},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 128, "BN": 256, "BK": 64, "GROUP": 8},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 128, "BN": 128, "BK": 64, "GROUP": 8},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BM": 256, "BN": 128, "BK": 64, "GROUP": 8},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _scaled_mm_i8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    M,
    N,
    K,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    EVEN_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SA_BCAST: tl.constexpr,
    SB_BCAST: tl.constexpr,
):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)

    num_in_group = GROUP * nn
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP
    group_size_m = min(nm - first_pid_m, GROUP)
    pid_m = first_pid_m + ((pid % num_in_group) % group_size_m)
    pid_n = (pid % num_in_group) // group_size_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)

    am = (rm < M)[:, None]
    bn = (rn < N)[None, :]

    a_ptrs = a_ptr + rm[:, None] * K + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] * N + rn[None, :]

    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, K, BK):
        if EVEN_K:
            av = tl.load(a_ptrs, mask=am, other=0)
            bv = tl.load(b_ptrs, mask=bn, other=0)
        else:
            k_mask = (k + rk) < K
            av = tl.load(a_ptrs, mask=am & k_mask[None, :], other=0)
            bv = tl.load(b_ptrs, mask=bn & k_mask[:, None], other=0)
        acc = tl.dot(av, bv, acc, out_dtype=tl.int32)
        a_ptrs += BK
        b_ptrs += BK * N

    accf = acc.to(tl.float32)
    if SA_BCAST:
        sa = tl.load(sa_ptr)
    else:
        sa = tl.load(sa_ptr + rm, mask=rm < M, other=1.0)
    if SB_BCAST:
        sb = tl.load(sb_ptr)
    else:
        sb = tl.load(sb_ptr + rn, mask=rn < N, other=1.0)
    accf = accf * sa[:, None] * sb[None, :]

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + rm[:, None] * N + rn[None, :], mask=am & bn, other=0.0
        )
        accf = accf + bias

    c_ptrs = c_ptr + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, accf.to(OUT_DTYPE), mask=am & bn)


_WIDEN_BLOCK = 2048
_WIDEN_WARPS = 4

# Below this M the output has too few M tiles to fill the device and the
# fused split-K path wins; above it, weight reuse makes a plain GEMM cheaper.
_FUSED_MAX_M = 64


# Split counts per shape class, measured on this device: the grid must reach
# several waves of the 80-CU device without over-fragmenting K.
def _pick_split_k(m, n, k, in_dtype):
    if m <= 8:
        sk = 8
    elif in_dtype == torch.int8:
        sk = 4
    else:
        sk = 8 if n >= 7168 else 4
    sk = max(1, min(sk, triton.cdiv(k, 128)))
    return sk


def _torch_to_triton_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float64:
        return tl.float64


def triton_scaled_mm(input, weight, scale_a, scale_b, out_dtype, bias=None):
    m, k = input.shape
    n = weight.shape[1]

    input = input.contiguous()
    weight = weight.contiguous()
    scale_a = scale_a.contiguous()
    scale_b = scale_b.contiguous()

    device = input.device
    out_dt = _torch_to_triton_dtype(out_dtype)
    has_bias = bias is not None
    bias_arg = bias.contiguous() if has_bias else input

    sa_bcast = scale_a.numel() == 1
    sb_bcast = scale_b.numel() == 1

    out = torch.empty((m, n), dtype=out_dtype, device=device)

    if m <= _FUSED_MAX_M:
        is_i8 = input.dtype == torch.int8
        sk = _pick_split_k(m, n, k, input.dtype)
        part_dt = torch.int32 if is_i8 else torch.float32
        part = torch.empty((sk, m, n), dtype=part_dt, device=device)
        k_per = triton.cdiv(k, sk)

        if is_i8:
            main_kernel = _fused_splitk_i8_kernel
        else:
            main_kernel = _fused_splitk_kernel

        def grid(META):
            return (
                triton.cdiv(m, META["BM"]) * triton.cdiv(n, META["BN"]) * sk,
            )

        main_kernel[grid](
            input,
            weight,
            part,
            m,
            n,
            k,
            k_per,
            SK=sk,
        )

        def grid_e(META):
            return (triton.cdiv(m, META["BM"]) * triton.cdiv(n, META["BN"]),)

        _splitk_epilogue_kernel[grid_e](
            part,
            out,
            scale_a,
            scale_b,
            bias_arg,
            m,
            n,
            sk,
            OUT_DTYPE=out_dt,
            PART_I32=is_i8,
            HAS_BIAS=has_bias,
            SA_BCAST=sa_bcast,
            SB_BCAST=sb_bcast,
        )
        return out

    if input.dtype == torch.int8:
        # int8: single exact-int32 pass straight off the int8 operands.
        def grid(META):
            return (triton.cdiv(m, META["BM"]) * triton.cdiv(n, META["BN"]),)

        _scaled_mm_i8_kernel[grid](
            input,
            weight,
            out,
            scale_a,
            scale_b,
            bias_arg,
            m,
            n,
            k,
            OUT_DTYPE=out_dt,
            EVEN_K=k % 64 == 0,
            HAS_BIAS=has_bias,
            SA_BCAST=sa_bcast,
            SB_BCAST=sb_bcast,
        )
        return out

    a_bf = torch.empty((m, k), dtype=torch.bfloat16, device=device)
    b_bf = torch.empty((k, n), dtype=torch.bfloat16, device=device)

    na = input.numel()
    nb = weight.numel()
    grid_w = (triton.cdiv(na, _WIDEN_BLOCK) + triton.cdiv(nb, _WIDEN_BLOCK),)
    _widen_kernel[grid_w](
        input,
        weight,
        a_bf,
        b_bf,
        na,
        nb,
        BLOCK=_WIDEN_BLOCK,
        num_warps=_WIDEN_WARPS,
        num_stages=1,
    )

    def grid(META):
        return (triton.cdiv(m, META["BM"]) * triton.cdiv(n, META["BN"]),)

    _scaled_mm_kernel[grid](
        a_bf,
        b_bf,
        out,
        scale_a,
        scale_b,
        bias_arg,
        m,
        n,
        k,
        OUT_DTYPE=out_dt,
        EVEN_K=k % 64 == 0,
        HAS_BIAS=has_bias,
        SA_BCAST=sa_bcast,
        SB_BCAST=sb_bcast,
    )
    return out


__all__ = ["triton_scaled_mm"]
