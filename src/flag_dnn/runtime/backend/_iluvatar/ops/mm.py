import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.ops.dot import dot
from flag_dnn.ops.mm import (
    _check_mm_inputs,
    _launch_addmm_fallback,
    _prepare_out,
    _resolve_out_dtype,
    mm as _generic_mm,
)
from flag_dnn.ops.mv import mv
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_LOW_PRECISION_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
)
_COMPLEX_DTYPES = (
    torch.complex64,
    torch.complex128,
)
_TENSOR_CORE_DTYPES = (
    torch.float16,
    torch.bfloat16,
)

_SKINNY_MM_CONFIGS = runtime.get_tuned_config("skinny_mm")


def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _choose_small_num_warps(block_m: int, block_n: int) -> int:
    tile = block_m * block_n
    if tile <= 32:
        return 1
    if tile <= 128:
        return 2
    return 4


def _should_use_iluvatar_safe_mm(
    input_dtype: torch.dtype,
    result_dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
) -> bool:
    if input_dtype not in _LOW_PRECISION_DTYPES:
        return False
    if result_dtype not in _LOW_PRECISION_DTYPES:
        return False
    if input_dtype == torch.float32 and (m > 128 or n > 128):
        return False
    return k <= 32 or max(m, n) <= 16


def _should_use_vendor_gemm_fastpath(
    input_dtype: torch.dtype,
    result_dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
) -> bool:
    if input_dtype != torch.float32 or result_dtype != torch.float32:
        return False
    return k <= 32 and m * n >= 65536


def _should_use_skinny_dot_mm(
    input_dtype: torch.dtype,
    result_dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
) -> bool:
    if input_dtype not in _TENSOR_CORE_DTYPES:
        return False
    if result_dtype not in _LOW_PRECISION_DTYPES:
        return False

    # K=1 你现在已经很快，保留 outer-product small kernel。
    # K=4/16/32 需要走 tl.dot，否则 bf16/fp16 吃不到 tensor-core 路径。
    if k < 4 or k > 32:
        return False

    # 只处理 large-output skinny-K。
    # 小 M/N 继续交给原 safe kernel，避免 dot padding 反而亏。
    if m < 32 or n < 32:
        return False

    return m * n >= 65536


@libentry()
@libtuner(
    configs=_SKINNY_MM_CONFIGS,
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def _skinny_dot_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tle.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # grouped ordering:
    # 对 skinny-K GEMM，B 很小但会被很多 M-block 复用；
    # group M 可以提高 B tile 的 cache 复用。
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a = tl.load(
        a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
        mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
        other=0.0,
    )

    b = tl.load(
        b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
        mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
        other=0.0,
    )

    acc = tl.dot(a, b)

    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _launch_skinny_dot_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.shape[1]

    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        )

    with torch_device_fn.device(a.device):
        _skinny_dot_mm_kernel[grid](
            a,
            b,
            y,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            y.stride(0),
            y.stride(1),
        )


@libentry()
@triton.jit
def _small_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K, num_stages=1):
        for kk in tl.static_range(0, BLOCK_K):
            cur_k = k_start + kk
            k_mask = cur_k < K

            a = tl.load(
                a_ptr + offs_m * stride_am + cur_k * stride_ak,
                mask=(offs_m < M) & k_mask,
                other=0.0,
            ).to(tl.float32)
            b = tl.load(
                b_ptr + cur_k * stride_bk + offs_n * stride_bn,
                mask=k_mask & (offs_n < N),
                other=0.0,
            ).to(tl.float32)

            acc += a[:, None] * b[None, :]

    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _launch_small_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
) -> None:
    m, k = a.shape
    n = b.shape[1]

    block_m = min(16, _next_power_of_two(min(m, 16)))
    block_n = min(16, _next_power_of_two(min(n, 16)))

    # 原来固定 32。这里至少避免 K=1/2/4 走太多无效展开。
    # large K=4/16 已经会被 _skinny_dot_mm_kernel 拦截，
    # 这里主要服务小 shape 或 K=1。
    block_k = min(32, _next_power_of_two(k))

    num_warps = _choose_small_num_warps(block_m, block_n)

    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    with torch_device_fn.device(a.device):
        _small_mm_kernel[grid](
            a,
            b,
            y,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )


def _try_vector_specialization(
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
    result_dtype: torch.dtype,
) -> bool:
    m, _ = a.shape
    n = b.shape[1]
    if result_dtype != a.dtype:
        return False

    if m == 1 and n == 1:
        dot(a.reshape(-1), b.reshape(-1), out=y.view(()))
        return True

    if n == 1:
        mv(a, b[:, 0], out=y.view(-1))
        return True

    if m == 1:
        mv(b.t(), a.reshape(-1), out=y.view(-1))
        return True

    return False


def _mm_real(
    input: torch.Tensor,
    mat2: torch.Tensor,
    result_dtype: torch.dtype,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    a = input if input.is_contiguous() else input.contiguous()
    b = mat2 if mat2.is_contiguous() else mat2.contiguous()

    m, k = a.shape
    n = b.shape[1]
    if not _should_use_iluvatar_safe_mm(a.dtype, result_dtype, m, n, k):
        return _generic_mm(input, mat2, out_dtype=result_dtype, out=out)

    y, out_tensor, need_copy_back = _prepare_out(
        out, (m, n), result_dtype, a.device
    )

    if y.numel() == 0:
        return out_tensor

    if k == 0:
        y.zero_()
        if need_copy_back:
            out_tensor.copy_(y)
        return out_tensor

    if _should_use_vendor_gemm_fastpath(a.dtype, result_dtype, m, n, k):
        _launch_addmm_fallback(a, b, y)
        if need_copy_back:
            out_tensor.copy_(y)
        return out_tensor

    if _try_vector_specialization(a, b, y, result_dtype):
        if need_copy_back:
            out_tensor.copy_(y)
        return out_tensor

    # 关键修复：
    # bf16/fp16 + large M/N + K=4/16/32 走 tl.dot。
    # K=1 继续走 _small_mm_kernel，因为你的结果里 K=1 已经很快。
    if _should_use_skinny_dot_mm(a.dtype, result_dtype, m, n, k):
        _launch_skinny_dot_mm(a, b, y)
    else:
        _launch_small_mm(a, b, y)

    if need_copy_back:
        out_tensor.copy_(y)
    return out_tensor


def _mm_complex(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    m, k = input.shape
    n = mat2.shape[1]
    y, out_tensor, need_copy_back = _prepare_out(
        out, (m, n), input.dtype, input.device
    )

    if y.numel() == 0:
        return out_tensor

    if k == 0:
        y.zero_()
        if need_copy_back:
            out_tensor.copy_(y)
        return out_tensor

    ar = input.real.contiguous()
    ai = input.imag.contiguous()
    br = mat2.real.contiguous()
    bi = mat2.imag.contiguous()

    real = mm(ar, br) - mm(ai, bi)
    imag = mm(ar, bi) + mm(ai, br)
    result = torch.complex(real, imag)
    y.copy_(result)

    if need_copy_back:
        out_tensor.copy_(y)
    return out_tensor


def mm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN ILUVATAR MM")

    _check_mm_inputs(input, mat2)
    result_dtype = _resolve_out_dtype(input.dtype, out_dtype)

    if input.dtype in _COMPLEX_DTYPES:
        return _mm_complex(input, mat2, out)

    return _mm_real(input, mat2, result_dtype, out)
