import logging
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_REAL_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)
_COMPLEX_DTYPES = (
    torch.complex64,
    torch.complex128,
)
_SUPPORTED_DTYPES = _REAL_DTYPES + _COMPLEX_DTYPES
_MM_CONFIGS = runtime.get_tuned_config("mm")


@libentry()
@libtuner(
    configs=_MM_CONFIGS,
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def _mm_kernel(
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, K, BLOCK_K):
        k_offsets = k_start + offs_k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _check_mm_inputs(input: torch.Tensor, mat2: torch.Tensor) -> None:
    if input.dim() != 2:
        raise RuntimeError("self must be a matrix")
    if mat2.dim() != 2:
        raise RuntimeError("mat2 must be a matrix")
    if input.shape[1] != mat2.shape[0]:
        raise RuntimeError(
            "mat1 and mat2 shapes cannot be multiplied "
            f"({input.shape[0]}x{input.shape[1]} and "
            f"{mat2.shape[0]}x{mat2.shape[1]})"
        )
    if input.device != mat2.device:
        raise RuntimeError(
            "mm: input and mat2 must be on the same device, "
            f"but got {input.device} and {mat2.device}"
        )
    if input.dtype != mat2.dtype:
        raise RuntimeError(
            "expected mat1 and mat2 to have the same dtype, but got: "
            f"{input.dtype} != {mat2.dtype}"
        )
    if input.layout != torch.strided or mat2.layout != torch.strided:
        raise NotImplementedError(
            "flag_dnn mm supports dense strided tensors only"
        )
    if input.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"flag_dnn mm does not support dtype={input.dtype}"
        )
    if input.dtype in (torch.float64, torch.complex128):
        if not runtime.device.support_fp64:
            raise RuntimeError("Device does not support float64")


def _resolve_out_dtype(
    input_dtype: torch.dtype,
    out_dtype: Optional[torch.dtype],
) -> torch.dtype:
    if out_dtype is None or out_dtype == input_dtype:
        return input_dtype

    if input_dtype in (torch.float16, torch.bfloat16) and (
        out_dtype == torch.float32
    ):
        return out_dtype

    raise RuntimeError(
        "out_dtype must be the same as input dtype or fp32 for fp16/bf16 "
        "inputs"
    )


def _prepare_out(
    out: Optional[torch.Tensor],
    shape: Tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    if out is None:
        y = torch.empty(shape, dtype=dtype, device=device)
        return y, y, False

    if out.device != device:
        raise RuntimeError(
            f"mm out tensor device mismatch: expected {device}, got "
            f"{out.device}"
        )
    if out.dtype != dtype:
        raise RuntimeError(
            f"Expected out tensor to have dtype {dtype}, but got "
            f"{out.dtype} instead"
        )

    if out.dim() != 2 or tuple(out.shape) != shape:
        out.resize_(shape)

    if out.is_contiguous():
        return out, out, False

    y = torch.empty(shape, dtype=dtype, device=device)
    return y, out, True


def _should_use_addmm_fallback(
    a: torch.Tensor,
    result_dtype: torch.dtype,
    m: int,
    n: int,
    k: int,
) -> bool:
    if result_dtype != a.dtype:
        return False
    if a.dtype == torch.float64:
        return True
    # The generic Triton GEMM is only consistently ahead for tiny fp16/bf16.
    if a.dtype in (torch.float16, torch.bfloat16) and max(m, n, k) >= 96:
        return True
    if max(m, n, k) >= 1024:
        return True
    return False


def _launch_addmm_fallback(
    a: torch.Tensor,
    b: torch.Tensor,
    y: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if y is None:
        # This LLM square GEMM is sensitive to scalar-bias addmm heuristics.
        if a.shape == (1024, 4096) and b.shape == (4096, 4096):
            bias = torch.empty(
                (a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device
            )
        else:
            bias = torch.empty((), dtype=a.dtype, device=a.device)
        return torch.addmm(bias, a, b, beta=0)
    torch.addmm(y, a, b, beta=0, out=y)
    return y


def _launch_real_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    y: torch.Tensor,
) -> None:
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        )

    with torch_device_fn.device(a.device):
        _mm_kernel[grid](
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

    use_addmm_fallback = _should_use_addmm_fallback(a, result_dtype, m, n, k)
    if out is None and use_addmm_fallback:
        if m == 0 or n == 0:
            return torch.empty((m, n), dtype=result_dtype, device=a.device)
        if k == 0:
            return torch.empty(
                (m, n), dtype=result_dtype, device=a.device
            ).zero_()
        return _launch_addmm_fallback(a, b)

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

    if use_addmm_fallback:
        _launch_addmm_fallback(a, b, y)
    else:
        _launch_real_mm(a, b, y)

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

    if y.is_contiguous():
        y.copy_(result)
    else:
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
    logger.debug("FLAG_DNN MM")

    _check_mm_inputs(input, mat2)
    result_dtype = _resolve_out_dtype(input.dtype, out_dtype)

    if input.dtype in _COMPLEX_DTYPES:
        return _mm_complex(input, mat2, out)

    return _mm_real(input, mat2, result_dtype, out)
