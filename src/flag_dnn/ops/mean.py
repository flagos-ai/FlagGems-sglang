import logging
from typing import Optional, Union, Tuple

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Shared configs for flat 2D / 3D kernels
# ------------------------------------------------------------
_MEAN_FLAT_CONFIGS = [
    triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
    triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
    triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
    triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
    triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
]


# ------------------------------------------------------------
# 2D loop kernel
# 直接写最终 dtype，避免额外 cast kernel
# ------------------------------------------------------------
@triton.autotune(
    configs=_MEAN_FLAT_CONFIGS,
    key=["M", "N"],
)
@triton.jit
def _mean_kernel_2d_loop_store(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float64 if IS_FP64 else tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)
        acc += tl.sum(x, axis=1)

    tl.store(out_ptr + m_offsets, acc / N, mask=m_mask)


# ------------------------------------------------------------
# 3D flat loop kernel
# input_view layout: [O, R, I]
# output flattened as M = O * I
# 直接写最终 dtype
# ------------------------------------------------------------
@triton.autotune(
    configs=_MEAN_FLAT_CONFIGS,
    key=["M", "N"],
)
@triton.jit
def _mean_kernel_3d_loop_store(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + (o_idx * stride_xo + i_idx * stride_xi)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float64 if IS_FP64 else tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)
        acc += tl.sum(x, axis=1)

    tl.store(out_ptr + m_offsets, acc / N, mask=m_mask)


# ------------------------------------------------------------
# 专用 dim=0 small-R kernel
# 仅用于 O == 1, R 小, I 很大的情况
# input_view layout: [1, R, I]
# grid = (ceil_div(I, BLOCK_I),)
# 直接写最终 dtype
# ------------------------------------------------------------
_MEAN_DIM0_SMALL_R_CONFIGS = [
    triton.Config({"BLOCK_I": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_I": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_I": 256}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=_MEAN_DIM0_SMALL_R_CONFIGS,
    key=["R", "I"],
)
@triton.jit
def _mean_kernel_dim0_small_r_store(
    x_ptr,
    out_ptr,
    R,
    I,
    stride_xr,
    stride_xi,
    IS_FP64: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_i = tle.program_id(0)

    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    i_mask = i_offsets < I

    base_ptrs = x_ptr + i_offsets * stride_xi
    acc = tl.zeros((BLOCK_I,), dtype=tl.float64 if IS_FP64 else tl.float32)

    for r in range(0, R):
        x = tl.load(base_ptrs + r * stride_xr, mask=i_mask, other=0.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)
        acc += x

    tl.store(out_ptr + i_offsets, acc / R, mask=i_mask)


# ------------------------------------------------------------
# 2D atomic kernel
# atomic 路径仍然写 acc_dtype scratch，再必要时 cast
# ------------------------------------------------------------
@triton.autotune(
    configs=_MEAN_FLAT_CONFIGS,
    key=["M", "N"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _mean_kernel_2d_atomic(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    mask = m_mask[:, None] & n_mask[None, :]
    x_ptrs = (
        x_ptr + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x = x.to(tl.float64 if IS_FP64 else tl.float32)

    sum_vals = tl.sum(x, axis=1) / N
    tl.atomic_add(out_ptr + m_offsets, sum_vals, mask=m_mask)


# ------------------------------------------------------------
# 3D atomic kernel
# ------------------------------------------------------------
@triton.autotune(
    configs=_MEAN_FLAT_CONFIGS,
    key=["M", "N"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _mean_kernel_3d_atomic(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    o_idx = m_offsets // I
    i_idx = m_offsets % I

    mask = m_mask[:, None] & n_mask[None, :]
    x_ptrs = x_ptr + (
        o_idx[:, None] * stride_xo
        + n_offsets[None, :] * stride_xr
        + i_idx[:, None] * stride_xi
    )

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x = x.to(tl.float64 if IS_FP64 else tl.float32)

    sum_vals = tl.sum(x, axis=1) / N
    tl.atomic_add(out_ptr + m_offsets, sum_vals, mask=m_mask)


def mean(
    input: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN MEAN")

    target_dtype = dtype if dtype is not None else input.dtype
    if target_dtype in (torch.int8, torch.int16, torch.bool):
        target_dtype = torch.float32

    ndim = input.ndim
    if dim is None:
        dims = list(range(ndim))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)

    dims = [d if d >= 0 else d + ndim for d in dims]
    dims = sorted(dims)

    out_shape = []
    for i in range(ndim):
        if i in dims:
            if keepdim:
                out_shape.append(1)
        else:
            out_shape.append(input.shape[i])

    if not out_shape and not keepdim:
        out_shape = []

    acc_dtype = (
        torch.float64
        if (input.dtype == torch.float64 or target_dtype == torch.float64)
        else torch.float32
    )

    is_reduce_at_end = dims == list(range(ndim - len(dims), ndim))

    is_consecutive = True
    for j in range(len(dims) - 1):
        if dims[j] + 1 != dims[j + 1]:
            is_consecutive = False
            break

    def _launch_kernel_2d(M, N, input_view):
        if M == 0 or N == 0:
            val = float("nan") if N == 0 else 0.0
            return torch.full(
                out_shape, val, dtype=target_dtype, device=input.device
            )

        is_fp64 = (input_view.dtype == torch.float64) or (
            target_dtype == torch.float64
        )
        use_atomic = (M <= 32) and (N > 65536)

        if use_atomic:
            out_buffer = torch.zeros(
                (M,), dtype=acc_dtype, device=input.device
            )

            def grid_2d_atomic(meta):
                return (
                    triton.cdiv(M, meta["BLOCK_M"])
                    * triton.cdiv(N, meta["BLOCK_N"]),
                )

            _mean_kernel_2d_atomic[grid_2d_atomic](
                input_view,
                out_buffer,
                M,
                N,
                input_view.stride(0),
                input_view.stride(1),
                is_fp64,
            )

            if out_buffer.dtype != target_dtype:
                out_buffer = out_buffer.to(target_dtype)
            return out_buffer

        out_buffer = torch.empty((M,), dtype=target_dtype, device=input.device)

        def grid_2d_loop(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        _mean_kernel_2d_loop_store[grid_2d_loop](
            input_view,
            out_buffer,
            M,
            N,
            input_view.stride(0),
            input_view.stride(1),
            is_fp64,
        )
        return out_buffer

    def _launch_kernel_3d_flat(M, N, I_dim, input_view):
        if M == 0 or N == 0:
            val = float("nan") if N == 0 else 0.0
            return torch.full(
                out_shape, val, dtype=target_dtype, device=input.device
            )

        is_fp64 = (input_view.dtype == torch.float64) or (
            target_dtype == torch.float64
        )
        use_atomic = (M <= 32) and (N > 65536)

        if use_atomic:
            out_buffer = torch.zeros(
                (M,), dtype=acc_dtype, device=input.device
            )

            def grid_3d_atomic(meta):
                return (
                    triton.cdiv(M, meta["BLOCK_M"])
                    * triton.cdiv(N, meta["BLOCK_N"]),
                )

            _mean_kernel_3d_atomic[grid_3d_atomic](
                input_view,
                out_buffer,
                M,
                N,
                I_dim,
                input_view.stride(0),
                input_view.stride(1),
                input_view.stride(2),
                is_fp64,
            )

            if out_buffer.dtype != target_dtype:
                out_buffer = out_buffer.to(target_dtype)
            return out_buffer

        out_buffer = torch.empty((M,), dtype=target_dtype, device=input.device)

        def grid_3d_loop(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        _mean_kernel_3d_loop_store[grid_3d_loop](
            input_view,
            out_buffer,
            M,
            N,
            I_dim,
            input_view.stride(0),
            input_view.stride(1),
            input_view.stride(2),
            is_fp64,
        )
        return out_buffer

    def _launch_kernel_dim0_small_r(R, I_dim, input_view):
        if R == 0 or I_dim == 0:
            val = float("nan") if R == 0 else 0.0
            return torch.full(
                out_shape, val, dtype=target_dtype, device=input.device
            )

        is_fp64 = (input_view.dtype == torch.float64) or (
            target_dtype == torch.float64
        )
        out_buffer = torch.empty(
            (I_dim,), dtype=target_dtype, device=input.device
        )

        def grid_dim0(meta):
            return (triton.cdiv(I_dim, meta["BLOCK_I"]),)

        _mean_kernel_dim0_small_r_store[grid_dim0](
            input_view,
            out_buffer,
            R,
            I_dim,
            input_view.stride(1),
            input_view.stride(2),
            is_fp64,
        )
        return out_buffer

    def _prefer_dim0_small_r(O_dim, R, I_dim):
        # 仅覆盖:
        # [32, 1024, 1024], dim=0 -> O=1, R=32, I=1048576
        return (O_dim == 1) and (R <= 64) and (I_dim >= 65536)

    with torch_device_fn.device(input.device):
        if is_reduce_at_end:
            M, N = 1, 1
            for i in range(ndim - len(dims)):
                M *= input.shape[i]
            for i in range(ndim - len(dims), ndim):
                N *= input.shape[i]

            out_buffer = _launch_kernel_2d(M, N, input.reshape(M, N))

        elif is_consecutive:
            dim_min, dim_max = dims[0], dims[-1]
            O, R, I_dim = 1, 1, 1
            for j in range(dim_min):
                O *= input.shape[j]
            for j in range(dim_min, dim_max + 1):
                R *= input.shape[j]
            for j in range(dim_max + 1, ndim):
                I_dim *= input.shape[j]

            input_view = input.reshape(O, R, I_dim)

            if _prefer_dim0_small_r(O, R, I_dim):
                out_buffer = _launch_kernel_dim0_small_r(R, I_dim, input_view)
            else:
                M, N = O * I_dim, R
                out_buffer = _launch_kernel_3d_flat(M, N, I_dim, input_view)

        else:
            kept_dims = [i for i in range(ndim) if i not in dims]
            M, N = 1, 1
            for d in kept_dims:
                M *= input.shape[d]
            for d in dims:
                N *= input.shape[d]

            input_view = input.permute(*kept_dims, *dims).reshape(M, N)
            out_buffer = _launch_kernel_2d(M, N, input_view)

    out_buffer = out_buffer.reshape(out_shape)

    if out is not None:
        out.resize_(out_shape)
        out.copy_(out_buffer)
        return out

    return out_buffer
