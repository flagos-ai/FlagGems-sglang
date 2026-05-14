import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.type_utils import is_integral_dtype


logger = logging.getLogger(__name__)


_SUPPORTED_DTYPES = {
    torch.bool,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


@triton.jit
def _prod_combine(a, b):
    return a * b


# ------------------------------------------------------------
# Shared configs for flat 2D / 3D kernels
# ------------------------------------------------------------
_PROD_FLAT_CONFIGS = [
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
# Dedicated full-reduce (M == 1) split-N kernels
# Fixed BLOCK_N chosen by host, so partial size is exact.
# ------------------------------------------------------------
@triton.jit
def _prod_kernel_1row_split_stage1(
    x_ptr,
    partial_ptr,
    N,
    stride_xn,
    IS_FP64: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    n_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    x = tl.load(x_ptr + n_offsets * stride_xn, mask=n_mask, other=1.0)
    x = x.to(tl.float64 if IS_FP64 else tl.float32)

    part_val = tl.reduce(x, axis=0, combine_fn=_prod_combine)
    tl.store(partial_ptr + pid, part_val)


@triton.jit
def _prod_kernel_1row_finalize(
    partial_ptr,
    out_ptr,
    N,
    stride_pn,
    IS_FP64: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    acc = tl.full((1,), 1.0, dtype=tl.float64 if IS_FP64 else tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        x = tl.load(
            partial_ptr + n_offsets * stride_pn, mask=n_mask, other=1.0
        )
        x = x.to(tl.float64 if IS_FP64 else tl.float32)

        acc *= tl.reduce(x, axis=0, combine_fn=_prod_combine)

    tl.store(out_ptr + tl.arange(0, 1), acc)


# ------------------------------------------------------------
# 2D flat loop kernel
# Directly stores to out_ptr dtype
# ------------------------------------------------------------
@triton.autotune(
    configs=_PROD_FLAT_CONFIGS,
    key=["M", "N"],
)
@triton.jit
def _prod_kernel_2d_loop_store(
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

    acc = tl.full((BLOCK_M,), 1.0, dtype=tl.float64 if IS_FP64 else tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = (
            x_ptr
            + m_offsets[:, None] * stride_xm
            + n_offsets[None, :] * stride_xn
        )

        x = tl.load(x_ptrs, mask=mask, other=1.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)

        acc *= tl.reduce(x, axis=1, combine_fn=_prod_combine)

    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


# ------------------------------------------------------------
# 3D flat loop kernel
# input_view layout: [O, R, I]
# output flattened as M = O * I
# Directly stores to out_ptr dtype
# ------------------------------------------------------------
@triton.autotune(
    configs=_PROD_FLAT_CONFIGS,
    key=["M", "N"],
)
@triton.jit
def _prod_kernel_3d_loop_store(
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

    acc = tl.full((BLOCK_M,), 1.0, dtype=tl.float64 if IS_FP64 else tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=1.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)

        acc *= tl.reduce(x, axis=1, combine_fn=_prod_combine)

    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


# ------------------------------------------------------------
# General split-N stage1 for small M / huge N
# partial buffer layout: [M, PARTIAL_N]
# ------------------------------------------------------------
@triton.jit
def _prod_kernel_2d_split_stage1(
    x_ptr,
    partial_ptr,
    M,
    N,
    PARTIAL_N,
    stride_xm,
    stride_xn,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    pid_m = pid // PARTIAL_N
    pid_n = pid % PARTIAL_N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    x_ptrs = (
        x_ptr + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
    )
    x = tl.load(x_ptrs, mask=mask, other=1.0)
    x = x.to(tl.float64 if IS_FP64 else tl.float32)

    part_vals = tl.reduce(x, axis=1, combine_fn=_prod_combine)

    partial_ptrs = partial_ptr + m_offsets * PARTIAL_N + pid_n
    tl.store(partial_ptrs, part_vals, mask=m_mask)


@triton.jit
def _prod_kernel_3d_split_stage1(
    x_ptr,
    partial_ptr,
    M,
    N,
    I,
    PARTIAL_N,
    stride_xo,
    stride_xr,
    stride_xi,
    IS_FP64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    pid_m = pid // PARTIAL_N
    pid_n = pid % PARTIAL_N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    o_idx = m_offsets // I
    i_idx = m_offsets % I

    x_ptrs = x_ptr + (
        o_idx[:, None] * stride_xo
        + n_offsets[None, :] * stride_xr
        + i_idx[:, None] * stride_xi
    )

    x = tl.load(x_ptrs, mask=mask, other=1.0)
    x = x.to(tl.float64 if IS_FP64 else tl.float32)

    part_vals = tl.reduce(x, axis=1, combine_fn=_prod_combine)

    partial_ptrs = partial_ptr + m_offsets * PARTIAL_N + pid_n
    tl.store(partial_ptrs, part_vals, mask=m_mask)


# ------------------------------------------------------------
# Specialized dim=0 small-R kernel
# Only for O == 1, R small, I very large
# input_view layout: [1, R, I]
# grid = (ceil_div(I, BLOCK_I),)
# Directly stores to out_ptr dtype
# ------------------------------------------------------------
_PROD_DIM0_SMALL_R_CONFIGS = [
    triton.Config({"BLOCK_I": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_I": 128}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_I": 256}, num_warps=8, num_stages=3),
]


@triton.autotune(
    configs=_PROD_DIM0_SMALL_R_CONFIGS,
    key=["R", "I"],
)
@triton.jit
def _prod_kernel_dim0_small_r_store(
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
    acc = tl.full((BLOCK_I,), 1.0, dtype=tl.float64 if IS_FP64 else tl.float32)

    for r in range(0, R):
        x = tl.load(base_ptrs + r * stride_xr, mask=i_mask, other=1.0)
        x = x.to(tl.float64 if IS_FP64 else tl.float32)
        acc *= x

    tl.store(out_ptr + i_offsets, acc, mask=i_mask)


def prod(
    input: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN PROD")

    target_dtype = (
        dtype
        if dtype is not None
        else (torch.int64 if is_integral_dtype(input.dtype) else input.dtype)
    )

    if input.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(f"prod does not support dtype={input.dtype}")
    if dtype is not None and dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(f"prod does not support dtype={dtype}")

    ndim = input.ndim
    if dim is None:
        dims = list(range(ndim))
    elif isinstance(dim, int):
        if dim < -ndim or dim >= ndim:
            raise IndexError(
                f"Dimension out of range (expected to be"
                f"in range of [{-ndim}, {ndim - 1}], but got {dim})"
            )
        dims = [dim]
    else:
        raise AssertionError("Not Support dim is tuple")

    dims = [d if d >= 0 else d + ndim for d in dims]

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

    def _select_1row_block_n():
        if input.dtype in (torch.float16, torch.bfloat16):
            return 16384, 1024
        if input.dtype == torch.float32:
            return 8192, 1024
        return 4096, 512

    def _launch_kernel_1row_split_n(N, input_view):
        if N == 0:
            return torch.ones(
                (1,), dtype=target_dtype, device=input_view.device
            )

        is_fp64 = (input_view.dtype == torch.float64) or (
            target_dtype == torch.float64
        )
        block_n_stage1, block_n_stage2 = _select_1row_block_n()

        num_tiles = triton.cdiv(N, block_n_stage1)
        partial_buffer = torch.empty(
            (num_tiles,), dtype=acc_dtype, device=input_view.device
        )

        grid_stage1 = (num_tiles,)
        _prod_kernel_1row_split_stage1[grid_stage1](
            input_view,
            partial_buffer,
            N,
            input_view.stride(1),
            is_fp64,
            BLOCK_N=block_n_stage1,
            num_warps=8,
        )

        out_buffer = torch.empty(
            (1,), dtype=target_dtype, device=input_view.device
        )
        _prod_kernel_1row_finalize[(1,)](
            partial_buffer,
            out_buffer,
            num_tiles,
            partial_buffer.stride(0),
            is_fp64,
            BLOCK_N=block_n_stage2,
            num_warps=8,
        )
        return out_buffer

    def _launch_kernel_split_n(M, N, input_view, is_3d=False, I_dim=1):
        if M == 0:
            return torch.empty(
                out_shape, dtype=target_dtype, device=input.device
            )
        if N == 0:
            return torch.ones(
                (M,), dtype=target_dtype, device=input_view.device
            )

        is_fp64 = (input_view.dtype == torch.float64) or (
            target_dtype == torch.float64
        )

        if input.dtype in (torch.float16, torch.bfloat16):
            BLOCK_N_SPLIT = 16384
            BLOCK_M_SPLIT = 8
        else:
            BLOCK_N_SPLIT = 8192
            BLOCK_M_SPLIT = 4 if input.dtype == torch.float64 else 8

        num_tiles = triton.cdiv(N, BLOCK_N_SPLIT)
        partial_buffer = torch.empty(
            (M, num_tiles), dtype=acc_dtype, device=input_view.device
        )

        if not is_3d:
            grid = (triton.cdiv(M, BLOCK_M_SPLIT) * num_tiles,)
            _prod_kernel_2d_split_stage1[grid](
                input_view,
                partial_buffer,
                M,
                N,
                num_tiles,
                input_view.stride(0),
                input_view.stride(1),
                is_fp64,
                BLOCK_M=BLOCK_M_SPLIT,
                BLOCK_N=BLOCK_N_SPLIT,
                num_warps=8,
            )
        else:
            grid = (triton.cdiv(M, BLOCK_M_SPLIT) * num_tiles,)
            _prod_kernel_3d_split_stage1[grid](
                input_view,
                partial_buffer,
                M,
                N,
                I_dim,
                num_tiles,
                input_view.stride(0),
                input_view.stride(1),
                input_view.stride(2),
                is_fp64,
                BLOCK_M=BLOCK_M_SPLIT,
                BLOCK_N=BLOCK_N_SPLIT,
                num_warps=8,
            )

        out_buffer = torch.empty(
            (M,), dtype=target_dtype, device=input_view.device
        )

        def grid_2d_loop(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        _prod_kernel_2d_loop_store[grid_2d_loop](
            partial_buffer,
            out_buffer,
            M,
            num_tiles,
            partial_buffer.stride(0),
            partial_buffer.stride(1),
            is_fp64,
        )

        return out_buffer

    def _launch_kernel_flat_2d(M, N, input_view):
        if M == 0:
            return torch.empty(
                out_shape, dtype=target_dtype, device=input.device
            )
        if N == 0:
            return torch.ones(
                (M,), dtype=target_dtype, device=input_view.device
            )

        is_fp64 = (input_view.dtype == torch.float64) or (
            target_dtype == torch.float64
        )
        out_buffer = torch.empty((M,), dtype=target_dtype, device=input.device)

        def grid_2d_loop(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        _prod_kernel_2d_loop_store[grid_2d_loop](
            input_view,
            out_buffer,
            M,
            N,
            input_view.stride(0),
            input_view.stride(1),
            is_fp64,
        )
        return out_buffer

    def _launch_kernel_flat_3d(M, N, I_dim, input_view):
        if M == 0:
            return torch.empty(
                out_shape, dtype=target_dtype, device=input.device
            )
        if N == 0:
            return torch.ones(
                (M,), dtype=target_dtype, device=input_view.device
            )

        is_fp64 = (input_view.dtype == torch.float64) or (
            target_dtype == torch.float64
        )
        out_buffer = torch.empty((M,), dtype=target_dtype, device=input.device)

        def grid_3d_loop(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        _prod_kernel_3d_loop_store[grid_3d_loop](
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
        if I_dim == 0:
            return torch.empty(
                out_shape, dtype=target_dtype, device=input.device
            )
        if R == 0:
            return torch.ones(
                (I_dim,), dtype=target_dtype, device=input_view.device
            )

        is_fp64 = (input_view.dtype == torch.float64) or (
            target_dtype == torch.float64
        )
        out_buffer = torch.empty(
            (I_dim,), dtype=target_dtype, device=input.device
        )

        def grid_dim0(meta):
            return (triton.cdiv(I_dim, meta["BLOCK_I"]),)

        _prod_kernel_dim0_small_r_store[grid_dim0](
            input_view,
            out_buffer,
            R,
            I_dim,
            input_view.stride(1),
            input_view.stride(2),
            is_fp64,
        )
        return out_buffer

    def _prefer_1row_split_n(M, N):
        return (M == 1) and (N >= (1 << 20))

    def _prefer_split_n(M, N):
        return (M <= 4 and N >= (1 << 18)) or (M <= 32 and N >= (1 << 20))

    def _prefer_dim0_small_r(O_dim, R, I_dim):
        return (O_dim == 1) and (R <= 64) and (I_dim >= 65536)

    with torch_device_fn.device(input.device):
        if is_reduce_at_end:
            M, N = 1, 1
            for i in range(ndim - len(dims)):
                M *= input.shape[i]
            for i in range(ndim - len(dims), ndim):
                N *= input.shape[i]

            input_view = input.reshape(M, N)

            if _prefer_1row_split_n(M, N):
                out_buffer = _launch_kernel_1row_split_n(N, input_view)
            elif _prefer_split_n(M, N):
                out_buffer = _launch_kernel_split_n(
                    M, N, input_view, is_3d=False
                )
            else:
                out_buffer = _launch_kernel_flat_2d(M, N, input_view)
        else:
            dim_min, dim_max = dims[0], dims[-1]
            O, R, I_dim = 1, 1, 1
            for j in range(dim_min):
                O *= input.shape[j]
            for j in range(dim_min, dim_max + 1):
                R *= input.shape[j]
            for j in range(dim_max + 1, ndim):
                I_dim *= input.shape[j]

            M, N = O * I_dim, R
            input_view = input.reshape(O, R, I_dim)

            if _prefer_dim0_small_r(O, R, I_dim):
                out_buffer = _launch_kernel_dim0_small_r(R, I_dim, input_view)
            elif _prefer_1row_split_n(M, N):
                out_buffer = _launch_kernel_1row_split_n(
                    N, input_view.reshape(1, N)
                )
            elif _prefer_split_n(M, N):
                out_buffer = _launch_kernel_split_n(
                    M, N, input_view, is_3d=True, I_dim=I_dim
                )
            else:
                out_buffer = _launch_kernel_flat_3d(M, N, I_dim, input_view)

    return out_buffer.reshape(out_shape)
