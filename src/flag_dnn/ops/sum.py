import logging
from typing import Optional, Union, Tuple

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.type_utils import is_integral_dtype


logger = logging.getLogger(__name__)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 4096}, num_warps=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 2048}, num_warps=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 2048}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=8),
    ],
    key=["M", "N"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _sum_kernel_2d_atomic_fp64(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

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
    sum_vals = tl.sum(x, axis=1, dtype=tl.float64)

    tl.atomic_add(out_ptr + m_offsets, sum_vals, mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 4096}, num_warps=4),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 2048}, num_warps=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 2048}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=8),
    ],
    key=["M", "N"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _sum_kernel_3d_atomic_fp64(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

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
    sum_vals = tl.sum(x, axis=1, dtype=tl.float64)

    tl.atomic_add(out_ptr + m_offsets, sum_vals, mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 2048}, num_warps=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 16}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _sum_kernel_2d_loop_fp64(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float64)

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
        acc += tl.sum(x, axis=1, dtype=tl.float64)

    out_ptrs = out_ptr + m_offsets
    tl.store(out_ptrs, acc, mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 2048}, num_warps=4),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 16}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _sum_kernel_3d_loop_fp64(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + (o_idx * stride_xo + i_idx * stride_xi)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float64)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=1, dtype=tl.float64)

    out_ptrs = out_ptr + m_offsets
    tl.store(out_ptrs, acc, mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _sum_kernel_3d_loop_transpose(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + o_idx * stride_xo + i_idx * stride_xi

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        # 注意这里把 tile 组织成 [BLOCK_N, BLOCK_M]
        mask = n_mask[:, None] & m_mask[None, :]
        x_ptrs = base_ptrs[None, :] + n_offsets[:, None] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)

        # 沿着 n 维归约，输出还是 BLOCK_M 个结果
        acc += tl.sum(x, axis=0)

    tl.store(out_ptr + m_offsets, acc, mask=m_mask)


@triton.autotune(
    configs=[
        # 1. 针对 N 极大，M 极小的极端场景 (Reduce at end)
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        # 2. 针对 M 极大，N 极小的极端场景 (例如 dim=0 的情况)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
        # 3. 增加高 Warp 数（8 warps）加强版，提升带宽高负载时的吞吐
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
        # 4. 常规均衡态
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 16}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _sum_kernel_2d_loop(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

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
        x = x.to(tl.float32)

        acc += tl.sum(x, axis=1)

    out_ptrs = out_ptr + m_offsets
    tl.store(out_ptrs, acc, mask=m_mask)


@triton.autotune(
    configs=[
        # 1. 针对 N 极大，M 极小的极端场景 (Reduce at end)
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        # 2. 针对 M 极大，N 极小的极端场景 (例如 dim=0 的情况)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
        # 3. 增加高 Warp 数（8 warps）加强版，提升带宽高负载时的吞吐
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
        # 4. 常规均衡态
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 32}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 16}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _sum_kernel_3d_loop(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tle.program_id(0)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M

    o_idx = m_offsets // I
    i_idx = m_offsets % I
    base_ptrs = x_ptr + (o_idx * stride_xo + i_idx * stride_xi)

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N

        mask = m_mask[:, None] & n_mask[None, :]
        x_ptrs = base_ptrs[:, None] + n_offsets[None, :] * stride_xr

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)

        acc += tl.sum(x, axis=1)

    out_ptrs = out_ptr + m_offsets
    tl.store(out_ptrs, acc, mask=m_mask)


@triton.autotune(
    configs=[
        # 1. 针对 N 极大，M 极小的极端场景 (Reduce at end)
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        # 2. 针对 M 极大，N 极小的极端场景 (例如 dim=0 的情况)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
        # 3. 增加高 Warp 数（8 warps）加强版，提升带宽高负载时的吞吐
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
        # 4. 常规均衡态
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
    ],
    key=["M", "N"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _sum_kernel_2d_atomic(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
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
    x = x.to(tl.float32)

    sum_vals = tl.sum(x, axis=1)

    out_ptrs = out_ptr + m_offsets
    tl.atomic_add(out_ptrs, sum_vals, mask=m_mask)


@triton.autotune(
    configs=[
        # 1. 针对 N 极大，M 极小的极端场景 (Reduce at end)
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 8192}, num_warps=8),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 4096}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 2048}, num_warps=8),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 1024}, num_warps=8),
        # 2. 针对 M 极大，N 极小的极端场景 (例如 dim=0 的情况)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=4),
        triton.Config({"BLOCK_M": 1024, "BLOCK_N": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 2048, "BLOCK_N": 4}, num_warps=8),
        # 3. 增加高 Warp 数（8 warps）加强版，提升带宽高负载时的吞吐
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8),
        # 4. 常规均衡态
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4),
    ],
    key=["M", "N"],
    reset_to_zero=["out_ptr"],
)
@triton.jit
def _sum_kernel_3d_atomic(
    x_ptr,
    out_ptr,
    M,
    N,
    I,
    stride_xo,
    stride_xr,
    stride_xi,
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
    x = x.to(tl.float32)

    sum_vals = tl.sum(x, axis=1)

    out_ptrs = out_ptr + m_offsets
    tl.atomic_add(out_ptrs, sum_vals, mask=m_mask)


def sum(
    input: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN SUM")
    target_dtype = (
        dtype
        if dtype is not None
        else (torch.int64 if is_integral_dtype(input.dtype) else input.dtype)
    )

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
        torch.float64 if input.dtype == torch.float64 else torch.float32
    )

    is_reduce_at_end = dims == list(range(ndim - len(dims), ndim))

    is_consecutive = True
    for j in range(len(dims) - 1):
        if dims[j] + 1 != dims[j + 1]:
            is_consecutive = False
            break

    def _launch_kernel(M, N, input_view, is_3d=False, I_dim=1):
        if M == 0 or N == 0:
            return torch.zeros(
                out_shape, dtype=target_dtype, device=input.device
            )

        is_fp64 = input_view.dtype == torch.float64

        # prefer_transpose_3d = (
        #     is_3d
        #     and input_view.stride(2) == 1
        #     and input_view.stride(1) > 1
        #     and I_dim >= 1024
        #     and N <= 512
        # )
        prefer_transpose_3d = False

        def _should_use_atomic(M: int, N: int, dtype: torch.dtype) -> bool:
            # atomic 只适合：输出行非常少，但每行要归约的长度非常大
            # 例如 dim=None / 最后一维超大归约
            if dtype == torch.float64:
                return False
            return (M <= 16) and (N >= 8192)

        # M 较小 且 N 比较大时使用 Atomic 以保证并发
        use_atomic_fp64 = is_fp64 and (M <= 32) and (N >= 4096)
        use_atomic = (not is_fp64) and _should_use_atomic(
            M, N, input_view.dtype
        )

        out_buffer = (
            torch.zeros((M,), dtype=acc_dtype, device=input.device)
            if (use_atomic or use_atomic_fp64)
            else torch.empty((M,), dtype=acc_dtype, device=input.device)
        )

        if is_fp64:
            if not is_3d:
                if use_atomic_fp64:

                    def grid_2d_atomic_fp64(meta):
                        return (
                            triton.cdiv(M, meta["BLOCK_M"])
                            * triton.cdiv(N, meta["BLOCK_N"]),
                        )

                    _sum_kernel_2d_atomic_fp64[grid_2d_atomic_fp64](
                        input_view,
                        out_buffer,
                        M,
                        N,
                        input_view.stride(0),
                        input_view.stride(1),
                    )
                else:

                    def grid_2d_loop_fp64(meta):
                        return (triton.cdiv(M, meta["BLOCK_M"]),)

                    _sum_kernel_2d_loop_fp64[grid_2d_loop_fp64](
                        input_view,
                        out_buffer,
                        M,
                        N,
                        input_view.stride(0),
                        input_view.stride(1),
                    )
            else:
                if use_atomic_fp64:

                    def grid_3d_atomic_fp64(meta):
                        return (
                            triton.cdiv(M, meta["BLOCK_M"])
                            * triton.cdiv(N, meta["BLOCK_N"]),
                        )

                    _sum_kernel_3d_atomic_fp64[grid_3d_atomic_fp64](
                        input_view,
                        out_buffer,
                        M,
                        N,
                        I_dim,
                        input_view.stride(0),
                        input_view.stride(1),
                        input_view.stride(2),
                    )
                else:

                    def grid_3d_loop_fp64(meta):
                        return (triton.cdiv(M, meta["BLOCK_M"]),)

                    _sum_kernel_3d_loop_fp64[grid_3d_loop_fp64](
                        input_view,
                        out_buffer,
                        M,
                        N,
                        I_dim,
                        input_view.stride(0),
                        input_view.stride(1),
                        input_view.stride(2),
                    )

            return out_buffer
        else:
            if not is_3d:
                if use_atomic:

                    def grid_2d_atomic(meta):
                        return (
                            triton.cdiv(M, meta["BLOCK_M"])
                            * triton.cdiv(N, meta["BLOCK_N"]),
                        )

                    _sum_kernel_2d_atomic[grid_2d_atomic](
                        input_view,
                        out_buffer,
                        M,
                        N,
                        input_view.stride(0),
                        input_view.stride(1),
                    )
                else:

                    def grid_2d_loop(meta):
                        return (triton.cdiv(M, meta["BLOCK_M"]),)

                    _sum_kernel_2d_loop[grid_2d_loop](
                        input_view,
                        out_buffer,
                        M,
                        N,
                        input_view.stride(0),
                        input_view.stride(1),
                    )
            else:
                if use_atomic:

                    def grid_3d_atomic(meta):
                        return (
                            triton.cdiv(M, meta["BLOCK_M"])
                            * triton.cdiv(N, meta["BLOCK_N"]),
                        )

                    _sum_kernel_3d_atomic[grid_3d_atomic](
                        input_view,
                        out_buffer,
                        M,
                        N,
                        I_dim,
                        input_view.stride(0),
                        input_view.stride(1),
                        input_view.stride(2),
                    )
                else:
                    if prefer_transpose_3d:

                        def grid_3d_loop_t(meta):
                            return (triton.cdiv(M, meta["BLOCK_M"]),)

                        _sum_kernel_3d_loop_transpose[grid_3d_loop_t](
                            input_view,
                            out_buffer,
                            M,
                            N,
                            I_dim,
                            input_view.stride(0),
                            input_view.stride(1),
                            input_view.stride(2),
                        )
                    else:

                        def grid_3d_loop(meta):
                            return (triton.cdiv(M, meta["BLOCK_M"]),)

                        _sum_kernel_3d_loop[grid_3d_loop](
                            input_view,
                            out_buffer,
                            M,
                            N,
                            I_dim,
                            input_view.stride(0),
                            input_view.stride(1),
                            input_view.stride(2),
                        )

            return out_buffer

    with torch_device_fn.device(input.device):
        if is_reduce_at_end:
            M, N = 1, 1
            for i in range(ndim - len(dims)):
                M *= input.shape[i]
            for i in range(ndim - len(dims), ndim):
                N *= input.shape[i]
            out_buffer = _launch_kernel(M, N, input.reshape(M, N), is_3d=False)
        elif is_consecutive:
            dim_min, dim_max = dims[0], dims[-1]
            O, R, I_dim = 1, 1, 1
            for j in range(dim_min):
                O *= input.shape[j]
            for j in range(dim_min, dim_max + 1):
                R *= input.shape[j]
            for j in range(dim_max + 1, ndim):
                I_dim *= input.shape[j]
            M, N = O * I_dim, R
            out_buffer = _launch_kernel(
                M, N, input.reshape(O, R, I_dim), is_3d=True, I_dim=I_dim
            )
        else:
            kept_dims = [i for i in range(ndim) if i not in dims]
            M, N = 1, 1
            for d in kept_dims:
                M *= input.shape[d]
            for d in dims:
                N *= input.shape[d]
            input_view = input.permute(*kept_dims, *dims).reshape(M, N)
            out_buffer = _launch_kernel(M, N, input_view, is_3d=False)

    out = out_buffer.to(target_dtype).reshape(out_shape)
    return out
