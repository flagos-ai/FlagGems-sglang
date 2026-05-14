import logging
import warnings
from math import prod
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_SOFTMIN_CONFIGS = runtime.get_tuned_config("softmin")
_SOFTMIN_FP64_CONFIGS = runtime.get_tuned_config("softmin_fp64")


def _get_softmax_dim(name: str, ndim: int, stacklevel: int) -> int:
    warnings.warn(
        f"Implicit dimension choice for {name} has been deprecated. "
        "Change the call to include dim=X as an argument.",
        stacklevel=stacklevel,
    )
    if ndim == 0 or ndim == 1 or ndim == 3:
        return 0
    return 1


def _choose_num_warps(block_size: int) -> int:
    if block_size <= 32:
        return 1
    if block_size <= 64:
        return 2
    if block_size <= 256:
        return 4
    if block_size <= 1024:
        return 8
    return 16


def _choose_num_stages(block_size: int) -> int:
    if block_size <= 256:
        return 2
    if block_size <= 1024:
        return 3
    return 4


def _choose_rows_per_program(n: int) -> int:
    if n <= 4:
        return 256
    if n <= 8:
        return 128
    if n <= 16:
        return 64
    if n <= 32:
        return 32
    if n <= 64:
        return 16
    if n <= 128:
        return 8
    return 2


# -----------------------------------------------------------------------------
# 常规 fast path：last-dim contiguous，一行一个 program
# -----------------------------------------------------------------------------
@triton.jit
def fast_softmin_kernel(
    x_ptr,
    y_ptr,
    N,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # softmin(x) = softmax(-x)
    # 越界填 +inf，取负后为 -inf，不影响 max/exp/sum
    x = tl.load(row_x_ptr + offsets, mask=mask, other=float("inf"))
    x_fp32 = (-x).to(tl.float32)

    row_max = tl.max(x_fp32, axis=0)
    numerator = tl.exp(x_fp32 - row_max)
    denominator = tl.sum(numerator, axis=0)

    out = numerator / denominator
    out = out.to(x.dtype)

    tl.store(row_y_ptr + offsets, out, mask=mask)


@triton.jit
def fast_softmin_fp64_kernel(
    x_ptr,
    y_ptr,
    N,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(row_x_ptr + offsets, mask=mask, other=float("inf"))
    x_fp64 = (-x).to(tl.float64)

    row_max = tl.max(x_fp64, axis=0)
    numerator = tl.exp(x_fp64 - row_max)
    denominator = tl.sum(numerator, axis=0)

    out = numerator / denominator
    out = out.to(x.dtype)

    tl.store(row_y_ptr + offsets, out, mask=mask)


# -----------------------------------------------------------------------------
# 常规 online path：last-dim contiguous，大 N
# -----------------------------------------------------------------------------
@libentry()
@libtuner(
    configs=_SOFTMIN_CONFIGS,
    key=["N"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def online_softmin_kernel(
    x_ptr,
    y_ptr,
    N,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    d_i = tl.full((), 0.0, dtype=tl.float32)

    ptrs = row_x_ptr + tl.arange(0, BLOCK_SIZE)

    for offset in tl.range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(ptrs, mask=mask, other=float("inf"))
        x_fp32 = (-x).to(tl.float32)

        m_block = tl.max(x_fp32, axis=0)
        m_new = tl.maximum(m_i, m_block)

        alpha = tl.exp(m_i - m_new)
        exp_vals = tl.where(mask, tl.exp(x_fp32 - m_new), 0.0)
        d_block = tl.sum(exp_vals, axis=0)

        d_i = d_i * alpha + d_block
        m_i = m_new

        ptrs += BLOCK_SIZE

    for offset in tl.range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(row_x_ptr + cols, mask=mask, other=float("inf"))
        x_fp32 = (-x).to(tl.float32)

        out = tl.exp(x_fp32 - m_i) / d_i
        out = out.to(x_ptr.dtype.element_ty)

        tl.store(row_y_ptr + cols, out, mask=mask)


@libentry()
@libtuner(
    configs=_SOFTMIN_FP64_CONFIGS,
    key=["N"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def online_softmin_fp64_kernel(
    x_ptr,
    y_ptr,
    N,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)

    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    m_i = tl.full((), -float("inf"), dtype=tl.float64)
    d_i = tl.full((), 0.0, dtype=tl.float64)

    ptrs = row_x_ptr + tl.arange(0, BLOCK_SIZE)

    for offset in tl.range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(ptrs, mask=mask, other=float("inf"))
        x_fp64 = (-x).to(tl.float64)

        m_block = tl.max(x_fp64, axis=0)
        m_new = tl.maximum(m_i, m_block)

        alpha = tl.exp(m_i - m_new)
        exp_vals = tl.where(mask, tl.exp(x_fp64 - m_new), 0.0)
        d_block = tl.sum(exp_vals, axis=0)

        d_i = d_i * alpha + d_block
        m_i = m_new

        ptrs += BLOCK_SIZE

    for offset in tl.range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(row_x_ptr + cols, mask=mask, other=float("inf"))
        x_fp64 = (-x).to(tl.float64)

        out = tl.exp(x_fp64 - m_i) / d_i
        out = out.to(x_ptr.dtype.element_ty)

        tl.store(row_y_ptr + cols, out, mask=mask)


# -----------------------------------------------------------------------------
# tiny contiguous path：last-dim contiguous，N 很小，一个 program 处理多行
# -----------------------------------------------------------------------------
@triton.jit
def tiny_softmin_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tle.program_id(0)

    row_ids = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    col_ids = tl.arange(0, BLOCK_SIZE)

    mask = (row_ids[:, None] < M) & (col_ids[None, :] < N)
    ptrs = x_ptr + row_ids[:, None] * stride_row + col_ids[None, :]

    x = tl.load(ptrs, mask=mask, other=float("inf")).to(tl.float32)
    x = -x

    row_max = tl.max(x, axis=1)
    num = tl.exp(x - row_max[:, None])
    den = tl.sum(num, axis=1)

    out = (num / den[:, None]).to(y_ptr.dtype.element_ty)
    tl.store(
        y_ptr + row_ids[:, None] * stride_row + col_ids[None, :],
        out,
        mask=mask,
    )


@triton.jit
def tiny_softmin_fp64_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tle.program_id(0)

    row_ids = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    col_ids = tl.arange(0, BLOCK_SIZE)

    mask = (row_ids[:, None] < M) & (col_ids[None, :] < N)
    ptrs = x_ptr + row_ids[:, None] * stride_row + col_ids[None, :]

    x = tl.load(ptrs, mask=mask, other=float("inf")).to(tl.float64)
    x = -x

    row_max = tl.max(x, axis=1)
    num = tl.exp(x - row_max[:, None])
    den = tl.sum(num, axis=1)

    out = (num / den[:, None]).to(y_ptr.dtype.element_ty)
    tl.store(
        y_ptr + row_ids[:, None] * stride_row + col_ids[None, :],
        out,
        mask=mask,
    )


# -----------------------------------------------------------------------------
# tiny strided path：非 last-dim，N 很小，一个 program 处理多行
# 避免 transpose(...).contiguous() 的大拷贝
# -----------------------------------------------------------------------------
@triton.jit
def tiny_softmin_strided_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    INNER_SIZE,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tle.program_id(0)

    row_ids = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    col_ids = tl.arange(0, BLOCK_SIZE)

    valid_rows = row_ids < M
    outer_ids = row_ids // INNER_SIZE
    inner_ids = row_ids % INNER_SIZE

    row_bases = outer_ids * N * INNER_SIZE + inner_ids

    mask = valid_rows[:, None] & (col_ids[None, :] < N)
    ptrs = x_ptr + row_bases[:, None] + col_ids[None, :] * INNER_SIZE

    x = tl.load(ptrs, mask=mask, other=float("inf")).to(tl.float32)
    x = -x

    row_max = tl.max(x, axis=1)
    num = tl.exp(x - row_max[:, None])
    den = tl.sum(num, axis=1)

    out = (num / den[:, None]).to(y_ptr.dtype.element_ty)
    tl.store(
        y_ptr + row_bases[:, None] + col_ids[None, :] * INNER_SIZE,
        out,
        mask=mask,
    )


@triton.jit
def tiny_softmin_strided_fp64_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    INNER_SIZE,
    BLOCK_SIZE: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tle.program_id(0)

    row_ids = pid * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    col_ids = tl.arange(0, BLOCK_SIZE)

    valid_rows = row_ids < M
    outer_ids = row_ids // INNER_SIZE
    inner_ids = row_ids % INNER_SIZE

    row_bases = outer_ids * N * INNER_SIZE + inner_ids

    mask = valid_rows[:, None] & (col_ids[None, :] < N)
    ptrs = x_ptr + row_bases[:, None] + col_ids[None, :] * INNER_SIZE

    x = tl.load(ptrs, mask=mask, other=float("inf")).to(tl.float64)
    x = -x

    row_max = tl.max(x, axis=1)
    num = tl.exp(x - row_max[:, None])
    den = tl.sum(num, axis=1)

    out = (num / den[:, None]).to(y_ptr.dtype.element_ty)
    tl.store(
        y_ptr + row_bases[:, None] + col_ids[None, :] * INNER_SIZE,
        out,
        mask=mask,
    )


def softmin(
    input: torch.Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN SOFTMIN (dim={dim}, dtype={dtype})")

    x = input
    if dtype is not None:
        x = x.to(dtype)

    if x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn softmin does not support dtype={x.dtype}"
        )

    if x.dtype == torch.float64 and not runtime.device.support_fp64:
        raise RuntimeError("Device does not support float64")

    if x.numel() == 0:
        return torch.empty_like(x)

    if x.ndim == 0:
        return torch.ones_like(x)

    if dim is None:
        dim = _get_softmax_dim("softmin", x.dim(), _stacklevel)

    if dim < 0:
        dim += x.ndim

    if dim < 0 or dim >= x.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[-{x.ndim}, {x.ndim - 1}], but got {dim})"
        )

    if not x.is_contiguous():
        x = x.contiguous()

    n_orig = x.shape[dim]

    y = torch.empty_like(x)

    # -------------------------------------------------------------------------
    # 路径1：dim != last 且 N 很小
    # 直接在原 contiguous tensor 上做 strided softmin，避免 transpose+contiguous
    # -------------------------------------------------------------------------
    use_strided_tiny = (
        dim != x.ndim - 1 and n_orig <= 64 and x.numel() < 1024 * 1024
    )
    if use_strided_tiny:
        outer = prod(x.shape[:dim]) if dim > 0 else 1
        inner = prod(x.shape[dim + 1 :]) if dim < x.ndim - 1 else 1
        m = outer * inner

        block_size = triton.next_power_of_2(n_orig)
        rows_per_program = _choose_rows_per_program(n_orig)
        num_warps = _choose_num_warps(block_size * rows_per_program)
        num_stages = _choose_num_stages(block_size)

        grid = (triton.cdiv(m, rows_per_program),)

        with torch_device_fn.device(x.device):
            if x.dtype == torch.float64:
                tiny_softmin_strided_fp64_kernel[grid](
                    x,
                    y,
                    m,
                    n_orig,
                    inner,
                    BLOCK_SIZE=block_size,
                    ROWS_PER_PROGRAM=rows_per_program,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            else:
                tiny_softmin_strided_kernel[grid](
                    x,
                    y,
                    m,
                    n_orig,
                    inner,
                    BLOCK_SIZE=block_size,
                    ROWS_PER_PROGRAM=rows_per_program,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
        return y

    # -------------------------------------------------------------------------
    # 路径2：dim == last 且 N 很小
    # 一个 program 处理多行，降低小 N 的 launch / 调度开销
    # -------------------------------------------------------------------------
    if dim == x.ndim - 1 and n_orig <= 128:
        m = x.numel() // n_orig

        block_size = triton.next_power_of_2(n_orig)
        rows_per_program = _choose_rows_per_program(n_orig)
        num_warps = _choose_num_warps(block_size * rows_per_program)
        num_stages = _choose_num_stages(block_size)

        grid = (triton.cdiv(m, rows_per_program),)

        with torch_device_fn.device(x.device):
            if x.dtype == torch.float64:
                tiny_softmin_fp64_kernel[grid](
                    x,
                    y,
                    m,
                    n_orig,
                    n_orig,
                    BLOCK_SIZE=block_size,
                    ROWS_PER_PROGRAM=rows_per_program,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            else:
                tiny_softmin_kernel[grid](
                    x,
                    y,
                    m,
                    n_orig,
                    n_orig,
                    BLOCK_SIZE=block_size,
                    ROWS_PER_PROGRAM=rows_per_program,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
        return y

    # -------------------------------------------------------------------------
    # 路径3：其他情况，继续复用“转到最后一维”的常规实现
    # -------------------------------------------------------------------------
    need_transpose = dim != x.ndim - 1
    if need_transpose:
        x_work = x.transpose(dim, -1).contiguous()
    else:
        x_work = x

    n = x_work.shape[-1]
    m = x_work.numel() // n
    y_work = torch.empty_like(x_work)

    grid = (m,)
    max_fast_block = 1024

    with torch_device_fn.device(x_work.device):
        if n <= max_fast_block:
            block_size = triton.next_power_of_2(n)
            num_warps = _choose_num_warps(block_size)
            num_stages = _choose_num_stages(block_size)

            if x_work.dtype == torch.float64:
                fast_softmin_fp64_kernel[grid](
                    x_work,
                    y_work,
                    n,
                    n,
                    n,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            else:
                fast_softmin_kernel[grid](
                    x_work,
                    y_work,
                    n,
                    n,
                    n,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
        else:
            if x_work.dtype == torch.float64:
                online_softmin_fp64_kernel[grid](
                    x_work,
                    y_work,
                    n,
                    n,
                    n,
                )
            else:
                online_softmin_kernel[grid](
                    x_work,
                    y_work,
                    n,
                    n,
                    n,
                )

    if need_transpose:
        return y_work.transpose(dim, -1).contiguous()
    return y_work
