import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_MV_CONFIGS = runtime.get_tuned_config("mv")
_MV_FP64_CONFIGS = runtime.get_tuned_config("mv_fp64")
_MV_WIDE_CONFIGS = runtime.get_tuned_config("mv_wide")
_MV_WIDE_FP64_CONFIGS = runtime.get_tuned_config("mv_wide_fp64")


_SUPPORTED_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def _choose_num_warps(block_n: int) -> int:
    if block_n <= 32:
        return 1
    if block_n <= 64:
        return 2
    if block_n <= 256:
        return 4
    if block_n <= 1024:
        return 8
    return 16


def _choose_num_stages(block_n: int) -> int:
    if block_n <= 256:
        return 2
    if block_n <= 1024:
        return 3
    return 4


def _choose_tiny_block_m(n: int) -> int:
    if n <= 8:
        return 64
    if n <= 16:
        return 32
    if n <= 32:
        return 16
    if n <= 64:
        return 8
    if n <= 128:
        return 4
    return 2


def _should_use_wide_fp64(M: int, N: int) -> bool:
    return M <= 512 and N >= 4096


def _should_use_fp64_two_rows(M: int, N: int) -> bool:
    # 专门覆盖：
    # [512, 16384]
    # [1892, 3584]
    # 不走 split-K，不走 rowwise
    return 384 < M <= 2048 and N >= 3072


# -----------------------------------------------------------------------------
# tiny-N path：N 小时，一个 program 处理多行
# -----------------------------------------------------------------------------
@triton.jit
def tiny_mv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_x,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)

    row_mask = row_offsets < M
    col_mask = col_offsets < N
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(
        x_ptr + col_offsets * stride_x,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    a = tl.load(
        a_ptr
        + row_offsets[:, None] * stride_am
        + col_offsets[None, :] * stride_an,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(a * x[None, :], axis=1)
    out = acc.to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + row_offsets * stride_y, out, mask=row_mask)


@triton.jit
def tiny_mv_fp64_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_x,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = tl.arange(0, BLOCK_N)

    row_mask = row_offsets < M
    col_mask = col_offsets < N
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(
        x_ptr + col_offsets * stride_x,
        mask=col_mask,
        other=0.0,
    ).to(tl.float64)

    a = tl.load(
        a_ptr
        + row_offsets[:, None] * stride_am
        + col_offsets[None, :] * stride_an,
        mask=mask,
        other=0.0,
    ).to(tl.float64)

    acc = tl.sum(a * x[None, :], axis=1)
    out = acc.to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + row_offsets * stride_y, out, mask=row_mask)


# -----------------------------------------------------------------------------
# wide path：very-small-M + huge-N
# 一行一个 program
# -----------------------------------------------------------------------------
@libentry()
@libtuner(
    configs=_MV_WIDE_CONFIGS,
    key=["M", "N"],
    strategy=["align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def wide_mv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_x,
    stride_y,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)
    row = pid

    if row >= M:
        return

    acc = tl.zeros((), dtype=tl.float32)

    for n_start in tl.range(0, N, BLOCK_N):
        col_offsets = n_start + tl.arange(0, BLOCK_N)
        col_mask = col_offsets < N

        x = tl.load(
            x_ptr + col_offsets * stride_x,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)

        a = tl.load(
            a_ptr + row * stride_am + col_offsets * stride_an,
            mask=col_mask,
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(a * x, axis=0)

    out = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * stride_y, out)


@libentry()
@libtuner(
    configs=_MV_WIDE_FP64_CONFIGS,
    key=["M", "N"],
    strategy=["align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def wide_mv_fp64_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_x,
    stride_y,
    BLOCK_N: tl.constexpr,
):
    row = tle.program_id(0)
    if row >= M:
        return

    # 预计算行指针，减少循环内乘法
    a_row_ptr = a_ptr + row * stride_am
    acc = tl.zeros((), dtype=tl.float64)

    for n_start in tl.range(0, N, BLOCK_N):
        col_offsets = n_start + tl.arange(0, BLOCK_N)
        mask = col_offsets < N

        # 1D 加载减少了 2D 掩码的寄存器压力
        x = tl.load(x_ptr + col_offsets * stride_x, mask=mask, other=0.0).to(
            tl.float64
        )
        a = tl.load(
            a_row_ptr + col_offsets * stride_an, mask=mask, other=0.0
        ).to(tl.float64)

        acc += tl.sum(a * x, axis=0)

    tl.store(y_ptr + row * stride_y, acc.to(y_ptr.dtype.element_ty))


# -----------------------------------------------------------------------------
# general path
# -----------------------------------------------------------------------------
@libentry()
@libtuner(
    configs=_MV_CONFIGS,
    key=["M", "N"],
    strategy=["align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def mv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_x,
    stride_y,  # 引入 stride
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    if BLOCK_M == 1:
        row = pid
        if row >= M:
            return

        a_row_ptr = a_ptr + row * stride_am
        acc = tl.zeros((), dtype=tl.float32)

        # 1D 循环优化
        for n_start in tl.range(0, N, BLOCK_N, num_stages=3):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            # 使用 1D 加载，减少寄存器压力
            a = tl.load(a_row_ptr + offs_n * stride_an, mask=mask_n, other=0.0)
            x = tl.load(x_ptr + offs_n * stride_x, mask=mask_n, other=0.0)
            acc += tl.sum(a * x, axis=0)

        tl.store(y_ptr + row * stride_y, acc.to(y_ptr.dtype.element_ty))
    else:
        # 通用多行路径
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for n_start in tl.range(0, N, BLOCK_N, num_stages=2):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            x = tl.load(x_ptr + offs_n * stride_x, mask=mask_n, other=0.0)
            # 仅在行方向有 mask，列方向动态计算
            a = tl.load(
                a_ptr
                + offs_m[:, None] * stride_am
                + offs_n[None, :] * stride_an,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.sum(a * x[None, :], axis=1)

        tl.store(
            y_ptr + offs_m * stride_y,
            acc.to(y_ptr.dtype.element_ty),
            mask=mask_m,
        )


@libentry()
@libtuner(
    configs=_MV_FP64_CONFIGS,
    key=["M", "N"],
    strategy=["align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def mv_fp64_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_x,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tle.program_id(0)

    if BLOCK_M == 1:
        row = pid
        if row >= M:
            return

        # 优化点：直接计算行基址
        a_row_ptr = a_ptr + row * stride_am
        acc = tl.zeros((), dtype=tl.float64)

        # 优化点：如果 N <= BLOCK_N，编译器会完全展开循环消除判断
        for n_start in tl.range(0, N, BLOCK_N):
            offsets = n_start + tl.arange(0, BLOCK_N)
            mask = offsets < N

            # 移除不必要的 .to(tl.float64)，直接加载
            # 确保 stride_an == 1 时，这里会触发最优的向量化加载
            a = tl.load(a_row_ptr + offsets * stride_an, mask=mask, other=0.0)
            x = tl.load(x_ptr + offsets * stride_x, mask=mask, other=0.0)

            acc += tl.sum(a * x, axis=0)

        tl.store(y_ptr + row * stride_y, acc.to(y_ptr.dtype.element_ty))

    else:
        # 通用路径（针对较大 BLOCK_M）
        row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row_offsets < M
        a_base_ptrs = a_ptr + row_offsets[:, None] * stride_am
        acc = tl.zeros((BLOCK_M,), dtype=tl.float64)

        for n_start in tl.range(0, N, BLOCK_N):
            col_offsets = n_start + tl.arange(0, BLOCK_N)
            col_mask = col_offsets < N

            x = tl.load(
                x_ptr + col_offsets * stride_x, mask=col_mask, other=0.0
            )
            # 使用更精简的 2D 掩码逻辑
            a = tl.load(
                a_base_ptrs + col_offsets[None, :] * stride_an,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            )

            acc += tl.sum(a * x[None, :], axis=1)

        tl.store(
            y_ptr + row_offsets * stride_y,
            acc.to(y_ptr.dtype.element_ty),
            mask=row_mask,
        )


def mv(
    input: torch.Tensor,
    vec: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN MV")

    if input.dim() != 2:
        raise RuntimeError(
            f"mv: expected input to be a matrix, but got {input.dim()}D tensor"
        )
    if vec.dim() != 1:
        raise RuntimeError(
            f"mv: expected vec to be a vector, but got {vec.dim()}D tensor"
        )

    if input.device != vec.device:
        raise RuntimeError(
            "mv: input and vec must be on the same device, "
            f"but got {input.device} and {vec.device}"
        )

    if input.shape[1] != vec.shape[0]:
        raise RuntimeError(
            f"size mismatch, got input ({input.shape[0]}x{input.shape[1]}) "
            f"and vec ({vec.shape[0]})"
        )

    result_dtype = torch.promote_types(input.dtype, vec.dtype)
    if result_dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"flag_dnn mv does not support result dtype={result_dtype}"
        )

    if result_dtype == torch.float64 and not runtime.device.support_fp64:
        raise RuntimeError("Device does not support float64")

    a = input if input.dtype == result_dtype else input.to(result_dtype)
    x = vec if vec.dtype == result_dtype else vec.to(result_dtype)

    if not a.is_contiguous():
        a = a.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()

    M, N = a.shape

    if N == 0:
        if out is None:
            return torch.zeros((M,), dtype=result_dtype, device=a.device)

        if out.device != a.device:
            raise RuntimeError(
                "mv out tensor device mismatch: "
                f"expected {a.device}, got {out.device}"
            )
        if out.dtype != result_dtype:
            raise RuntimeError(
                "mv out tensor dtype mismatch: "
                f"expected {result_dtype}, got {out.dtype}"
            )

        if out.dim() != 1 or out.shape[0] != M:
            out.resize_((M,))
        out.zero_()
        return out

    need_copy_back = False
    if out is None:
        y = torch.empty((M,), dtype=result_dtype, device=a.device)
        out_tensor = y
    else:
        if out.device != a.device:
            raise RuntimeError(
                "mv out tensor device mismatch: "
                f"expected {a.device}, got {out.device}"
            )
        if out.dtype != result_dtype:
            raise RuntimeError(
                "mv out tensor dtype mismatch: "
                f"expected {result_dtype}, got {out.dtype}"
            )

        if out.dim() != 1 or out.shape[0] != M:
            out.resize_((M,))

        if out.is_contiguous():
            y = out
            out_tensor = out
        else:
            y = torch.empty((M,), dtype=result_dtype, device=a.device)
            out_tensor = out
            need_copy_back = True

    stride_am = a.stride(0)
    stride_an = a.stride(1)
    stride_x = x.stride(0)
    stride_y = y.stride(0)

    # 1) tiny-N
    if N <= 256:
        block_n = triton.next_power_of_2(N)
        block_m = _choose_tiny_block_m(N)
        num_warps = _choose_num_warps(block_n)
        num_stages = _choose_num_stages(block_n)

        tiny_grid = (triton.cdiv(M, block_m),)

        with torch_device_fn.device(a.device):
            if result_dtype == torch.float64:
                tiny_mv_fp64_kernel[tiny_grid](
                    a,
                    x,
                    y,
                    M,
                    N,
                    stride_am,
                    stride_an,
                    stride_x,
                    stride_y,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            else:
                tiny_mv_kernel[tiny_grid](
                    a,
                    x,
                    y,
                    M,
                    N,
                    stride_am,
                    stride_an,
                    stride_x,
                    stride_y,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )

    # 2) fp64 wide
    elif result_dtype == torch.float64 and _should_use_wide_fp64(M, N):
        wide_grid = (M,)
        with torch_device_fn.device(a.device):
            wide_mv_fp64_kernel[wide_grid](
                a,
                x,
                y,
                M,
                N,
                stride_am,
                stride_an,
                stride_x,
                stride_y,
            )

    # 3) fp64 two-rows special
    elif result_dtype == torch.float64 and _should_use_fp64_two_rows(M, N):

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        with torch_device_fn.device(a.device):
            mv_fp64_kernel[grid](
                a, x, y, M, N, stride_am, stride_an, stride_x, stride_y
            )

    # 4) 非 fp64 wide
    elif result_dtype != torch.float64 and M <= 1024 and N >= 4096:
        wide_grid = (M,)
        with torch_device_fn.device(a.device):
            wide_mv_kernel[wide_grid](
                a,
                x,
                y,
                M,
                N,
                stride_am,
                stride_an,
                stride_x,
                stride_y,
            )

    # 5) general
    else:

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_M"]),)

        with torch_device_fn.device(a.device):
            if result_dtype == torch.float64:
                mv_fp64_kernel[grid](
                    a,
                    x,
                    y,
                    M,
                    N,
                    stride_am,
                    stride_an,
                    stride_x,
                    stride_y,
                )
            else:
                mv_kernel[grid](
                    a,
                    x,
                    y,
                    M,
                    N,
                    stride_am,
                    stride_an,
                    stride_x,
                    stride_y,
                )

    if need_copy_back:
        out_tensor.copy_(y)

    return out_tensor
