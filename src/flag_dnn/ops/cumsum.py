import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle
from flag_dnn.utils.type_utils import is_integral_dtype


logger = logging.getLogger(__name__)


@triton.jit
def cumsum_global_pass1(in_ptr, out_ptr, sums_ptr, N, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    cols = pid * BLOCK_N + offs
    mask = cols < N

    val = tl.load(in_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    c_sum = tl.cumsum(val, axis=0)

    tl.store(out_ptr + cols, c_sum.to(out_ptr.dtype.element_ty), mask=mask)

    if sums_ptr is not None:
        last_mask = offs == (BLOCK_N - 1)
        block_total = tl.sum(tl.where(last_mask, c_sum, 0.0), axis=0)
        tl.store(sums_ptr + pid, block_total)


@triton.jit
def cumsum_global_pass2(sums_ptr, num_blocks, BLOCK_N: tl.constexpr):
    offs = tl.arange(0, BLOCK_N)
    running_sum = 0.0

    for i in range(0, num_blocks, BLOCK_N):
        idx = i + offs
        mask = idx < num_blocks

        val = tl.load(sums_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        c_sum = tl.cumsum(val, axis=0) + running_sum
        tl.store(sums_ptr + idx, c_sum, mask=mask)

        # Triton 不支持 c_sum[BLOCK_N - 1]
        # 改成 one-hot mask + reduce 取最后一个元素
        last_mask = offs == (BLOCK_N - 1)
        running_sum = tl.sum(tl.where(last_mask, c_sum, 0.0), axis=0)


@triton.jit
def cumsum_global_pass3(out_ptr, sums_ptr, N, BLOCK_N: tl.constexpr):
    pid = tle.program_id(0)

    if pid > 0:
        add_val = tl.load(sums_ptr + pid - 1).to(tl.float32)
        cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N

        val = tl.load(out_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        val += add_val

        tl.store(out_ptr + cols, val.to(out_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M_POST": 16, "BLOCK_N": 256}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M_POST": 32, "BLOCK_N": 128}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M_POST": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M_POST": 128, "BLOCK_N": 32}, num_warps=4, num_stages=5
        ),
        triton.Config(
            {"BLOCK_M_POST": 8, "BLOCK_N": 512}, num_warps=8, num_stages=3
        ),
    ],
    key=["M_pre", "N", "M_post"],
)
@triton.jit
def cumsum_inner_dim_kernel(
    in_ptr,
    out_ptr,
    M_pre,
    N,
    M_post,
    BLOCK_N: tl.constexpr,
    BLOCK_M_POST: tl.constexpr,
):
    pid_pre = tle.program_id(0)
    pid_post = tle.program_id(1)

    offsets_post = pid_post * BLOCK_M_POST + tl.arange(0, BLOCK_M_POST)
    mask_post = offsets_post < M_post

    base_idx = pid_pre * (N * M_post) + offsets_post
    running_sum = tl.zeros([BLOCK_M_POST], dtype=tl.float32)
    offsets_n = tl.arange(0, BLOCK_N)

    for n_offset in range(0, N, BLOCK_N):
        cols = n_offset + offsets_n
        mask_n = cols < N

        idx = base_idx[None, :] + cols[:, None] * M_post
        mask = mask_n[:, None] & mask_post[None, :]

        val = tl.load(in_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        chunk_cumsum = tl.cumsum(val, axis=0)
        out_val = chunk_cumsum + running_sum[None, :]

        tl.store(
            out_ptr + idx, out_val.to(out_ptr.dtype.element_ty), mask=mask
        )

        # Triton 不支持 out_val[BLOCK_N - 1, :]
        # 用 one-hot 选最后一行，再沿 axis=0 reduce
        last_row_mask = offsets_n[:, None] == (BLOCK_N - 1)
        running_sum = tl.sum(
            tl.where(last_row_mask, out_val, 0.0),
            axis=0,
        )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M_PRE": 4, "BLOCK_N": 4096}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M_PRE": 8, "BLOCK_N": 2048}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M_PRE": 16, "BLOCK_N": 1024}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_M_PRE": 32, "BLOCK_N": 512}, num_warps=4, num_stages=5
        ),
        triton.Config(
            {"BLOCK_M_PRE": 64, "BLOCK_N": 256}, num_warps=4, num_stages=5
        ),
    ],
    key=["M_pre", "N"],
)
@triton.jit
def cumsum_last_dim_kernel(
    in_ptr, out_ptr, M_pre, N, BLOCK_M_PRE: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_pre = tle.program_id(0)
    offsets_pre = pid_pre * BLOCK_M_PRE + tl.arange(0, BLOCK_M_PRE)
    mask_pre = offsets_pre < M_pre

    running_sum = tl.zeros([BLOCK_M_PRE], dtype=tl.float32)
    offsets_n = tl.arange(0, BLOCK_N)

    for n_offset in range(0, N, BLOCK_N):
        cols = n_offset + offsets_n
        mask_n = cols < N

        idx = cols[:, None] + offsets_pre[None, :] * N
        mask = mask_n[:, None] & mask_pre[None, :]

        val = tl.load(in_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        chunk_cumsum = tl.cumsum(val, axis=0)
        out_val = chunk_cumsum + running_sum[None, :]

        tl.store(
            out_ptr + idx, out_val.to(out_ptr.dtype.element_ty), mask=mask
        )

        # Triton 不支持 out_val[BLOCK_N - 1, :]
        # 用 one-hot 选最后一行，再沿 axis=0 reduce
        last_row_mask = offsets_n[:, None] == (BLOCK_N - 1)
        running_sum = tl.sum(
            tl.where(last_row_mask, out_val, 0.0),
            axis=0,
        )


def cumsum(
    input: torch.Tensor,
    dim: int,
    *,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("FLAG_DNN CUMSUM OPTIMIZED")

    out_dtype = (
        dtype
        if dtype is not None
        else (torch.int64 if is_integral_dtype(input.dtype) else input.dtype)
    )

    if input.numel() == 0:
        if out is not None:
            if out.shape != input.shape:
                out.resize_(input.shape)
            return out
        return torch.empty_like(input, dtype=out_dtype)

    d = dim if dim >= 0 else dim + input.ndim
    if not (0 <= d < input.ndim):
        raise IndexError(f"Dimension out of range: {dim}")

    input_c = input.contiguous()

    if out is not None:
        if out.shape != input_c.shape:
            out.resize_(input_c.shape)
        out_c = (
            out.contiguous()
            if out.is_contiguous()
            else torch.empty_like(input_c, dtype=out_dtype)
        )
    else:
        out_c = torch.empty_like(input_c, dtype=out_dtype)

    M_pre, N, M_post = 1, input_c.shape[d], 1
    for i in range(d):
        M_pre *= input_c.shape[i]
    for i in range(d + 1, input_c.ndim):
        M_post *= input_c.shape[i]

    with torch_device_fn.device(input.device):
        if M_post == 1:
            # 最后一维扫描
            if M_pre == 1 and N > 65536:
                # 3-pass 全局并行扫描
                if input.dtype in (torch.float16, torch.bfloat16):
                    block_n_pass1_3 = 4096
                else:
                    block_n_pass1_3 = 8192

                grid_size = triton.cdiv(N, block_n_pass1_3)
                sums = torch.empty(
                    (grid_size,), dtype=torch.float32, device=input.device
                )

                cumsum_global_pass1[(grid_size,)](
                    input_c,
                    out_c,
                    sums,
                    N,
                    BLOCK_N=block_n_pass1_3,
                )
                cumsum_global_pass2[(1,)](
                    sums,
                    grid_size,
                    BLOCK_N=4096,
                )
                cumsum_global_pass3[(grid_size,)](
                    out_c,
                    sums,
                    N,
                    BLOCK_N=block_n_pass1_3,
                )
            else:

                def grid(meta):
                    return (triton.cdiv(M_pre, meta["BLOCK_M_PRE"]),)

                cumsum_last_dim_kernel[grid](
                    input_c,
                    out_c,
                    M_pre,
                    N,
                )
        else:

            def grid(meta):
                return (
                    M_pre,
                    triton.cdiv(M_post, meta["BLOCK_M_POST"]),
                )

            cumsum_inner_dim_kernel[grid](
                input_c,
                out_c,
                M_pre,
                N,
                M_post,
            )

    if out is not None:
        if out.data_ptr() != out_c.data_ptr():
            out.copy_(out_c)
        return out

    return out_c.view(input.shape)
