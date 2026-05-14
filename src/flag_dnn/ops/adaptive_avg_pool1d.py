import logging
import math
from typing import Tuple, Union, Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


def _exact_max_window(in_size: int, out_size: int) -> int:
    gcd_val = math.gcd(in_size, out_size)
    max_r = out_size - gcd_val
    return (max_r + in_size + out_size - 1) // out_size


def _small_ow_meta(ow: int):
    if ow <= 8:
        return 8, 1
    return 16, 1


# OW == 1
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_avg_pool1d_global"),
    key=["N", "C", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def global_avg_pool1d_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    W,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    nc = N * C
    if pid_nc >= nc:
        return

    x_base = x_ptr + pid_nc * W
    offs = tl.arange(0, BLOCK_SIZE)

    total = tl.sum(tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE), axis=0)

    for w0 in range(0, W, BLOCK_SIZE):
        idx = w0 + offs
        mask = idx < W
        vals = tl.load(x_base + idx, mask=mask, other=0).to(ACC_DTYPE)
        total += tl.sum(vals, axis=0)

    avg_val = total / W
    tl.store(y_ptr + pid_nc, avg_val.to(y_ptr.dtype.element_ty))


# 小 OW, 固定小 block, divisible
@libentry()
@triton.jit
def adaptive_avg_pool1d_divisible_small_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    W,
    OW,
    K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    nc = N * C
    if pid_nc >= nc:
        return

    ow = tl.arange(0, BLOCK_SIZE)
    ow_mask = ow < OW

    n = pid_nc // C
    c = pid_nc % C

    x_base = x_ptr + (n * C + c) * W
    y_base = y_ptr + (n * C + c) * OW

    start_w = ow * K_W
    acc = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    for kw in range(K_W):
        idx = start_w + kw
        vals = tl.load(x_base + idx, mask=ow_mask, other=0).to(ACC_DTYPE)
        acc += vals

    out = acc / K_W
    tl.store(y_base + ow, out.to(y_ptr.dtype.element_ty), mask=ow_mask)


# 小 OW, 固定小 block, general
@libentry()
@triton.jit
def adaptive_avg_pool1d_general_small_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    W,
    OW,
    MAX_K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    nc = N * C
    if pid_nc >= nc:
        return

    ow = tl.arange(0, BLOCK_SIZE)
    ow_mask = ow < OW

    n = pid_nc // C
    c = pid_nc % C

    x_base = x_ptr + (n * C + c) * W
    y_base = y_ptr + (n * C + c) * OW

    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW

    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)

    pool_size = tl.maximum(end_w - start_w, 1)
    acc = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    for kw in range(MAX_K_W):
        idx = start_w + kw
        mask = ow_mask & (idx < end_w)
        vals = tl.load(x_base + idx, mask=mask, other=0).to(ACC_DTYPE)
        acc += vals

    out = acc / pool_size.to(ACC_DTYPE)
    tl.store(y_base + ow, out.to(y_ptr.dtype.element_ty), mask=ow_mask)


# 大 OW, divisible
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_avg_pool1d_divisible_large"),
    key=["N", "C", "W", "OW"],
    warmup=5,
    rep=10,
)
@triton.jit
def adaptive_avg_pool1d_divisible_large_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    W,
    OW,
    K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    pid_ow = tle.program_id(1)

    nc = N * C
    if pid_nc >= nc:
        return

    ow = pid_ow * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow_mask = ow < OW

    n = pid_nc // C
    c = pid_nc % C

    x_base = x_ptr + (n * C + c) * W
    y_base = y_ptr + (n * C + c) * OW

    start_w = ow * K_W
    acc = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    for kw in range(K_W):
        idx = start_w + kw
        vals = tl.load(x_base + idx, mask=ow_mask, other=0).to(ACC_DTYPE)
        acc += vals

    out = acc / K_W
    tl.store(y_base + ow, out.to(y_ptr.dtype.element_ty), mask=ow_mask)


# 大 OW, general
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_avg_pool1d_general_large"),
    key=["N", "C", "W", "OW"],
    warmup=5,
    rep=10,
)
@triton.jit
def adaptive_avg_pool1d_general_large_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    W,
    OW,
    MAX_K_W: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tle.program_id(0)
    pid_ow = tle.program_id(1)

    nc = N * C
    if pid_nc >= nc:
        return

    ow = pid_ow * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    ow_mask = ow < OW

    n = pid_nc // C
    c = pid_nc % C

    x_base = x_ptr + (n * C + c) * W
    y_base = y_ptr + (n * C + c) * OW

    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW

    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)

    pool_size = tl.maximum(end_w - start_w, 1)
    acc = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    for kw in range(MAX_K_W):
        idx = start_w + kw
        mask = ow_mask & (idx < end_w)
        vals = tl.load(x_base + idx, mask=mask, other=0).to(ACC_DTYPE)
        acc += vals

    out = acc / pool_size.to(ACC_DTYPE)
    tl.store(y_base + ow, out.to(y_ptr.dtype.element_ty), mask=ow_mask)


def adaptive_avg_pool1d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[Optional[int]]],
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN ADAPTIVE_AVG_POOL1D (output_size={output_size})")

    if isinstance(output_size, int):
        OW = output_size
    else:
        OW = output_size[0] if output_size[0] is not None else input.shape[-1]

    assert input.ndim in [2, 3], "Input must be 2D or 3D"
    is_2d = input.ndim == 2
    if is_2d:
        input = input.unsqueeze(0)

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    N, C, W = input.shape
    y = torch.empty((N, C, OW), dtype=input.dtype, device=input.device)

    if N == 0 or C == 0 or OW == 0:
        return y.squeeze(0) if is_2d else y

    acc_dtype = tl.float64 if input.dtype == torch.float64 else tl.float32

    # identity fast path
    if OW == W:
        out = input.clone()
        return out.squeeze(0) if is_2d else out

    with torch_device_fn.device(input.device):
        # global avg
        if OW == 1:

            def grid(meta):
                return (N * C,)

            global_avg_pool1d_kernel[grid](
                input,
                y,
                N,
                C,
                W,
                ACC_DTYPE=acc_dtype,
            )
        # small OW: 直接固定小 block，不做 autotune
        elif OW <= 16:
            block_size, num_warps = _small_ow_meta(OW)

            if (W % OW) == 0:
                k_w = W // OW
                grid = (N * C,)  # type: ignore[assignment]
                adaptive_avg_pool1d_divisible_small_kernel[grid](
                    input,
                    y,
                    N,
                    C,
                    W,
                    OW,
                    K_W=k_w,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                    num_stages=1,
                )
            else:
                max_k_w = _exact_max_window(W, OW)
                grid = (N * C,)  # type: ignore[assignment]
                adaptive_avg_pool1d_general_small_kernel[grid](
                    input,
                    y,
                    N,
                    C,
                    W,
                    OW,
                    MAX_K_W=max_k_w,
                    ACC_DTYPE=acc_dtype,
                    BLOCK_SIZE=block_size,
                    num_warps=num_warps,
                    num_stages=1,
                )
        # OW==32 的 non-divisible
        elif OW == 32 and (W % OW) != 0:
            max_k_w = _exact_max_window(W, OW)
            grid = (N * C,)  # type: ignore[assignment]
            adaptive_avg_pool1d_general_small_kernel[grid](
                input,
                y,
                N,
                C,
                W,
                OW,
                MAX_K_W=max_k_w,
                ACC_DTYPE=acc_dtype,
                BLOCK_SIZE=32,
                num_warps=1,
                num_stages=1,
            )
        # large OW: divisible
        elif (W % OW) == 0:
            k_w = W // OW

            def grid(meta):  # type: ignore[assignment]
                return (N * C, triton.cdiv(OW, meta["BLOCK_SIZE"]))

            adaptive_avg_pool1d_divisible_large_kernel[grid](
                input,
                y,
                N,
                C,
                W,
                OW,
                K_W=k_w,
                ACC_DTYPE=acc_dtype,
            )
        # large OW: general
        else:
            max_k_w = _exact_max_window(W, OW)

            def grid(meta):  # type: ignore[assignment]
                return (N * C, triton.cdiv(OW, meta["BLOCK_SIZE"]))

            adaptive_avg_pool1d_general_large_kernel[grid](
                input,
                y,
                N,
                C,
                W,
                OW,
                MAX_K_W=max_k_w,
                ACC_DTYPE=acc_dtype,
            )

    return y.squeeze(0) if is_2d else y
