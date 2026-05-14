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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("avg_pool2d"),
    key=["N", "C", "H", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def avg_pool2d_kernel_1d(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    OH,
    OW,
    pad_h,
    pad_w,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    num_elements = N * C * OH * OW
    mask = offsets < num_elements

    spatial_size = OH * OW
    spatial_idx = offsets % spatial_size
    ow = spatial_idx % OW
    oh = spatial_idx // OW

    batch_channel_idx = offsets // spatial_size
    c = batch_channel_idx % C
    n = batch_channel_idx // C

    x_base_idx = n * (C * H * W) + c * (H * W)
    h_start = oh * STRIDE_H - pad_h
    w_start = ow * STRIDE_W - pad_w

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kh in tl.static_range(KERNEL_H):
        for kw in tl.static_range(KERNEL_W):
            ih = h_start + kh
            iw = w_start + kw

            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            load_idx = x_base_idx + ih * W + iw

            val = tl.load(x_ptr + load_idx, mask=mask & valid, other=0.0).to(
                tl.float32
            )
            sum_val += val

    if DIVISOR_OVERRIDE > 0:
        divisor = DIVISOR_OVERRIDE
    elif COUNT_INCLUDE_PAD:
        hend_bounded = tl.where(
            h_start + KERNEL_H > H + pad_h, H + pad_h, h_start + KERNEL_H
        )
        pool_h = hend_bounded - h_start
        wend_bounded = tl.where(
            w_start + KERNEL_W > W + pad_w, W + pad_w, w_start + KERNEL_W
        )
        pool_w = wend_bounded - w_start
        divisor = pool_h * pool_w
    else:
        ih_start_clamp = tl.where(h_start < 0, 0, h_start)
        ih_end_clamp = tl.where(h_start + KERNEL_H > H, H, h_start + KERNEL_H)
        valid_h = ih_end_clamp - ih_start_clamp

        iw_start_clamp = tl.where(w_start < 0, 0, w_start)
        iw_end_clamp = tl.where(w_start + KERNEL_W > W, W, w_start + KERNEL_W)
        valid_w = iw_end_clamp - iw_start_clamp
        divisor = valid_h * valid_w

    divisor = tl.where(divisor <= 0, 1, divisor)
    avg_val = sum_val / divisor

    tl.store(y_ptr + offsets, avg_val.to(x_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("avg_pool2d"),
    key=["OH", "OW"],
    warmup=5,
    rep=10,
)
@triton.jit
def avg_pool2d_kernel_2d(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    OH,
    OW,
    pad_h,
    pad_w,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_spatial = tle.program_id(0)
    pid_batch_channel = tle.program_id(1)

    n = pid_batch_channel // C
    c = pid_batch_channel % C

    offsets = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_numel = OH * OW
    mask = offsets < spatial_numel

    ow = offsets % OW
    oh = offsets // OW

    x_base_idx = n * (C * H * W) + c * (H * W)
    y_base_idx = n * (C * OH * OW) + c * (OH * OW)

    h_start = oh * STRIDE_H - pad_h
    w_start = ow * STRIDE_W - pad_w

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kh in tl.static_range(KERNEL_H):
        for kw in tl.static_range(KERNEL_W):
            ih = h_start + kh
            iw = w_start + kw

            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            load_idx = ih * W + iw

            val = tl.load(
                x_ptr + x_base_idx + load_idx, mask=mask & valid, other=0.0
            ).to(tl.float32)
            sum_val += val

    if DIVISOR_OVERRIDE > 0:
        divisor = DIVISOR_OVERRIDE
    elif COUNT_INCLUDE_PAD:
        hend_bounded = tl.where(
            h_start + KERNEL_H > H + pad_h, H + pad_h, h_start + KERNEL_H
        )
        pool_h = hend_bounded - h_start
        wend_bounded = tl.where(
            w_start + KERNEL_W > W + pad_w, W + pad_w, w_start + KERNEL_W
        )
        pool_w = wend_bounded - w_start
        divisor = pool_h * pool_w
    else:
        ih_start_clamp = tl.where(h_start < 0, 0, h_start)
        ih_end_clamp = tl.where(h_start + KERNEL_H > H, H, h_start + KERNEL_H)
        valid_h = ih_end_clamp - ih_start_clamp

        iw_start_clamp = tl.where(w_start < 0, 0, w_start)
        iw_end_clamp = tl.where(w_start + KERNEL_W > W, W, w_start + KERNEL_W)
        valid_w = iw_end_clamp - iw_start_clamp
        divisor = valid_h * valid_w

    divisor = tl.where(divisor <= 0, 1, divisor)
    avg_val = sum_val / divisor

    tl.store(
        y_ptr + y_base_idx + offsets,
        avg_val.to(x_ptr.dtype.element_ty),
        mask=mask,
    )


def avg_pool2d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> torch.Tensor:
    logger.debug(
        f"FLAG_DNN AVG_POOL2D "
        f"(kernel={kernel_size}, "
        f"count_include_pad={count_include_pad})"
    )

    def _pair(x):
        return (x, x) if isinstance(x, int) else tuple(x)

    kernel_size = _pair(kernel_size)
    stride = _pair(stride) if stride is not None else kernel_size
    padding = _pair(padding)

    assert input.ndim in [3, 4], "Input must be 3D or 4D"
    is_3d = input.ndim == 3
    if is_3d:
        input = input.unsqueeze(0)

    N, C, H, W = input.shape

    def _out_size(L, pad, k, s, ceil):
        out = (L + 2 * pad - k) / s + 1
        return math.ceil(out) if ceil else math.floor(out)

    OH = _out_size(
        H,
        padding[0],  # type: ignore[index]
        kernel_size[0],  # type: ignore[index]
        stride[0],  # type: ignore[index]
        ceil_mode,
    )
    OW = _out_size(
        W,
        padding[1],  # type: ignore[index]
        kernel_size[1],  # type: ignore[index]
        stride[1],  # type: ignore[index]
        ceil_mode,
    )

    if ceil_mode:
        if (OH - 1) * stride[0] >= H + padding[0]:  # type: ignore[index]
            OH -= 1
        if (OW - 1) * stride[1] >= W + padding[1]:  # type: ignore[index]
            OW -= 1

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    y = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)

    M = N * C * OH * OW
    if M == 0:
        return y.squeeze(0) if is_3d else y

    div_over = divisor_override if divisor_override is not None else -1

    with torch_device_fn.device(input.device):
        if OH * OW <= 64:

            def grid_1d(meta):
                return (triton.cdiv(M, meta["BLOCK_SIZE"]),)

            avg_pool2d_kernel_1d[grid_1d](
                input,
                y,
                N,
                C,
                H,
                W,
                OH,
                OW,
                padding[0],  # type: ignore[index]
                padding[1],  # type: ignore[index]
                STRIDE_H=stride[0],  # type: ignore[index]
                STRIDE_W=stride[1],  # type: ignore[index]
                KERNEL_H=kernel_size[0],  # type: ignore[index]
                KERNEL_W=kernel_size[1],  # type: ignore[index]
                COUNT_INCLUDE_PAD=count_include_pad,
                DIVISOR_OVERRIDE=div_over,
            )
        else:

            def grid_2d(meta):
                return (
                    triton.cdiv(OH * OW, meta["BLOCK_SIZE"]),
                    N * C,
                )

            avg_pool2d_kernel_2d[grid_2d](
                input,
                y,
                N,
                C,
                H,
                W,
                OH,
                OW,
                padding[0],  # type: ignore[index]
                padding[1],  # type: ignore[index]
                STRIDE_H=stride[0],  # type: ignore[index]
                STRIDE_W=stride[1],  # type: ignore[index]
                KERNEL_H=kernel_size[0],  # type: ignore[index]
                KERNEL_W=kernel_size[1],  # type: ignore[index]
                COUNT_INCLUDE_PAD=count_include_pad,
                DIVISOR_OVERRIDE=div_over,
            )

    return y.squeeze(0) if is_3d else y
