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
    configs=runtime.get_tuned_config("avg_pool3d"),
    key=["spatial_size"],
    warmup=5,
    rep=10,
)
@triton.jit
def avg_pool3d_gap_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    spatial_size,
    DIVISOR: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    x_base = pid * spatial_size

    offsets = tl.arange(0, BLOCK_SIZE)
    sum_val = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    for i in range(0, spatial_size, BLOCK_SIZE):
        idx = i + offsets
        mask = idx < spatial_size
        val = tl.load(x_ptr + x_base + idx, mask=mask, other=0.0).to(ACC_DTYPE)
        sum_val += val

    total_sum = tl.sum(sum_val, axis=0)

    # 除数保护
    divisor = tl.where(DIVISOR <= 0, 1, DIVISOR)
    avg_val = total_sum / divisor

    tl.store(
        y_ptr + pid + tl.zeros([1], dtype=tl.int32),
        avg_val.to(x_ptr.dtype.element_ty),
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("avg_pool3d"),
    key=["N", "C", "D", "H", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def avg_pool3d_kernel_1d(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    pad_d,
    pad_h,
    pad_w,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    KERNEL_D: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    HAS_DIVISOR_OVERRIDE: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    num_elements = N * C * OD * OH * OW
    mask = offsets < num_elements

    spatial_size = OD * OH * OW
    spatial_idx = offsets % spatial_size

    ow = spatial_idx % OW
    oh = (spatial_idx // OW) % OH
    od = spatial_idx // (OW * OH)

    batch_channel_idx = offsets // spatial_size
    c = batch_channel_idx % C
    n = batch_channel_idx // C

    x_base_idx = n * (C * D * H * W) + c * (D * H * W)

    d_start = od * STRIDE_D - pad_d
    h_start = oh * STRIDE_H - pad_h
    w_start = ow * STRIDE_W - pad_w

    sum_val = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    # for kd in tl.static_range(KERNEL_D):
    #     for kh in tl.static_range(KERNEL_H):
    # for kw in tl.static_range(KERNEL_W):
    for kd in range(KERNEL_D):
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                id_ = d_start + kd
                ih = h_start + kh
                iw = w_start + kw

                valid = (
                    (id_ >= 0)
                    & (id_ < D)
                    & (ih >= 0)
                    & (ih < H)
                    & (iw >= 0)
                    & (iw < W)
                )
                load_idx = x_base_idx + id_ * (H * W) + ih * W + iw

                val = tl.load(
                    x_ptr + load_idx, mask=mask & valid, other=0.0
                ).to(ACC_DTYPE)
                sum_val += val

    if HAS_DIVISOR_OVERRIDE:
        divisor = DIVISOR_OVERRIDE
    elif COUNT_INCLUDE_PAD:
        dend_bounded = tl.where(
            d_start + KERNEL_D > D + pad_d, D + pad_d, d_start + KERNEL_D
        )
        pool_d = dend_bounded - d_start
        hend_bounded = tl.where(
            h_start + KERNEL_H > H + pad_h, H + pad_h, h_start + KERNEL_H
        )
        pool_h = hend_bounded - h_start
        wend_bounded = tl.where(
            w_start + KERNEL_W > W + pad_w, W + pad_w, w_start + KERNEL_W
        )
        pool_w = wend_bounded - w_start
        divisor = pool_d * pool_h * pool_w
    else:
        id_start_clamp = tl.where(d_start < 0, 0, d_start)
        id_end_clamp = tl.where(d_start + KERNEL_D > D, D, d_start + KERNEL_D)
        valid_d = id_end_clamp - id_start_clamp

        ih_start_clamp = tl.where(h_start < 0, 0, h_start)
        ih_end_clamp = tl.where(h_start + KERNEL_H > H, H, h_start + KERNEL_H)
        valid_h = ih_end_clamp - ih_start_clamp

        iw_start_clamp = tl.where(w_start < 0, 0, w_start)
        iw_end_clamp = tl.where(w_start + KERNEL_W > W, W, w_start + KERNEL_W)
        valid_w = iw_end_clamp - iw_start_clamp

        divisor = valid_d * valid_h * valid_w

    divisor = tl.where(divisor <= 0, 1, divisor)
    avg_val = sum_val / divisor

    tl.store(y_ptr + offsets, avg_val.to(x_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("avg_pool3d"),
    key=["OD", "OH", "OW"],
    warmup=5,
    rep=10,
)
@triton.jit
def avg_pool3d_kernel_2d(
    x_ptr,
    y_ptr,
    N,
    C,
    D,
    H,
    W,
    OD,
    OH,
    OW,
    pad_d,
    pad_h,
    pad_w,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    KERNEL_D: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    HAS_DIVISOR_OVERRIDE: tl.constexpr,
    DIVISOR_OVERRIDE: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_spatial = tle.program_id(0)
    pid_batch_channel = tle.program_id(1)

    n = pid_batch_channel // C
    c = pid_batch_channel % C

    offsets = pid_spatial * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    spatial_numel = OD * OH * OW
    mask = offsets < spatial_numel

    ow = offsets % OW
    oh = (offsets // OW) % OH
    od = offsets // (OW * OH)

    x_base_idx = n * (C * D * H * W) + c * (D * H * W)
    y_base_idx = n * (C * OD * OH * OW) + c * (OD * OH * OW)

    d_start = od * STRIDE_D - pad_d
    h_start = oh * STRIDE_H - pad_h
    w_start = ow * STRIDE_W - pad_w

    sum_val = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    # for kd in tl.static_range(KERNEL_D):
    #     for kh in tl.static_range(KERNEL_H):
    #         for kw in tl.static_range(KERNEL_W):
    for kd in range(KERNEL_D):
        for kh in range(KERNEL_H):
            for kw in range(KERNEL_W):
                id_ = d_start + kd
                ih = h_start + kh
                iw = w_start + kw

                valid = (
                    (id_ >= 0)
                    & (id_ < D)
                    & (ih >= 0)
                    & (ih < H)
                    & (iw >= 0)
                    & (iw < W)
                )
                load_idx = id_ * (H * W) + ih * W + iw

                val = tl.load(
                    x_ptr + x_base_idx + load_idx, mask=mask & valid, other=0.0
                ).to(ACC_DTYPE)
                sum_val += val

    if HAS_DIVISOR_OVERRIDE:
        divisor = DIVISOR_OVERRIDE
    elif COUNT_INCLUDE_PAD:
        dend_bounded = tl.where(
            d_start + KERNEL_D > D + pad_d, D + pad_d, d_start + KERNEL_D
        )
        pool_d = dend_bounded - d_start
        hend_bounded = tl.where(
            h_start + KERNEL_H > H + pad_h, H + pad_h, h_start + KERNEL_H
        )
        pool_h = hend_bounded - h_start
        wend_bounded = tl.where(
            w_start + KERNEL_W > W + pad_w, W + pad_w, w_start + KERNEL_W
        )
        pool_w = wend_bounded - w_start
        divisor = pool_d * pool_h * pool_w
    else:
        id_start_clamp = tl.where(d_start < 0, 0, d_start)
        id_end_clamp = tl.where(d_start + KERNEL_D > D, D, d_start + KERNEL_D)
        valid_d = id_end_clamp - id_start_clamp

        ih_start_clamp = tl.where(h_start < 0, 0, h_start)
        ih_end_clamp = tl.where(h_start + KERNEL_H > H, H, h_start + KERNEL_H)
        valid_h = ih_end_clamp - ih_start_clamp

        iw_start_clamp = tl.where(w_start < 0, 0, w_start)
        iw_end_clamp = tl.where(w_start + KERNEL_W > W, W, w_start + KERNEL_W)
        valid_w = iw_end_clamp - iw_start_clamp

        divisor = valid_d * valid_h * valid_w

    divisor = tl.where(divisor <= 0, 1, divisor)
    avg_val = sum_val / divisor

    tl.store(
        y_ptr + y_base_idx + offsets,
        avg_val.to(x_ptr.dtype.element_ty),
        mask=mask,
    )


def avg_pool3d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Optional[Union[int, Tuple[int, int, int]]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> torch.Tensor:
    logger.debug(
        f"FLAG_DNN AVG_POOL3D "
        f"(kernel={kernel_size}, "
        f"divisor={divisor_override})"
    )

    def _triple(x):
        return (x, x, x) if isinstance(x, int) else tuple(x)

    kernel_size = _triple(kernel_size)
    stride = _triple(stride) if stride is not None else kernel_size
    padding = _triple(padding)

    assert input.ndim in [
        4,
        5,
    ], "Input must be 4D (C, D, H, W) or 5D (N, C, D, H, W)"
    is_4d = input.ndim == 4
    if is_4d:
        input = input.unsqueeze(0)

    N, C, D, H, W = input.shape

    def _out_size(L, pad, k, s, ceil):
        out = (L + 2 * pad - k) / s + 1
        return math.ceil(out) if ceil else math.floor(out)

    OD = _out_size(
        D,
        padding[0],  # type: ignore[index]
        kernel_size[0],  # type: ignore[index]
        stride[0],  # type: ignore[index]
        ceil_mode,
    )
    OH = _out_size(
        H,
        padding[1],  # type: ignore[index]
        kernel_size[1],  # type: ignore[index]
        stride[1],  # type: ignore[index]
        ceil_mode,
    )
    OW = _out_size(
        W,
        padding[2],  # type: ignore[index]
        kernel_size[2],  # type: ignore[index]
        stride[2],  # type: ignore[index]
        ceil_mode,
    )

    if ceil_mode:
        if (OD - 1) * stride[0] >= D + padding[0]:  # type: ignore[index]
            OD -= 1
        if (OH - 1) * stride[1] >= H + padding[1]:  # type: ignore[index]
            OH -= 1
        if (OW - 1) * stride[2] >= W + padding[2]:  # type: ignore[index]
            OW -= 1

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    y = torch.empty((N, C, OD, OH, OW), dtype=input.dtype, device=input.device)

    M = N * C * OD * OH * OW
    if M == 0:
        return y.squeeze(0) if is_4d else y

    acc_dtype = tl.float64 if input.dtype == torch.float64 else tl.float32
    has_divisor_override = divisor_override is not None
    div_override_val = divisor_override if has_divisor_override else -1

    with torch_device_fn.device(input.device):
        # 只有一个输出元素, Kernel 的范围必须能覆盖整个输入 (确保 start <= 0 且 end >= InputSize)
        is_gap_full_coverage = (
            OD == 1
            and OH == 1
            and OW == 1
            and (-padding[0] + kernel_size[0] >= D)  # type: ignore[index]
            and (-padding[1] + kernel_size[1] >= H)  # type: ignore[index]
            and (-padding[2] + kernel_size[2] >= W)  # type: ignore[index]
        )

        if is_gap_full_coverage:
            if has_divisor_override:
                gap_divisor = divisor_override
            elif count_include_pad:
                pool_d = min(
                    -padding[0] + kernel_size[0],  # type: ignore[index]
                    D + padding[0],  # type: ignore[index]
                ) - (
                    -padding[0]  # type: ignore[index]
                )
                pool_h = min(
                    -padding[1] + kernel_size[1],  # type: ignore[index]
                    H + padding[1],  # type: ignore[index]
                ) - (
                    -padding[1]  # type: ignore[index]
                )
                pool_w = min(
                    -padding[2] + kernel_size[2],  # type: ignore[index]
                    W + padding[2],  # type: ignore[index]
                ) - (
                    -padding[2]  # type: ignore[index]
                )
                gap_divisor = pool_d * pool_h * pool_w
            else:
                valid_d = min(
                    -padding[0] + kernel_size[0], D  # type: ignore[index]
                ) - max(
                    -padding[0], 0  # type: ignore[index]
                )
                valid_h = min(
                    -padding[1] + kernel_size[1], H  # type: ignore[index]
                ) - max(
                    -padding[1], 0  # type: ignore[index]
                )
                valid_w = min(
                    -padding[2] + kernel_size[2], W  # type: ignore[index]
                ) - max(
                    -padding[2], 0  # type: ignore[index]
                )
                gap_divisor = valid_d * valid_h * valid_w

            def grid_gap(meta):
                return (N * C,)

            spatial_size = D * H * W
            avg_pool3d_gap_kernel[grid_gap](
                input,
                y,
                N,
                C,
                spatial_size,
                DIVISOR=gap_divisor,
                ACC_DTYPE=acc_dtype,
            )
        elif OD * OH * OW <= 64:

            def grid_1d(meta):
                return (triton.cdiv(M, meta["BLOCK_SIZE"]),)

            avg_pool3d_kernel_1d[grid_1d](
                input,
                y,
                N,
                C,
                D,
                H,
                W,
                OD,
                OH,
                OW,
                padding[0],  # type: ignore[index]
                padding[1],  # type: ignore[index]
                padding[2],  # type: ignore[index]
                STRIDE_D=stride[0],  # type: ignore[index]
                STRIDE_H=stride[1],  # type: ignore[index]
                STRIDE_W=stride[2],  # type: ignore[index]
                KERNEL_D=kernel_size[0],  # type: ignore[index]
                KERNEL_H=kernel_size[1],  # type: ignore[index]
                KERNEL_W=kernel_size[2],  # type: ignore[index]
                COUNT_INCLUDE_PAD=count_include_pad,
                HAS_DIVISOR_OVERRIDE=has_divisor_override,
                DIVISOR_OVERRIDE=div_override_val,
                ACC_DTYPE=acc_dtype,
            )
        else:

            def grid_2d(meta):
                return (
                    triton.cdiv(OD * OH * OW, meta["BLOCK_SIZE"]),
                    N * C,
                )

            avg_pool3d_kernel_2d[grid_2d](
                input,
                y,
                N,
                C,
                D,
                H,
                W,
                OD,
                OH,
                OW,
                padding[0],  # type: ignore[index]
                padding[1],  # type: ignore[index]
                padding[2],  # type: ignore[index]
                STRIDE_D=stride[0],  # type: ignore[index]
                STRIDE_H=stride[1],  # type: ignore[index]
                STRIDE_W=stride[2],  # type: ignore[index]
                KERNEL_D=kernel_size[0],  # type: ignore[index]
                KERNEL_H=kernel_size[1],  # type: ignore[index]
                KERNEL_W=kernel_size[2],  # type: ignore[index]
                COUNT_INCLUDE_PAD=count_include_pad,
                HAS_DIVISOR_OVERRIDE=has_divisor_override,
                DIVISOR_OVERRIDE=div_override_val,
                ACC_DTYPE=acc_dtype,
            )

    return y.squeeze(0) if is_4d else y
