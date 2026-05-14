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
    configs=runtime.get_tuned_config("max_pool3d"),
    key=["N", "C", "D", "H", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def max_pool3d_kernel_1d(
    x_ptr,
    y_ptr,
    idx_ptr,
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
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KERNEL_D: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
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

    max_val = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)
    max_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    for kd in tl.static_range(KERNEL_D):
        for kh in tl.static_range(KERNEL_H):
            for kw in tl.static_range(KERNEL_W):
                id_ = d_start + kd * DIL_D
                ih = h_start + kh * DIL_H
                iw = w_start + kw * DIL_W

                valid = (
                    (id_ >= 0)
                    & (id_ < D)
                    & (ih >= 0)
                    & (ih < H)
                    & (iw >= 0)
                    & (iw < W)
                )
                load_idx = x_base_idx + id_ * (H * W) + ih * W + iw

                # 统一转换为 float32 进行比较，避免 fp16/bf16 的精度截断问题
                val = tl.load(
                    x_ptr + load_idx, mask=mask & valid, other=-float("inf")
                ).to(tl.float32)

                update_mask = val > max_val
                max_val = tl.where(update_mask, val, max_val)

                if RETURN_INDICES:
                    current_idx = id_ * (H * W) + ih * W + iw
                    max_idx = tl.where(update_mask, current_idx, max_idx)

    tl.store(y_ptr + offsets, max_val.to(x_ptr.dtype.element_ty), mask=mask)
    if RETURN_INDICES:
        tl.store(idx_ptr + offsets, max_idx, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("max_pool3d"),
    key=["OD", "OH", "OW"],
    warmup=5,
    rep=10,
)
@triton.jit
def max_pool3d_kernel_2d(
    x_ptr,
    y_ptr,
    idx_ptr,
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
    DIL_D: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    KERNEL_D: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
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

    max_val = tl.full([BLOCK_SIZE], -float("inf"), dtype=tl.float32)
    max_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    for kd in tl.static_range(KERNEL_D):
        for kh in tl.static_range(KERNEL_H):
            for kw in tl.static_range(KERNEL_W):
                id_ = d_start + kd * DIL_D
                ih = h_start + kh * DIL_H
                iw = w_start + kw * DIL_W

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
                    x_ptr + x_base_idx + load_idx,
                    mask=mask & valid,
                    other=-float("inf"),
                ).to(tl.float32)

                update_mask = val > max_val
                max_val = tl.where(update_mask, val, max_val)

                if RETURN_INDICES:
                    max_idx = tl.where(update_mask, load_idx, max_idx)

    tl.store(
        y_ptr + y_base_idx + offsets,
        max_val.to(x_ptr.dtype.element_ty),
        mask=mask,
    )
    if RETURN_INDICES:
        tl.store(idx_ptr + y_base_idx + offsets, max_idx, mask=mask)


def max_pool3d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Optional[Union[int, Tuple[int, int, int]]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    logger.debug(
        f"FLAG_DNN MAX_POOL3D "
        f"(kernel={kernel_size}, "
        f"return_indices={return_indices})"
    )

    def _triple(x):
        return (x, x, x) if isinstance(x, int) else tuple(x)

    kernel_size = _triple(kernel_size)
    stride = _triple(stride) if stride is not None else kernel_size
    padding = _triple(padding)
    dilation = _triple(dilation)

    assert input.ndim in [
        4,
        5,
    ], "Input must be 4D (C, D, H, W) or 5D (N, C, D, H, W)"
    is_4d = input.ndim == 4
    if is_4d:
        input = input.unsqueeze(0)

    N, C, D, H, W = input.shape

    def _out_size(L, pad, dil, k, s, ceil):
        out = (L + 2 * pad - dil * (k - 1) - 1) / s + 1
        return math.ceil(out) if ceil else math.floor(out)

    OD = _out_size(
        D,
        padding[0],  # type: ignore[index]
        dilation[0],  # type: ignore[index]
        kernel_size[0],  # type: ignore[index]
        stride[0],  # type: ignore[index]
        ceil_mode,
    )
    OH = _out_size(
        H,
        padding[1],  # type: ignore[index]
        dilation[1],  # type: ignore[index]
        kernel_size[1],  # type: ignore[index]
        stride[1],  # type: ignore[index]
        ceil_mode,
    )
    OW = _out_size(
        W,
        padding[2],  # type: ignore[index]
        dilation[2],  # type: ignore[index]
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
    idx = (
        torch.empty((N, C, OD, OH, OW), dtype=torch.int64, device=input.device)
        if return_indices
        else None
    )

    M = N * C * OD * OH * OW
    if M == 0:
        out_y = y.squeeze(0) if is_4d else y
        if return_indices:
            return out_y, (idx.squeeze(0) if is_4d else idx)
        return out_y

    with torch_device_fn.device(input.device):
        # 3D 情况下，体积小于等于 64 依然走 1D 高并发
        if OD * OH * OW <= 64:

            def grid_1d(meta):
                return (triton.cdiv(M, meta["BLOCK_SIZE"]),)

            max_pool3d_kernel_1d[grid_1d](
                input,
                y,
                idx,
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
                DIL_D=dilation[0],  # type: ignore[index]
                DIL_H=dilation[1],  # type: ignore[index]
                DIL_W=dilation[2],  # type: ignore[index]
                KERNEL_D=kernel_size[0],  # type: ignore[index]
                KERNEL_H=kernel_size[1],  # type: ignore[index]
                KERNEL_W=kernel_size[2],  # type: ignore[index]
                RETURN_INDICES=return_indices,
            )
        else:

            def grid_2d(meta):
                return (
                    triton.cdiv(OD * OH * OW, meta["BLOCK_SIZE"]),
                    N * C,
                )

            max_pool3d_kernel_2d[grid_2d](
                input,
                y,
                idx,
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
                DIL_D=dilation[0],  # type: ignore[index]
                DIL_H=dilation[1],  # type: ignore[index]
                DIL_W=dilation[2],  # type: ignore[index]
                KERNEL_D=kernel_size[0],  # type: ignore[index]
                KERNEL_H=kernel_size[1],  # type: ignore[index]
                KERNEL_W=kernel_size[2],  # type: ignore[index]
                RETURN_INDICES=return_indices,
            )

    out_y = y.squeeze(0) if is_4d else y
    if return_indices:
        return out_y, (idx.squeeze(0) if is_4d else idx)
    return out_y
