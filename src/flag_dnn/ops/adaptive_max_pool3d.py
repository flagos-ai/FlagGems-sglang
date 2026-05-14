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
    """
    精确计算自适应池化在给定输入和输出大小时的最大滑动窗口。
    原理: end_i - start_i = (i * IN % OUT + IN + OUT - 1) // OUT
    其余数的最大值必定为 OUT - gcd(IN, OUT)
    """
    gcd_val = math.gcd(in_size, out_size)
    max_r = out_size - gcd_val
    return (max_r + in_size + out_size - 1) // out_size


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_max_pool3d"),
    key=["D", "H", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def global_max_pool3d_kernel(
    x_ptr,
    y_ptr,
    idx_ptr,
    D,
    H,
    W,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    nc_idx = tle.program_id(0)
    DHW = D * H * W
    x_base = x_ptr + nc_idx * DHW
    input_dtype = x_ptr.dtype.element_ty

    max_vals = tl.full([BLOCK_SIZE], -float("inf"), dtype=input_dtype)
    max_idxs = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    for dhw_offset in range(0, DHW, BLOCK_SIZE):
        offsets = dhw_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < DHW

        vals = tl.load(x_base + offsets, mask=mask, other=-float("inf"))
        vals = tl.cast(vals, input_dtype)

        update_mask = (vals > max_vals) & mask
        max_vals = tl.where(update_mask, vals, max_vals)
        if RETURN_INDICES:
            max_idxs = tl.where(update_mask, offsets, max_idxs)

    best_val = tl.max(max_vals, axis=0)
    tl.store(y_ptr + nc_idx, best_val)

    if RETURN_INDICES:
        local_argmax = tl.argmax(max_vals, axis=0)
        extract_mask = tl.arange(0, BLOCK_SIZE) == local_argmax
        zero_tensor = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)
        best_idx = tl.sum(
            tl.where(extract_mask, max_idxs, zero_tensor), axis=0
        )
        tl.store(idx_ptr + nc_idx, best_idx)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_max_pool3d"),
    key=["N", "C", "D", "H", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def adaptive_max_pool3d_kernel(
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
    OW_OH,
    OW_OH_OD,
    C_OW_OH_OD,  # 传入外部预计算好的常量，避免 GPU 内部乘法计算
    MAX_K_D: tl.constexpr,
    MAX_K_H: tl.constexpr,
    MAX_K_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    num_elements = N * C * OD * OH * OW
    valid_mask = offsets < num_elements

    # 反推 3D 输出坐标 (利用预计算的常量加速除法)
    ow = offsets % OW
    oh = (offsets // OW) % OH
    od = (offsets // OW_OH) % OD
    c = (offsets // OW_OH_OD) % C
    n = offsets // C_OW_OH_OD

    x_base_idx = n * (C * D * H * W) + c * (D * H * W)

    start_d = (od * D) // OD
    end_d = ((od + 1) * D + OD - 1) // OD

    start_h = (oh * H) // OH
    end_h = ((oh + 1) * H + OH - 1) // OH

    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW

    start_d = tl.minimum(start_d, D)
    end_d = tl.minimum(end_d, D)
    start_h = tl.minimum(start_h, H)
    end_h = tl.minimum(end_h, H)
    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)

    input_dtype = x_ptr.dtype.element_ty
    max_val = tl.full([BLOCK_SIZE], -float("inf"), dtype=input_dtype)
    max_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    for kd in range(MAX_K_D):
        id_ = start_d + kd
        in_window_d = id_ < end_d

        for kh in range(MAX_K_H):
            ih = start_h + kh
            in_window_h = in_window_d & (ih < end_h)

            for kw in range(MAX_K_W):
                iw = start_w + kw
                in_window = in_window_h & (iw < end_w)

                spatial_idx = id_ * (H * W) + ih * W + iw
                load_idx = x_base_idx + spatial_idx

                val = tl.load(
                    x_ptr + load_idx,
                    mask=valid_mask & in_window,
                    other=-float("inf"),
                )

                is_new_max = val > max_val
                update_mask = is_new_max & in_window & valid_mask

                max_val = tl.where(update_mask, val, max_val)
                if RETURN_INDICES:
                    max_idx = tl.where(update_mask, spatial_idx, max_idx)

    tl.store(y_ptr + offsets, max_val, mask=valid_mask)
    if RETURN_INDICES:
        tl.store(idx_ptr + offsets, max_idx.to(tl.int64), mask=valid_mask)


def adaptive_max_pool3d(
    input: torch.Tensor,
    output_size: Union[
        int, Tuple[Optional[int], Optional[int], Optional[int]]
    ],
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug(
        f"FLAG_DNN ADAPTIVE_MAX_POOL3D "
        f"(output_size={output_size}, "
        f"return_indices={True})"
    )

    return_indices: bool = True

    assert input.ndim in [
        4,
        5,
    ], "Input must be 4D (C, D, H, W) or 5D (N, C, D, H, W)"
    is_4d = input.ndim == 4
    if is_4d:
        input = input.unsqueeze(0)

    N, C, D, H, W = input.shape

    if isinstance(output_size, int):
        OD = OH = OW = output_size
    else:
        assert (
            len(output_size) == 3
        ), "output_size must be an int or a tuple of 3 ints"
        OD = output_size[0] if output_size[0] is not None else D
        OH = output_size[1] if output_size[1] is not None else H
        OW = output_size[2] if output_size[2] is not None else W

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    y = torch.empty((N, C, OD, OH, OW), dtype=input.dtype, device=input.device)

    indices = None
    idx_ptr = input
    if return_indices:
        indices = torch.empty(
            (N, C, OD, OH, OW), dtype=torch.int64, device=input.device
        )
        idx_ptr = indices

    M = N * C * OD * OH * OW

    if M == 0:
        y_out = y.squeeze(0) if is_4d else y
        if return_indices:
            idx_out = indices.squeeze(0) if is_4d else indices
            return y_out, idx_out
        return y_out

    with torch_device_fn.device(input.device):
        if OD == 1 and OH == 1 and OW == 1:

            def grid_global(meta):
                return (N * C,)

            global_max_pool3d_kernel[grid_global](
                input, y, idx_ptr, D, H, W, RETURN_INDICES=return_indices
            )
        else:
            max_k_d = _exact_max_window(D, OD)
            max_k_h = _exact_max_window(H, OH)
            max_k_w = _exact_max_window(W, OW)

            # Python 层预计算维度积，避免在 GPU Kernel 中做除法与乘法
            OW_OH = OW * OH
            OW_OH_OD = OW_OH * OD
            C_OW_OH_OD = C * OW_OH_OD

            def grid(meta):
                return (triton.cdiv(M, meta["BLOCK_SIZE"]),)

            adaptive_max_pool3d_kernel[grid](
                input,
                y,
                idx_ptr,
                N,
                C,
                D,
                H,
                W,
                OD,
                OH,
                OW,
                OW_OH,
                OW_OH_OD,
                C_OW_OH_OD,
                MAX_K_D=max_k_d,
                MAX_K_H=max_k_h,
                MAX_K_W=max_k_w,
                RETURN_INDICES=return_indices,
            )

    y_out = y.squeeze(0) if is_4d else y
    if return_indices:
        idx_out = indices.squeeze(0) if is_4d else indices
        return y_out, idx_out

    return y_out
