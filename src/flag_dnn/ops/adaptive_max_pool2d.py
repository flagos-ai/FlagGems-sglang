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
    configs=runtime.get_tuned_config("adaptive_max_pool2d"),
    key=["H", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def global_max_pool2d_kernel(
    x_ptr,
    y_ptr,
    idx_ptr,
    H,
    W,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 映射到 N * C，保证极高的 Occupancy
    nc_idx = tle.program_id(0)
    HW = H * W
    x_base = x_ptr + nc_idx * HW
    input_dtype = x_ptr.dtype.element_ty

    # 维护 Tensor，彻底规避标量类型推导问题
    max_vals = tl.full([BLOCK_SIZE], -float("inf"), dtype=input_dtype)
    max_idxs = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    # 将 2D 拉平为 1D 向量化读取
    for hw_offset in range(0, HW, BLOCK_SIZE):
        offsets = hw_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW

        # 读取并强制洗掉 other 带来的可能隐式提升
        vals = tl.load(x_base + offsets, mask=mask, other=-float("inf"))
        vals = tl.cast(vals, input_dtype)

        # 向量化更新
        update_mask = (vals > max_vals) & mask
        max_vals = tl.where(update_mask, vals, max_vals)
        if RETURN_INDICES:
            # 记录的是一维拉平后的偏移，正好符合 PyTorch 的 2D 索引返回格式
            max_idxs = tl.where(update_mask, offsets, max_idxs)

    # Block 内部做唯一的一次规约降维
    best_val = tl.max(max_vals, axis=0)
    tl.store(y_ptr + nc_idx, best_val)

    if RETURN_INDICES:
        local_argmax = tl.argmax(max_vals, axis=0)

        # 利用 mask 提取对应的真实全局索引
        extract_mask = tl.arange(0, BLOCK_SIZE) == local_argmax
        zero_tensor = tl.full([BLOCK_SIZE], 0, dtype=tl.int64)
        best_idx = tl.sum(
            tl.where(extract_mask, max_idxs, zero_tensor), axis=0
        )

        tl.store(idx_ptr + nc_idx, best_idx)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("adaptive_max_pool2d"),
    key=["N", "C", "H", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def adaptive_max_pool2d_kernel(
    x_ptr,
    y_ptr,
    indices_ptr,
    N,
    C,
    H,
    W,
    OH,
    OW,
    MAX_KH: tl.constexpr,
    MAX_KW: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    num_elements = N * C * OH * OW
    mask = offsets < num_elements

    # 反推输出坐标 (n, c, oh, ow)
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (C * OW * OH)

    x_base_idx = n * (C * H * W) + c * (H * W)

    # 动态计算自适应滑动窗口在原图上的起始和结束边界
    ih_start = (oh * H) // OH
    ih_end = ((oh + 1) * H + OH - 1) // OH
    iw_start = (ow * W) // OW
    iw_end = ((ow + 1) * W + OW - 1) // OW

    # 防止 end 越出图片边界
    ih_end = tl.minimum(ih_end, H)
    iw_end = tl.minimum(iw_end, W)

    input_dtype = x_ptr.dtype.element_ty

    max_val = tl.full([BLOCK_SIZE], -float("inf"), dtype=input_dtype)
    max_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    # 遍历最大可能的自适应窗口区域
    for kh in range(MAX_KH):
        for kw in range(MAX_KW):
            ih = ih_start + kh
            iw = iw_start + kw

            # 判断当前 (ih, iw) 是否在动态窗口的有效范围内
            in_window = (ih < ih_end) & (iw < iw_end)

            load_idx = x_base_idx + ih * W + iw

            # 加载时超出边界的填充负无穷大
            val = tl.load(
                x_ptr + load_idx, mask=mask & in_window, other=-float("inf")
            )

            is_new_max = val > max_val
            update_mask = is_new_max & in_window & mask

            max_val = tl.where(update_mask, val, max_val)

            if RETURN_INDICES:
                # PyTorch 2D Pooling 返回的是拉平到 HW 级别的 1D 偏移
                local_idx = ih * W + iw
                max_idx = tl.where(update_mask, local_idx, max_idx)

    # 写回输出结果
    tl.store(y_ptr + offsets, max_val, mask=mask)
    if RETURN_INDICES:
        tl.store(indices_ptr + offsets, max_idx, mask=mask)


def adaptive_max_pool2d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[Optional[int], Optional[int]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug(
        f"FLAG_DNN ADAPTIVE_MAX_POOL2D "
        f"(output_size={output_size}, "
        f"return_indices={True})"
    )

    return_indices: bool = True

    assert input.ndim in [3, 4], "Input must be 3D or 4D"
    is_3d = input.ndim == 3
    if is_3d:
        input = input.unsqueeze(0)

    N, C, H, W = input.shape

    # 解析 output_size 逻辑
    if isinstance(output_size, int):
        OH = OW = output_size
    else:
        OH = output_size[0] if output_size[0] is not None else H
        OW = output_size[1] if output_size[1] is not None else W

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    M = N * C * OH * OW

    # 拦截特例：空张量
    if M == 0:
        y = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)
        indices = torch.empty(
            (N, C, OH, OW), dtype=torch.int64, device=input.device
        )
        out_y = y.squeeze(0) if is_3d else y
        if return_indices:
            out_idx = indices.squeeze(0) if is_3d else indices
            return out_y, out_idx
        return out_y

    y = torch.empty((N, C, OH, OW), dtype=input.dtype, device=input.device)

    indices = None
    idx_ptr = input
    if return_indices:
        indices = torch.empty(
            (N, C, OH, OW), dtype=torch.int64, device=input.device
        )
        idx_ptr = indices

    with torch_device_fn.device(input.device):
        if OH == 1 and OW == 1:

            def grid_global(meta):
                return (N * C,)

            global_max_pool2d_kernel[grid_global](
                input, y, idx_ptr, H, W, RETURN_INDICES=return_indices
            )
        else:
            # 计算最大可能的池化核大小，给 Triton 提供静态上限
            max_k_h = math.ceil(H / OH) + 1
            max_k_w = math.ceil(W / OW) + 1

            def grid(meta):
                return (triton.cdiv(M, meta["BLOCK_SIZE"]),)

            adaptive_max_pool2d_kernel[grid](
                input,
                y,
                idx_ptr,
                N,
                C,
                H,
                W,
                OH,
                OW,
                MAX_KH=max_k_h,
                MAX_KW=max_k_w,
                RETURN_INDICES=return_indices,
            )

    out_y = y.squeeze(0) if is_3d else y
    if return_indices:
        out_idx = indices.squeeze(0) if is_3d else indices
        return out_y, out_idx

    return out_y
