import logging
import math
from typing import Tuple, Union

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
    configs=runtime.get_tuned_config("adaptive_max_pool1d"),
    key=["W"],
    warmup=5,
    rep=10,
)
@triton.jit
def global_max_pool1d_kernel(
    x_ptr,
    y_ptr,
    idx_ptr,
    W,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    nc_idx = tle.program_id(0)
    x_base = x_ptr + nc_idx * W
    input_dtype = x_ptr.dtype.element_ty

    max_vals = tl.full([BLOCK_SIZE], -float("inf"), dtype=input_dtype)
    max_idxs = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    for w_offset in range(0, W, BLOCK_SIZE):
        offsets = w_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < W

        # 读取并强制洗掉 other 带来的可能隐式提升
        vals = tl.load(x_base + offsets, mask=mask, other=-float("inf"))
        vals = tl.cast(vals, input_dtype)

        # 向量化更新
        update_mask = (vals > max_vals) & mask
        max_vals = tl.where(update_mask, vals, max_vals)
        if RETURN_INDICES:
            max_idxs = tl.where(update_mask, offsets, max_idxs)

    # Block 内部做唯一的一次规约降维
    best_val = tl.max(max_vals, axis=0)
    tl.store(y_ptr + nc_idx, best_val)

    if RETURN_INDICES:
        # 找到最大值在 Tensor 内部的局部偏移量 (0 ~ BLOCK_SIZE-1)
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
    configs=runtime.get_tuned_config("adaptive_max_pool1d"),
    key=["N", "C", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def adaptive_max_pool1d_kernel(
    x_ptr,
    y_ptr,
    idx_ptr,
    N,
    C,
    W,
    OW,
    MAX_K_W: tl.constexpr,
    RETURN_INDICES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    num_elements = N * C * OW
    valid_mask = offsets < num_elements

    # 反推 1D 输出坐标 (n, c, ow)
    ow = offsets % OW
    c = (offsets // OW) % C
    n = offsets // (C * OW)

    x_base_idx = n * (C * W) + c * W

    # 动态推导 1D 起点和终点
    start_w = (ow * W) // OW
    end_w = ((ow + 1) * W + OW - 1) // OW

    # 防止边界溢出
    start_w = tl.minimum(start_w, W)
    end_w = tl.minimum(end_w, W)

    input_dtype = x_ptr.dtype.element_ty
    max_val = tl.full([BLOCK_SIZE], -float("inf"), dtype=input_dtype)

    max_idx = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)

    # 1D 动态窗口寻找最大值
    for kw in range(MAX_K_W):
        iw = start_w + kw
        in_window = iw < end_w

        load_idx = x_base_idx + iw

        # 只有当 valid_mask 和 in_window 均为真时，才会去比较更新
        val = tl.load(x_ptr + load_idx, mask=valid_mask & in_window, other=0.0)

        is_new_max = val > max_val
        update_mask = is_new_max & in_window & valid_mask

        max_val = tl.where(update_mask, val, max_val)
        if RETURN_INDICES:
            max_idx = tl.where(update_mask, iw, max_idx)

    # 存储结果
    tl.store(y_ptr + offsets, max_val, mask=valid_mask)
    if RETURN_INDICES:
        tl.store(idx_ptr + offsets, max_idx.to(tl.int64), mask=valid_mask)


def adaptive_max_pool1d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.debug(
        f"FLAG_DNN ADAPTIVE_MAX_POOL1D "
        f"(output_size={output_size}, "
        f"return_indices={True})"
    )

    return_indices: bool = True

    if isinstance(output_size, int):
        OW = output_size
    else:
        OW = output_size[0]

    assert input.ndim in [2, 3], "Input must be 2D or 3D"
    is_2d = input.ndim == 2
    if is_2d:
        input = input.unsqueeze(0)

    N, C, W = input.shape

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    y = torch.empty((N, C, OW), dtype=input.dtype, device=input.device)

    indices = None
    idx_ptr = input
    if return_indices:
        indices = torch.empty(
            (N, C, OW), dtype=torch.int64, device=input.device
        )
        idx_ptr = indices

    M = N * C * OW
    if M == 0:
        y_out = y.squeeze(0) if is_2d else y
        if return_indices:
            idx_out = indices.squeeze(0) if is_2d else indices
            return y_out, idx_out
        return y_out

    # 计算 1D 维度最大可能窗口，给 Triton 的 range() 提供静态上限
    max_k_w = math.ceil(W / OW) + 1

    with torch_device_fn.device(input.device):
        if OW == 1:

            def grid_global(meta):
                return (N * C,)

            global_max_pool1d_kernel[grid_global](
                input, y, idx_ptr, W, RETURN_INDICES=return_indices
            )
        else:

            def grid(meta):
                return (triton.cdiv(M, meta["BLOCK_SIZE"]),)

            adaptive_max_pool1d_kernel[grid](
                input,
                y,
                idx_ptr,
                N,
                C,
                W,
                OW,
                MAX_K_W=max_k_w,
                RETURN_INDICES=return_indices,
            )

    y_out = y.squeeze(0) if is_2d else y
    if return_indices:
        idx_out = indices.squeeze(0) if is_2d else indices
        return y_out, idx_out

    return y_out
