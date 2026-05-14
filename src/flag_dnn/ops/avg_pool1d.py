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
    configs=runtime.get_tuned_config("avg_pool1d"),
    key=["N", "C", "W"],
    warmup=5,
    rep=10,
)
@triton.jit
def avg_pool1d_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    W,
    OW,
    pad_w,
    STRIDE_W: tl.constexpr,
    KERNEL_W: tl.constexpr,
    COUNT_INCLUDE_PAD: tl.constexpr,
    ACC_DTYPE: tl.constexpr,  # 动态指定累加精度，防止 fp64 掉精度，也防止 fp16 溢出
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    num_elements = N * C * OW
    mask = offsets < num_elements

    # 反推 1D 坐标 (n, c, ow)
    ow = offsets % OW
    c = (offsets // OW) % C
    n = offsets // (C * OW)

    x_base_idx = n * (C * W) + c * W

    w_start = ow * STRIDE_W - pad_w

    # 根据 count_include_pad 规则，动态计算当前窗口除数
    if COUNT_INCLUDE_PAD:
        start_pad = tl.maximum(w_start, -pad_w)
        end_pad = tl.minimum(w_start + KERNEL_W, W + pad_w)
        pool_size = end_pad - start_pad
    else:
        start_valid = tl.maximum(w_start, 0)
        end_valid = tl.minimum(w_start + KERNEL_W, W)
        pool_size = end_valid - start_valid

    # 防止极端情况下除数为 0
    pool_size = tl.where(pool_size == 0, 1, pool_size)

    # 使用高精度初始化累加器
    sum_val = tl.zeros([BLOCK_SIZE], dtype=ACC_DTYPE)

    # 循环展开累加
    for kw in tl.static_range(KERNEL_W):
        iw = w_start + kw
        valid = (iw >= 0) & (iw < W)
        load_idx = x_base_idx + iw

        # 加载时转为累加精度 (fp32 或 fp64)，越界区域补 0.0
        val = tl.load(x_ptr + load_idx, mask=mask & valid, other=0.0).to(
            ACC_DTYPE
        )
        sum_val += val

    # 计算平均值并转回原数据类型
    res = sum_val / pool_size
    tl.store(y_ptr + offsets, res.to(x_ptr.dtype.element_ty), mask=mask)


def avg_pool1d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int]],
    stride: Optional[Union[int, Tuple[int]]] = None,
    padding: Union[int, Tuple[int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> torch.Tensor:
    logger.debug(
        f"FLAG_DNN AVG_POOL1D "
        f"(kernel={kernel_size}, "
        f"count_include_pad={count_include_pad})"
    )

    def _single(x):
        return (x,) if isinstance(x, int) else tuple(x)

    kernel_size = _single(kernel_size)
    stride = _single(stride) if stride is not None else kernel_size
    padding = _single(padding)

    assert input.ndim in [2, 3], "Input must be 2D or 3D"
    is_2d = input.ndim == 2
    if is_2d:
        input = input.unsqueeze(0)

    N, C, W = input.shape

    def _out_size(L, pad, k, s, ceil):
        out = (L + 2 * pad - k) / s + 1
        return math.ceil(out) if ceil else math.floor(out)

    OW = _out_size(
        W,
        padding[0],  # type: ignore[index]
        kernel_size[0],  # type: ignore[index]
        stride[0],  # type: ignore[index]
        ceil_mode,
    )

    # ceil_mode 边缘丢弃
    if ceil_mode:
        if (OW - 1) * stride[0] >= W + padding[0]:  # type: ignore[index]
            OW -= 1

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    y = torch.empty((N, C, OW), dtype=input.dtype, device=input.device)

    M = N * C * OW
    if M == 0:
        return y.squeeze(0) if is_2d else y

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE"]),)

    # 动态分发累加精度：float64 原生保留，其他统统升级到 float32 运算
    acc_dtype = tl.float64 if input.dtype == torch.float64 else tl.float32

    with torch_device_fn.device(input.device):
        avg_pool1d_kernel[grid](
            input,
            y,
            N,
            C,
            W,
            OW,
            padding[0],  # type: ignore[index]
            STRIDE_W=stride[0],  # type: ignore[index]
            KERNEL_W=kernel_size[0],  # type: ignore[index]
            COUNT_INCLUDE_PAD=count_include_pad,
            ACC_DTYPE=acc_dtype,
        )

    return y.squeeze(0) if is_2d else y
