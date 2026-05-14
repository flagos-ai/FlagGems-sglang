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


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_norm"),
    key=["C", "HxW"],
    warmup=5,
    rep=10,
)
@triton.jit
def group_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    N,
    C,
    HxW,
    num_groups,
    group_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    # 每一个程序实例负责一个 Group, 总共有 N * num_groups 个 Group
    group_id = tle.program_id(0)

    # 推导出当前 Group 属于哪个 Batch (n_idx) 和 组索引 (g_idx)
    g_idx = group_id % num_groups

    # 当前 Group 在拉平后的起始内存偏移
    group_base_idx = group_id * group_size

    # 当前 Group 起始的绝对 Channel 索引
    C_per_group = C // num_groups
    c_base = g_idx * C_per_group

    # 循环累加求均值 (Mean)
    sum_ = 0.0
    for offset in range(0, group_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < group_size
        x = tl.load(x_ptr + group_base_idx + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        sum_ += tl.sum(x, axis=0)
    mean = sum_ / group_size

    # 循环累加求方差 (Variance)
    var_sum = 0.0
    for offset in range(0, group_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < group_size
        x = tl.load(x_ptr + group_base_idx + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        # 用 mask 屏蔽无效区域的平方和计算
        x_centered = tl.where(mask, x - mean, 0.0)
        var_sum += tl.sum(x_centered * x_centered, axis=0)
    var = var_sum / group_size
    rstd = 1.0 / tl.sqrt(var + eps)

    # 归一化、仿射变换并写回
    for offset in range(0, group_size, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < group_size

        x_orig = tl.load(x_ptr + group_base_idx + cols, mask=mask, other=0.0)
        x = x_orig.to(tl.float32)

        x_hat = (x - mean) * rstd

        if HAS_WEIGHT or HAS_BIAS:
            # 计算当前 offset 下每个元素的具体 Channel ID
            # 因为数据在内存的排列是 (N, C, HxW)
            # 所以在当前组内，每走完 HxW 个元素，Channel 就加 1
            c_idx = c_base + (cols // HxW)

            if HAS_WEIGHT:
                w = tl.load(weight_ptr + c_idx, mask=mask, other=0.0).to(
                    tl.float32
                )
                x_hat = x_hat * w
            if HAS_BIAS:
                b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(
                    tl.float32
                )
                x_hat = x_hat + b

        y = x_hat.to(x_orig.dtype)
        tl.store(y_ptr + group_base_idx + cols, y, mask=mask)


def group_norm(
    input: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN GROUP_NORM (num_groups={num_groups}, eps={eps})")

    # 拦截空张量
    if input.numel() == 0:
        return torch.empty_like(input)

    assert input.ndim >= 2, "Input must have at least 2 dimensions (N, C, ...)"

    N = input.shape[0]
    C = input.shape[1]

    assert (
        C % num_groups == 0
    ), f"Channels ({C}) must be divisible by num_groups ({num_groups})"

    HxW = input.numel() // (N * C)
    C_per_group = C // num_groups
    group_size = C_per_group * HxW

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    y = torch.empty_like(input)

    # 开启 N * num_groups 个并行的线程块
    grid = (N * num_groups,)

    with torch_device_fn.device(input.device):
        group_norm_kernel[grid](
            input,
            y,
            weight,
            bias,
            N,
            C,
            HxW,
            num_groups,
            group_size,
            eps,
            HAS_WEIGHT=(weight is not None),
            HAS_BIAS=(bias is not None),
        )

    return y
