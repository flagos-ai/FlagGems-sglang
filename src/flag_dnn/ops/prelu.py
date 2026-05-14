import logging
import math

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
    configs=runtime.get_tuned_config("prelu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def prelu_scalar_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    a = tl.load(weight_ptr)
    y = tl.where(x >= 0, x, x * a)
    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("prelu"),
    key=["n_elements", "num_channels"],
    warmup=5,
    rep=10,
)
@triton.jit
def prelu_flat_channel_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    n_elements,
    num_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """
    适用于 inner_size == 1 的多通道情况。
    例如:
      - (N, C)
      - (N, C, 1, 1)
    这时 flatten 后每个元素对应的 channel 索引就是 offsets % num_channels。
    """
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    c_idx = offsets % num_channels
    a = tl.load(weight_ptr + c_idx, mask=mask)
    y = tl.where(x >= 0, x, x * a)
    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("prelu"),
    key=["inner_size", "num_channels"],
    warmup=5,
    rep=10,
)
@triton.jit
def prelu_channelwise_kernel(
    x_ptr,
    weight_ptr,
    y_ptr,
    inner_size,  # H * W * ...
    num_channels,  # C
    outer_size,  # N * C
    BLOCK_SIZE: tl.constexpr,
):
    """
    适用于 inner_size > 1 的多通道情况。
    将输入视为 (outer_size, inner_size)，其中 outer_size = N * C。
    每个 program 处理同一个 channel 段中的一段连续元素：
      - channel 索引只算一次
      - slope a 只加载一次
    """
    pid_col = tle.program_id(0)  # 当前 row 内的 block id
    pid_row = tle.program_id(1)  # row id, 范围 [0, outer_size)

    offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < inner_size

    base = pid_row * inner_size
    x = tl.load(x_ptr + base + offsets, mask=mask)

    c_idx = pid_row % num_channels
    a = tl.load(weight_ptr + c_idx)

    y = tl.where(x >= 0, x, x * a)
    tl.store(y_ptr + base + offsets, y, mask=mask)


def prelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logger.debug("FLAG_DNN PRELU")

    assert x.is_contiguous(), "x must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"
    assert x.device == weight.device, "x and weight must be on the same device"
    assert x.dtype == weight.dtype, "x and weight must have the same dtype"

    n_elements = x.numel()
    if n_elements == 0:
        return torch.empty_like(x)

    y = torch.empty_like(x)

    num_parameters = weight.numel()

    with torch_device_fn.device(x.device):
        # 路径1：单参数，共享一个 slope
        if num_parameters == 1:

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            prelu_scalar_kernel[grid](
                x_ptr=x,
                weight_ptr=weight,
                y_ptr=y,
                n_elements=n_elements,
            )
            return y

        # 多参数时，必须满足 PyTorch prelu 的通道规则
        assert x.dim() >= 2, "when weight.numel() > 1, input dim must be >= 2"
        assert num_parameters == x.shape[1], (
            f"weight numel ({num_parameters}) must equal "
            f"channel size ({x.shape[1]})"
        )

        inner_size = math.prod(x.shape[2:]) if x.dim() > 2 else 1

        # 路径2：多通道，但 inner_size == 1
        # 例如 (N, C) 或 (N, C, 1, 1)
        if inner_size == 1:

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            prelu_flat_channel_kernel[grid](
                x_ptr=x,
                weight_ptr=weight,
                y_ptr=y,
                n_elements=n_elements,
                num_channels=num_parameters,
            )
            return y

        # 路径3：多通道，且 inner_size > 1
        # 例如 (N, C, H, W)、(N, C, D, H, W)
        outer_size = x.numel() // inner_size  # = N * C

        def grid(meta):
            return (
                triton.cdiv(inner_size, meta["BLOCK_SIZE"]),
                outer_size,
            )

        prelu_channelwise_kernel[grid](
            x_ptr=x,
            weight_ptr=weight,
            y_ptr=y,
            inner_size=inner_size,
            num_channels=num_parameters,
            outer_size=outer_size,
        )

    return y
