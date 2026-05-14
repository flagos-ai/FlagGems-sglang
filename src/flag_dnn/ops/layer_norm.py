import logging
from typing import Optional, Tuple

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
    configs=runtime.get_tuned_config("layer_norm"),
    key=["N"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    row_idx = tle.program_id(0)

    x_row_ptr = x_ptr + row_idx * N
    y_row_ptr = y_ptr + row_idx * N

    sum_x = 0.0
    sum_x2 = 0.0
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    # 均值和方差
    mean = sum_x / N
    var = (sum_x2 / N) - (mean * mean)
    # 防浮点精度越界产生负数
    var = tl.maximum(var, 0.0)
    rstd = 1.0 / tl.sqrt(var + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd

        # 仿射变换
        if HAS_WEIGHT:
            weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x_hat = x_hat * weight

        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(
                tl.float32
            )
            x_hat = x_hat + bias

        y = x_hat.to(x_ptr.dtype.element_ty)
        tl.store(y_row_ptr + cols, y, mask=mask)


def layer_norm(
    input: torch.Tensor,
    normalized_shape: Tuple[int, ...],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN LAYER_NORM (eps={eps})")

    if input.numel() == 0:
        return torch.empty_like(input)

    assert input.ndim >= len(
        normalized_shape
    ), "Input dimensions must be >= normalized_shape length"

    if not input.is_contiguous():
        assert False, "input must be contiguous."
        input = input.contiguous()

    y = torch.empty_like(input)

    N = 1
    tail_shape = input.shape[-len(normalized_shape) :]
    if tuple(normalized_shape) != tuple(tail_shape):
        raise ValueError(
            "The normalized_shape must match"
            " the last few dimensions of"
            " the input tensor."
        )

    for dim in normalized_shape:
        N *= dim

    M = input.numel() // N

    grid = (M,)

    with torch_device_fn.device(input.device):
        layer_norm_kernel[grid](
            input,
            y,
            weight,
            bias,
            M,
            N,
            eps,
            HAS_WEIGHT=(weight is not None),
            HAS_BIAS=(bias is not None),
        )

    return y
