import logging

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
    configs=runtime.get_tuned_config("elu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def elu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    alpha,
    scale,
    input_scale,
    BLOCK_SIZE: tl.constexpr,
    USE_FP32_MATH: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    if USE_FP32_MATH:
        x_math = x.to(tl.float32)
    else:
        x_math = x.to(tl.float64)

    zero = x_math * 0
    one = zero + 1
    alpha_v = zero + alpha
    scale_v = zero + scale
    input_scale_v = zero + input_scale

    y_math = tl.where(
        x_math > zero,
        scale_v * x_math,
        scale_v * alpha_v * (tl.exp(x_math * input_scale_v) - one),
    )

    tl.store(y_ptr + offsets, y_math.to(y_ptr.dtype.element_ty), mask=mask)


def _elu_impl(
    input: torch.Tensor,
    alpha: float = 1.0,
    scale: float = 1.0,
    input_scale: float = 1.0,
    inplace: bool = False,
) -> torch.Tensor:
    logger.debug("FLAG_DNN ELU/SELU INTERNAL")

    alpha = float(alpha)
    scale = float(scale)
    input_scale = float(input_scale)

    orig_input = input
    need_copy_back = inplace and (not input.is_contiguous())

    if not input.is_contiguous():
        input = input.contiguous()

    n_elements = input.numel()

    if n_elements == 0:
        if inplace:
            return orig_input
        return torch.empty_like(input)

    y = input if inplace else torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    use_fp32_math = input.dtype != torch.float64

    with torch_device_fn.device(input.device):
        elu_kernel[grid](
            input,
            y,
            n_elements,
            alpha,
            scale,
            input_scale,
            USE_FP32_MATH=use_fp32_math,
        )

    if need_copy_back:
        orig_input.copy_(y)
        return orig_input

    return y


def elu(
    input: torch.Tensor,
    alpha: float = 1.0,
    inplace: bool = False,
) -> torch.Tensor:
    logger.debug("FLAG_DNN ELU")
    return _elu_impl(
        input,
        alpha=alpha,
        scale=1.0,
        input_scale=1.0,
        inplace=inplace,
    )
