import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_SOFTPLUS_CONFIGS = runtime.get_tuned_config("softplus")
_SOFTPLUS_FP64_CONFIGS = runtime.get_tuned_config("softplus")


@triton.jit
def _softplus_stable_fp32(x, beta, threshold):
    bx = beta * x
    abs_bx = tl.abs(bx)
    sp = (tl.maximum(bx, 0.0) + tl.log(1.0 + tl.exp(-abs_bx))) / beta
    return tl.where(bx > threshold, x, sp)


@triton.jit
def _softplus_stable_fp64(x, beta, threshold):
    bx = beta * x
    abs_bx = tl.abs(bx)
    sp = (tl.maximum(bx, 0.0) + tl.log(1.0 + tl.exp(-abs_bx))) / beta
    return tl.where(bx > threshold, x, sp)


@libentry()
@libtuner(
    configs=_SOFTPLUS_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def softplus_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = _softplus_stable_fp32(x, beta, threshold)

    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=_SOFTPLUS_FP64_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def softplus_fp64_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    beta,
    threshold,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float64)
    y = _softplus_stable_fp64(x, beta, threshold)

    tl.store(y_ptr + offsets, y, mask=mask)


def softplus(
    input: torch.Tensor, beta: float = 1.0, threshold: float = 20.0
) -> torch.Tensor:
    logger.debug("FLAG_DNN SOFTPLUS")

    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn softplus does not support dtype={input.dtype}"
        )

    beta = float(beta)
    threshold = float(threshold)

    if beta <= 0.0:
        raise ValueError(f"beta must be positive, but got beta={beta}")

    if not input.is_contiguous():
        input = input.contiguous()

    n_elements = input.numel()
    if n_elements == 0:
        return torch.empty_like(input)

    y = torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        if input.dtype == torch.float64:
            softplus_fp64_kernel[grid](
                input,
                y,
                n_elements,
                beta,
                threshold,
            )
        else:
            softplus_kernel[grid](
                input,
                y,
                n_elements,
                beta,
                threshold,
            )

    return y
