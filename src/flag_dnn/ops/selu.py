import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_SELU_CONFIGS = runtime.get_tuned_config("selu")
_SELU_FP64_CONFIGS = runtime.get_tuned_config("selu")


@triton.jit
def _selu(x):
    zero = x * 0
    one = zero + 1
    alpha = zero + 1.6732632423543772848170429916717
    scale = zero + 1.0507009873554804934193349852946

    neg = alpha * (tl.exp(x) - one)
    y = tl.where(x > zero, x, neg)
    return scale * y


@libentry()
@libtuner(
    configs=_SELU_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def selu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = _selu(x)

    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=_SELU_FP64_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def selu_fp64_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float64)
    y = _selu(x)

    tl.store(y_ptr + offsets, y, mask=mask)


def selu(input: torch.Tensor, inplace: bool = False):
    logger.debug("FLAG_DNN SELU")

    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn selu does not support dtype={input.dtype}"
        )

    orig_input = input
    need_copy_back = inplace and (not input.is_contiguous())

    if not input.is_contiguous():
        input = input.contiguous()

    n_elements = input.numel()
    if n_elements == 0:
        return orig_input if inplace else torch.empty_like(input)

    y = input if inplace else torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        if input.dtype == torch.float64:
            selu_fp64_kernel[grid](
                input,
                y,
                n_elements,
            )
        else:
            selu_kernel[grid](
                input,
                y,
                n_elements,
            )

    if need_copy_back:
        orig_input.copy_(y)
        return orig_input

    return y
