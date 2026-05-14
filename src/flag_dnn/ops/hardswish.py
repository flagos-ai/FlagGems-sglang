import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_HARDSWISH_CONFIGS = runtime.get_tuned_config("hardswish")
_HARDSWISH_FP64_CONFIGS = runtime.get_tuned_config("hardswish")


@triton.jit
def _hardswish(x):
    zero = x * 0
    three = zero + 3
    six = zero + 6

    return tl.where(
        x <= -three,
        zero,
        tl.where(
            x >= three,
            x,
            x * (x + three) / six,
        ),
    )


@libentry()
@libtuner(
    configs=_HARDSWISH_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def hardswish_kernel(
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
    y = _hardswish(x)

    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=_HARDSWISH_FP64_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def hardswish_fp64_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float64)
    y = _hardswish(x)

    tl.store(y_ptr + offsets, y, mask=mask)


def hardswish(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    logger.debug("FLAG_DNN HARDSWISH")

    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn hardswish does not support dtype={input.dtype}"
        )

    if not input.is_contiguous():
        input = input.contiguous()

    n_elements = input.numel()
    if n_elements == 0:
        return input if inplace else torch.empty_like(input)

    y = input if inplace else torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        if input.dtype == torch.float64:
            hardswish_fp64_kernel[grid](
                input,
                y,
                n_elements,
            )
        else:
            hardswish_kernel[grid](
                input,
                y,
                n_elements,
            )

    return y
