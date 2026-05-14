import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_SOFTSHRINK_CONFIGS = runtime.get_tuned_config("softshrink")
_SOFTSHRINK_FP64_CONFIGS = runtime.get_tuned_config("softshrink")


@libentry()
@libtuner(
    configs=_SOFTSHRINK_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def softshrink_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    lambd,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = tl.where(
        x > lambd,
        x - lambd,
        tl.where(x < -lambd, x + lambd, 0.0),
    )

    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=_SOFTSHRINK_FP64_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def softshrink_fp64_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    lambd,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float64)
    y = tl.where(
        x > lambd,
        x - lambd,
        tl.where(x < -lambd, x + lambd, 0.0),
    )

    tl.store(y_ptr + offsets, y, mask=mask)


def softshrink(input: torch.Tensor, lambd: float = 0.5) -> torch.Tensor:
    logger.debug("FLAG_DNN SOFTSHRINK")

    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn softshrink does not support dtype={input.dtype}"
        )

    lambd = float(lambd)
    if lambd < 0.0:
        raise ValueError(f"lambda must be non-negative, but got lambd={lambd}")

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
            softshrink_fp64_kernel[grid](
                input,
                y,
                n_elements,
                lambd,
            )
        else:
            softshrink_kernel[grid](
                input,
                y,
                n_elements,
                lambd,
            )

    return y
