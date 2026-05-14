import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_TANH_CONFIGS = runtime.get_tuned_config("tanh")
_TANH_FP64_CONFIGS = runtime.get_tuned_config("tanh")

# Triton 的 tanh 在不同版本里导入路径不一致
if tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 0):
    try:
        from triton.language.extra.libdevice import tanh as triton_tanh
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import tanh as triton_tanh
else:
    from triton.language.math import tanh as triton_tanh


@libentry()
@libtuner(
    configs=_TANH_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def tanh_kernel(
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
    y = triton_tanh(x)

    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=_TANH_FP64_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def tanh_fp64_kernel(
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
    y = triton_tanh(x)

    tl.store(y_ptr + offsets, y, mask=mask)


def tanh(input: torch.Tensor) -> torch.Tensor:
    logger.debug("FLAG_DNN TANH")

    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn tanh does not support dtype={input.dtype}"
        )

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
            tanh_fp64_kernel[grid](
                input,
                y,
                n_elements,
            )
        else:
            tanh_kernel[grid](
                input,
                y,
                n_elements,
            )

    return y
