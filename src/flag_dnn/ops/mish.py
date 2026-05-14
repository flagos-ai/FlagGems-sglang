import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_MISH_CONFIGS = runtime.get_tuned_config("mish")
_MISH_FP64_CONFIGS = runtime.get_tuned_config("mish")

# Triton 的 tanh 在不同版本里导入路径不一致
if tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 0):
    try:
        from triton.language.extra.libdevice import tanh as triton_tanh
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import tanh as triton_tanh
else:
    from triton.language.math import tanh as triton_tanh


@triton.jit
def _softplus_stable_fp32(x):
    abs_x = tl.abs(x)
    softplus_val = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    return tl.where(x > 20.0, x, softplus_val)


@triton.jit
def _softplus_stable_fp64(x):
    abs_x = tl.abs(x)
    softplus_val = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    return tl.where(x > 20.0, x, softplus_val)


@libentry()
@libtuner(
    configs=_MISH_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def mish_kernel(
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

    softplus_x = _softplus_stable_fp32(x)
    y = x * triton_tanh(softplus_x)

    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=_MISH_FP64_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def mish_fp64_kernel(
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

    softplus_x = _softplus_stable_fp64(x)
    y = x * triton_tanh(softplus_x)

    tl.store(y_ptr + offsets, y, mask=mask)


def mish(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    logger.debug("FLAG_DNN MISH")

    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn mish does not support dtype={input.dtype}"
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
            mish_fp64_kernel[grid](
                input,
                y,
                n_elements,
            )
        else:
            mish_kernel[grid](
                input,
                y,
                n_elements,
            )

    return y
