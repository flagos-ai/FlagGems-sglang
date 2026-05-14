import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_CELU_CONFIGS = runtime.get_tuned_config("celu")
_CELU_FP64_CONFIGS = runtime.get_tuned_config("celu")

# Triton 的 exp 在不同版本里导入路径不一致
if tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 0):
    try:
        from triton.language.extra.libdevice import exp as triton_exp
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import exp as triton_exp
else:
    from triton.language.math import exp as triton_exp


@triton.jit
def _celu(x, alpha):
    zero = x * 0
    one = zero + 1
    alpha_v = zero + alpha
    return tl.where(x > zero, x, alpha_v * (triton_exp(x / alpha_v) - one))


@libentry()
@libtuner(
    configs=_CELU_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def celu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float32)
    y = _celu(x, alpha)

    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=_CELU_FP64_CONFIGS,
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def celu_fp64_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.float64)
    y = _celu(x, alpha)

    tl.store(y_ptr + offsets, y, mask=mask)


def celu(
    input: torch.Tensor,
    alpha: float = 1.0,
    inplace: bool = False,
) -> torch.Tensor:
    logger.debug("FLAG_DNN CELU")

    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn celu does not support dtype={input.dtype}"
        )

    alpha = float(alpha)
    if alpha == 0.0:
        raise ValueError("alpha must be non-zero for celu")

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
            celu_fp64_kernel[grid](
                input,
                y,
                n_elements,
                alpha,
            )
        else:
            celu_kernel[grid](
                input,
                y,
                n_elements,
                alpha,
            )

    if need_copy_back:
        orig_input.copy_(y)
        return orig_input

    return y
