import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)

_NEG_TENSOR_CACHE: dict[
    tuple[str, int | None, torch.dtype, float], torch.Tensor
] = {}


def _get_neg_tensor(
    device: torch.device, dtype: torch.dtype, negative_slope: float
):
    key = (device.type, device.index, dtype, float(negative_slope))
    neg = _NEG_TENSOR_CACHE.get(key)
    if neg is None:
        neg = torch.tensor(float(negative_slope), dtype=dtype, device=device)
        _NEG_TENSOR_CACHE[key] = neg
    return neg


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("leaky_relu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def leaky_relu_kernel(
    x_ptr,
    y_ptr,
    neg_ptr,
    n_elements,
    COMPUTE_FP64: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    neg = tl.load(neg_ptr)

    if COMPUTE_FP64:
        x = x.to(tl.float64)
        neg = neg.to(tl.float64)
    else:
        x = x.to(tl.float32)
        neg = neg.to(tl.float32)

    y = tl.where(x > 0, x, x * neg)
    tl.store(y_ptr + offsets, y.to(y_ptr.dtype.element_ty), mask=mask)


def leaky_relu(
    x: torch.Tensor, negative_slope: float = 0.01, inplace: bool = False
) -> torch.Tensor:
    logger.debug(
        "FLAG_DNN LEAKY_RELU "
        f"(negative_slope={negative_slope}, inplace={inplace})"
    )

    assert x.is_contiguous(), "x must be contiguous"

    n_elements = x.numel()
    if n_elements == 0:
        return x if inplace else torch.empty_like(x)

    y = x if inplace else torch.empty_like(x)

    compute_fp64 = x.dtype == torch.float64
    compute_dtype = torch.float64 if compute_fp64 else torch.float32

    # 关键：不要每次新建，走缓存
    neg = _get_neg_tensor(x.device, compute_dtype, negative_slope)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(x.device):
        leaky_relu_kernel[grid](
            x,
            y,
            neg,
            n_elements,
            compute_fp64,
        )

    return y
