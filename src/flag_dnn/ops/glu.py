import logging
from math import prod

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


_GLU_CONFIGS = runtime.get_tuned_config("glu")
_GLU_FP64_CONFIGS = runtime.get_tuned_config("glu")

if tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 0):
    try:
        from triton.language.extra.libdevice import exp as triton_exp
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import exp as triton_exp
else:
    from triton.language.math import exp as triton_exp


@triton.jit
def _sigmoid(x):
    one = x * 0 + 1
    return one / (one + triton_exp(-x))


@libentry()
@libtuner(
    configs=_GLU_CONFIGS,
    key=["n_out_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def glu_kernel(
    x_ptr,
    y_ptr,
    n_out_elements,
    split_size,
    inner_size,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_out_elements

    inner_idx = offsets % inner_size
    tmp = offsets // inner_size
    split_idx = tmp % split_size
    outer_idx = tmp // split_size

    base = (
        outer_idx * (dim_size * inner_size)
        + split_idx * inner_size
        + inner_idx
    )
    a_offsets = base
    b_offsets = base + split_size * inner_size

    a = tl.load(x_ptr + a_offsets, mask=mask, other=0).to(tl.float32)
    b = tl.load(x_ptr + b_offsets, mask=mask, other=0).to(tl.float32)

    y = a * _sigmoid(b)

    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=_GLU_FP64_CONFIGS,
    key=["n_out_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def glu_fp64_kernel(
    x_ptr,
    y_ptr,
    n_out_elements,
    split_size,
    inner_size,
    dim_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_out_elements

    inner_idx = offsets % inner_size
    tmp = offsets // inner_size
    split_idx = tmp % split_size
    outer_idx = tmp // split_size

    base = (
        outer_idx * (dim_size * inner_size)
        + split_idx * inner_size
        + inner_idx
    )
    a_offsets = base
    b_offsets = base + split_size * inner_size

    a = tl.load(x_ptr + a_offsets, mask=mask, other=0).to(tl.float64)
    b = tl.load(x_ptr + b_offsets, mask=mask, other=0).to(tl.float64)

    y = a * _sigmoid(b)

    tl.store(y_ptr + offsets, y, mask=mask)


def glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logger.debug("FLAG_DNN GLU")

    if input.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise NotImplementedError(
            f"flag_dnn glu does not support dtype={input.dtype}"
        )

    if input.dim() == 0:
        raise RuntimeError("glu does not support a 0-dimensional tensor")

    ndim = input.dim()
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            f"Dimension out of range (got dim={dim}, ndim={ndim})"
        )

    dim_size = input.size(dim)
    if dim_size % 2 != 0:
        raise RuntimeError(
            "Halving dimension must be even,"
            f"but dimension {dim} is size {dim_size}"
        )

    if not input.is_contiguous():
        input = input.contiguous()

    split_size = dim_size // 2
    out_shape = list(input.shape)
    out_shape[dim] = split_size

    n_out_elements = prod(out_shape)
    if n_out_elements == 0:
        return torch.empty(out_shape, dtype=input.dtype, device=input.device)

    inner_size = prod(input.shape[dim + 1 :]) if dim < ndim - 1 else 1

    y = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    def grid(meta):
        return (triton.cdiv(n_out_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        if input.dtype == torch.float64:
            glu_fp64_kernel[grid](
                input,
                y,
                n_out_elements,
                split_size,
                inner_size,
                dim_size,
            )
        else:
            glu_kernel[grid](
                input,
                y,
                n_out_elements,
                split_size,
                inner_size,
                dim_size,
            )

    return y
