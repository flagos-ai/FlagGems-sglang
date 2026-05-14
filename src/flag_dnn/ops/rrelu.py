import logging
import math

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


# 仅用于 float64 inference path，避免每次都重新创建 0-dim tensor
_NEG_TENSOR_CACHE: dict[
    tuple[str, int | None, torch.dtype, float], torch.Tensor
] = {}


def _get_neg_tensor(
    device: torch.device,
    dtype: torch.dtype,
    negative_slope: float,
) -> torch.Tensor:
    key = (device.type, device.index, dtype, float(negative_slope))
    neg = _NEG_TENSOR_CACHE.get(key)
    if neg is None:
        neg = torch.tensor(float(negative_slope), dtype=dtype, device=device)
        _NEG_TENSOR_CACHE[key] = neg
    return neg


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("threshold"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def rrelu_inference_kernel(
    x_ptr,  # 输入张量
    y_ptr,  # 输出张量
    n_elements,  # 元素总数
    negative_slope,  # 推理态固定 slope，fast path
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, x * negative_slope)

    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("threshold"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def rrelu_inference_kernel_fp64(
    x_ptr,  # 输入张量
    neg_ptr,  # 0-dim / 1-element slope tensor
    y_ptr,  # 输出张量
    n_elements,  # 元素总数
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    negative_slope = tl.load(neg_ptr)
    y = tl.where(x > 0, x, x * negative_slope)

    tl.store(y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("threshold"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def rrelu_training_kernel(
    x_ptr,  # 输入张量
    slope_ptr,  # 每元素随机 slope
    y_ptr,  # 输出张量
    n_elements,  # 元素总数
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    slope = tl.load(slope_ptr + offsets, mask=mask)
    y = tl.where(x > 0, x, x * slope)

    tl.store(y_ptr + offsets, y, mask=mask)


def rrelu(
    input: torch.Tensor,
    lower: float = 1.0 / 8,
    upper: float = 1.0 / 3,
    training: bool = False,
    inplace: bool = False,
) -> torch.Tensor:
    """
    torch.nn.functional.rrelu(
        input, lower=1./8, upper=1./3, training=False, inplace=False
    ) -> Tensor

    PyTorch 语义：
    - training=False: 等价于 leaky_relu，negative_slope = (lower + upper) / 2
    - training=True : 对负值位置采样 [lower, upper] 的随机 slope
    """
    logger.debug("FLAG_DNN RRELU")

    lower = float(lower)
    upper = float(upper)

    if not math.isfinite(lower):
        raise ValueError(f"rrelu: lower bound must be finite, got {lower}")
    if not math.isfinite(upper):
        raise ValueError(f"rrelu: upper bound must be finite, got {upper}")
    if lower > upper:
        raise ValueError(
            f"Lower bound should be less than or equal to the upper bound, "
            f"but got lower={lower}, upper={upper}"
        )

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

    with torch_device_fn.device(input.device):
        if training:
            # training=True 时，每个元素一个随机 slope
            # 这条路径的重点是语义正确，不是极致 unary fast path
            slopes = torch.empty_like(input).uniform_(lower, upper)
            rrelu_training_kernel[grid](
                input,
                slopes,
                y,
                n_elements,
            )
        else:
            negative_slope = (lower + upper) * 0.5

            # 性能关键：只有 float64 才走 cached tensor path 修精度
            # 其余 dtype 继续走 scalar fast path
            if input.dtype == torch.float64:
                neg_tensor = _get_neg_tensor(
                    input.device, input.dtype, negative_slope
                )
                rrelu_inference_kernel_fp64[grid](
                    input,
                    neg_tensor,
                    y,
                    n_elements,
                )
            else:
                rrelu_inference_kernel[grid](
                    input,
                    y,
                    n_elements,
                    negative_slope,
                )

    if need_copy_back:
        orig_input.copy_(y)
        return orig_input

    return y
