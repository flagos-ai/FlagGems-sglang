import logging

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("threshold"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def threshold_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    n_elements,  # 张量总元素个数
    threshold_val,  # 阈值 (标量)
    value_val,  # 替换值 (标量)
    BLOCK_SIZE: tl.constexpr,  # 编译期常量：线程块大小
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x > threshold_val, x, value_val)

    tl.store(y_ptr + offsets, y, mask=mask)


def threshold(
    input: torch.Tensor, threshold: float, value: float, inplace: bool = False
) -> torch.Tensor:
    logger.debug("FLAG_DNN THRESHOLD")

    if not input.is_contiguous():
        input = input.contiguous()

    n_elements = input.numel()

    if n_elements == 0:
        return input if inplace else torch.empty_like(input)

    if inplace:
        y = input
    else:
        y = torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        threshold_kernel[grid](
            input,
            y,
            n_elements,
            threshold,
            value,
        )

    return y
