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
    configs=runtime.get_tuned_config("hardtanh"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def hardtanh_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    n_elements,  # 元素总数
    min_val,  # hardtanh 下界
    max_val,  # hardtanh 上界
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    # 高性能实现：branchless clamp
    y = tl.minimum(tl.maximum(x, min_val), max_val)

    tl.store(y_ptr + offsets, y, mask=mask)


def hardtanh(
    input: torch.Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
    inplace: bool = False,
) -> torch.Tensor:
    logger.debug("FLAG_DNN HARDTANH")

    min_val = float(min_val)
    max_val = float(max_val)

    if min_val > max_val:
        raise ValueError(
            f"min_val must be less than or equal to max_val, but got "
            f"min_val={min_val}, max_val={max_val}"
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

    if inplace:
        y = input
    else:
        y = torch.empty_like(input)

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    with torch_device_fn.device(input.device):
        hardtanh_kernel[grid](
            input,
            y,
            n_elements,
            min_val,
            max_val,
        )

    if need_copy_back:
        orig_input.copy_(y)
        return orig_input

    return y
