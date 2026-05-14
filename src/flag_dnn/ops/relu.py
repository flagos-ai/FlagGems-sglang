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
    configs=runtime.get_tuned_config("relu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def relu_1d_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 纯 1D 极简并行
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载数据
    x = tl.load(in_ptr + offsets, mask=mask, other=0)

    # ReLU 核心逻辑：x > 0 则保留 x，否则填 0
    # 使用 tl.where 非常安全，Triton 会自动处理 0 的隐式类型转换，不会报错
    out = tl.where(x > 0, x, 0)

    tl.store(out_ptr + offsets, out, mask=mask)


def relu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    logger.debug("FLAG_DNN RELU")
    # 空张量处理
    if input.numel() == 0:
        if inplace:
            return input
        return torch.empty_like(input)

    # 内存连续性处理
    if not input.is_contiguous():
        input = input.contiguous()

    # Inplace 逻辑控制
    if inplace:
        out = input
    else:
        out = torch.empty_like(input)

    n_elements = input.numel()

    # Grid 只需要一维，计算出需要多少个 Block 能覆盖所有的元素
    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 启动 Kernel
    with torch_device_fn.device(input.device):
        relu_1d_kernel[grid](input, out, n_elements)

    return out
