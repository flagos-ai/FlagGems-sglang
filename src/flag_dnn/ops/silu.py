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
    configs=runtime.get_tuned_config("silu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def silu_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针 (如果 inplace=True，y_ptr 将等同于 x_ptr)
    n_elements,  # 张量总元素个数
    BLOCK_SIZE: tl.constexpr,  # 编译期常量：线程块大小
):
    # 计算当前线程处理的全局索引
    pid = tle.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建掩码，防止处理元素总数非 BLOCK_SIZE 整数倍时越界访问
    mask = offsets < n_elements

    # 从显存中加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    # 将数据 upcast 到 fp32 以满足 tl.sigmoid 的编译要求
    x_fp32 = x.to(tl.float32)

    # 计算 SiLU 激活函数: x * sigmoid(x)
    y_fp32 = x_fp32 * tl.sigmoid(x_fp32)
    # 将计算结果 downcast 回输入时的原始类型
    y = y_fp32.to(x.dtype)

    # 将计算结果写回显存
    # 如果是 inplace 操作，这里的 y_ptr 实际上就是 x_ptr，会直接覆盖原数据
    tl.store(y_ptr + offsets, y, mask=mask)


def silu(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    logger.debug(f"FLAG_DNN SILU (inplace={inplace})")

    assert x.is_contiguous(), "x must be contiguous"

    # 根据 inplace 参数决定是否复用输入张量的显存
    if inplace:
        y = x  # 原地修改，输出张量直接引用输入张量
    else:
        y = torch.empty_like(x)  # 非原地修改，新分配一块显存

    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # 启动 Triton Kernel
    with torch_device_fn.device(x.device):
        silu_kernel[grid](
            x,
            y,
            n_elements,
        )

    return y
