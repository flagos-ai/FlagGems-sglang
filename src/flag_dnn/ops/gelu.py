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
    configs=runtime.get_tuned_config("gelu"),
    key=["n_elements"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def gelu_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    n_elements,  # 张量总元素个数
    APPROXIMATE: tl.constexpr,  # 编译期常量：判断是否使用 tanh 近似
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

    # 根据传入的模式计算 GELU
    if APPROXIMATE:
        # Tanh 近似版公式
        # GELU(x) = 0.5 * x * (1 + Tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        sqrt_2_over_pi = 0.7978845608  # sqrt(2/pi) 的近似值
        cube = x * x * x
        inner = sqrt_2_over_pi * (x + 0.044715 * cube)
        # tanh_val = tl.math.tanh(inner)
        # 替换为基于 sigmoid 的等价实现
        tanh_val = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        y = 0.5 * x * (1.0 + tanh_val)
    else:
        # 精确版公式 (None)
        # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        inv_sqrt_2 = 0.7071067811  # 1/sqrt(2) 的近似值
        erf_val = tl.math.erf(x * inv_sqrt_2)
        y = 0.5 * x * (1.0 + erf_val)

    # 将计算结果写回显存
    tl.store(y_ptr + offsets, y, mask=mask)


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    logger.debug("FLAG_DNN GELU")

    assert x.is_contiguous(), "x must be contiguous"
    assert approximate in [
        "none",
        "tanh",
    ], "approximate must be 'none' or 'tanh'"

    # 预分配输出显存
    y = torch.empty_like(x)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    is_approximate = approximate == "tanh"

    # 启动 Triton Kernel
    with torch_device_fn.device(x.device):
        gelu_kernel[grid](
            x,
            y,
            n_elements,
            APPROXIMATE=is_approximate,
        )

    return y
