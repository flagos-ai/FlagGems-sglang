import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn import runtime
from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import libentry, libtuner
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


@triton.jit
def fast_softmax_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    N,  # 需要进行 softmax 的维度的大小
    stride_x_row,  # 输入张量每一行的跨度
    stride_y_row,  # 输出张量每一行的跨度
    BLOCK_SIZE: tl.constexpr,  # 编译期常量：线程块大小（必须是 2 的幂）
):
    # 获取当前处理的行索引 (pid)
    pid = tle.program_id(0)

    # 计算当前行的起始内存地址
    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    # 生成列索引和掩码
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # 加载数据
    # 越界部分使用 -inf 填充，这样在计算 exp(-inf) 时它们会变成 0，完全不影响 sum 的计算
    x = tl.load(row_x_ptr + offsets, mask=mask, other=-float("inf"))

    # 防护机制：将低精度数据提升为 fp32 进行复杂非线性运算
    x_fp32 = x.to(tl.float32)

    # 计算最大值 (Numerical Stability 技巧)
    row_max = tl.max(x_fp32, axis=0)

    # 计算分子: exp(x_i - max(x))
    numerator = tl.exp(x_fp32 - row_max)

    # 计算分母: sum_j(exp(x_j - max(x)))
    denominator = tl.sum(numerator, axis=0)

    # 计算 Softmax 并降级回原始数据类型
    softmax_val = numerator / denominator
    y = softmax_val.to(x.dtype)

    # 写回显存
    tl.store(row_y_ptr + offsets, y, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("softmax"),
    key=["N"],
    strategy=["align32"],
    warmup=5,
    rep=10,
)
@triton.jit
def online_softmax_kernel(
    x_ptr,
    y_ptr,
    N,
    stride_x_row,
    stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tle.program_id(0)
    row_x_ptr = x_ptr + pid * stride_x_row
    row_y_ptr = y_ptr + pid * stride_y_row

    # Online 计算全局最大值 (m_i) 和全局指数和 (d_i)
    m_i = -float("inf")  # 运行时的全局最大值
    d_i = 0.0  # 运行时的全局指数和

    ptrs = row_x_ptr + tl.arange(0, BLOCK_SIZE)

    for offset in range(0, N, BLOCK_SIZE):
        mask = (offset + tl.arange(0, BLOCK_SIZE)) < N

        # 加载块数据
        x = tl.load(ptrs, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)

        # 计算局部最大值
        m_block = tl.max(x_fp32, axis=0)

        # 更新全局最大值
        m_new = tl.maximum(m_i, m_block)

        # 计算旧 denominator 的衰减系数 alpha
        alpha = tl.exp(m_i - m_new)

        # 计算当前块的指数和
        # 使用 tl.where 确保越界的部分绝对是 0.0，避免参与 sum 计算
        exp_vals = tl.where(mask, tl.exp(x_fp32 - m_new), 0.0)
        d_block = tl.sum(exp_vals, axis=0)

        # 更新全局指数和
        d_i = d_i * alpha + d_block
        m_i = m_new

        ptrs += BLOCK_SIZE

    # 计算最终的 Softmax 概率
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        # 第二次从 HBM 加载数据
        x = tl.load(row_x_ptr + cols, mask=mask, other=-float("inf"))
        x_fp32 = x.to(tl.float32)

        # 计算概率: e^(x - m_max) / d_sum
        out = tl.exp(x_fp32 - m_i) / d_i

        # 降精度并存回显存
        out = out.to(x_ptr.dtype.element_ty)
        tl.store(row_y_ptr + cols, out, mask=mask)


def softmax(
    input: torch.Tensor,
    dim: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN SOFTMAX (dim={dim}, dtype={dtype})")

    # 处理 dtype 转换
    x = input
    if dtype is not None:
        x = x.to(dtype)

    # 如果是空张量，直接返回，跳过后续所有逻辑
    if x.numel() == 0:
        return torch.empty_like(x)

    # 处理维度的默认值
    if dim is None:
        dim = -1
    if dim < 0:
        dim = x.ndim + dim

    # 将 target dim 置换到最后一维，方便映射为 2D 矩阵结构
    need_transpose = dim != x.ndim - 1
    if need_transpose:
        x = x.transpose(dim, -1)

    # 必须保证内存连续，才能安全使用简单的偏移计算
    if not x.is_contiguous():
        x = x.contiguous()

    # 计算二维视图下的 M (行数) 和 N (列数)
    N = x.shape[-1]
    M = x.numel() // N

    y = torch.empty_like(x)

    grid = (M,)

    MAX_FAST_BLOCK = 1024

    # 启动 Triton Kernel
    with torch_device_fn.device(x.device):
        if N <= MAX_FAST_BLOCK:
            BLOCK_SIZE = triton.next_power_of_2(N)
            fast_softmax_kernel[grid](x, y, N, N, N, BLOCK_SIZE=BLOCK_SIZE)
        else:
            online_softmax_kernel[grid](
                x,
                y,
                N,
                N,  # stride_x_row (因为已经 contiguous 了，行跨度等于 N)
                N,  # stride_y_row
            )

    # 维度置换
    if need_transpose:
        y = y.transpose(dim, -1).contiguous()

    return y
