import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_dnn.runtime import torch_device_fn
from flag_dnn.utils import triton_lang_extension as tle


logger = logging.getLogger(__name__)


@triton.jit
def batch_norm_inference_kernel(
    x_ptr,
    y_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    total_elements,
    C,
    S,
    eps,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # 根据一维偏移量，反推当前元素属于哪个 Channel
    # 内存布局是 [N, C, S]，所以一维索引 flat_idx = n * (C * S) + c * S + s
    c_idx = (offsets // S) % C

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)

    mean = tl.load(mean_ptr + c_idx, mask=mask).to(tl.float32)
    var = tl.load(var_ptr + c_idx, mask=mask).to(tl.float32)
    weight = (
        tl.load(weight_ptr + c_idx, mask=mask).to(tl.float32)
        if HAS_WEIGHT
        else 1.0
    )
    bias = (
        tl.load(bias_ptr + c_idx, mask=mask).to(tl.float32)
        if HAS_BIAS
        else 0.0
    )

    rstd = 1.0 / tl.sqrt(var + eps)
    y = (x - mean) * rstd * weight + bias

    tl.store(y_ptr + offsets, y.to(x_ptr.dtype.element_ty), mask=mask)


def get_autotune_configs():
    return [
        # --- 1. 针对小 M / 小 S 的轻量配置 ---
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        # --- 2. 针对中等规模 ---
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        # --- 3. 针对超大 M (FP64 核心优化区) ---
        # 增加 num_warps 到 16，尝试在寄存器允许的情况下压榨带宽
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=2),
        # 极端情况：减少 stages 释放寄存器，提升 Occupancy
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=1),
        # 针对 512x512, 2048x2048 等超大平面
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
    ]


# @libentry()
# @libtuner(
#     configs=runtime.get_tuned_config("batch_norm"),
#     key=["S"],
#     strategy=["align32"],
#     warmup=5,
#     rep=10,
# )
@triton.autotune(
    configs=get_autotune_configs(),
    key=["N", "C", "S"],  # 根据 S (数据长度) 的不同，缓存不同的最优配置
    restore_value=["mean_ptr", "var_ptr"],
)
@triton.jit
def batch_norm_fused_kernel_optimized_(
    x_ptr,
    y_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    N,
    C,
    S,
    eps,
    momentum,
    BLOCK_SIZE: tl.constexpr,
    IS_TRAINING: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RUNNING_STATS: tl.constexpr,
):
    c = tl.program_id(0)
    M = N * S

    stride_gap = S * (C - 1)
    base_x_ptr = x_ptr + c * S
    base_y_ptr = y_ptr + c * S

    if IS_TRAINING:
        sum_x = 0.0
        sum_x2 = 0.0

        for i_offset in range(0, M, BLOCK_SIZE):
            i = i_offset + tl.arange(0, BLOCK_SIZE)
            mask = i < M

            # 计算正确的内存地址 (Stride 逻辑保持不变)
            mem_ptrs = base_x_ptr + i + (i // S) * stride_gap

            x = tl.load(mem_ptrs, mask=mask, other=0.0).to(tl.float32)

            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)

        mean = sum_x / M
        var = (sum_x2 / M) - (mean * mean)
        var = tl.maximum(var, 0.0)

        if HAS_RUNNING_STATS:
            rm = tl.load(mean_ptr + c).to(tl.float32)
            rv = tl.load(var_ptr + c).to(tl.float32)
            unbiased_var = var * (M / (M - 1)) if M > 1 else var
            new_rm = rm * (1.0 - momentum) + mean * momentum
            new_rv = rv * (1.0 - momentum) + unbiased_var * momentum
            tl.store(mean_ptr + c, new_rm.to(mean_ptr.dtype.element_ty))
            tl.store(var_ptr + c, new_rv.to(var_ptr.dtype.element_ty))
    else:
        mean = tl.load(mean_ptr + c).to(tl.float32)
        var = tl.load(var_ptr + c).to(tl.float32)

    # 归一化参数准备
    rstd = 1.0 / tl.sqrt(var + eps)
    weight = tl.load(weight_ptr + c).to(tl.float32) if HAS_WEIGHT else 1.0
    bias = tl.load(bias_ptr + c).to(tl.float32) if HAS_BIAS else 0.0

    for i_offset in range(0, M, BLOCK_SIZE):
        i = i_offset + tl.arange(0, BLOCK_SIZE)
        mask = i < M

        mem_ptrs = base_x_ptr + i + (i // S) * stride_gap
        x = tl.load(mem_ptrs, mask=mask).to(tl.float32)

        x_hat = (x - mean) * rstd
        y_f32 = x_hat * weight + bias

        y = y_f32.to(x_ptr.dtype.element_ty)
        out_ptrs = base_y_ptr + i + (i // S) * stride_gap
        tl.store(out_ptrs, y, mask=mask)


# def batch_norm(
#     input: torch.Tensor,
#     running_mean: Optional[torch.Tensor],
#     running_var: Optional[torch.Tensor],
#     weight: Optional[torch.Tensor] = None,
#     bias: Optional[torch.Tensor] = None,
#     training: bool = False,
#     momentum: float = 0.1,
#     eps: float = 1e-05,
# ) -> torch.Tensor:
def batch_norm_aten(
    input: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    cudnn_enabled: bool = True,
) -> torch.Tensor:
    logger.debug(f"FLAG_DNN FUSED BATCH_NORM (training={training}, eps={eps})")

    _ = cudnn_enabled

    if input.numel() == 0:
        return torch.empty_like(input)

    assert input.ndim >= 2, "BatchNorm requires at least 2D input (N, C, ...)"

    if not training:
        assert (
            running_mean is not None and running_var is not None
        ), "running stats must be provided in eval mode"

    # 内存连续性处理
    if not input.is_contiguous():
        # assert False, "input must be contiguous"
        input = input.contiguous()

    y = torch.empty_like(input)

    N = input.shape[0]
    C = input.shape[1]
    S = input.numel() // (N * C)
    total_elements = input.numel()

    # 简单的假指针，防止传入 None 时 Triton 报错
    dummy_ptr = torch.empty(0, device=input.device)
    mean_ptr = running_mean if running_mean is not None else dummy_ptr
    var_ptr = running_var if running_var is not None else dummy_ptr
    has_running_stats = running_mean is not None and running_var is not None

    grid = (C,)

    with torch_device_fn.device(input.device):
        if not training:
            # 根据总元素数计算一维 Grid 大小
            def grid(meta):  # type: ignore[misc]  # noqa: F811
                return (
                    triton.cdiv(
                        total_elements,
                        meta["BLOCK_SIZE"],
                    ),
                )

            batch_norm_inference_kernel[grid](
                input,
                y,
                mean_ptr,
                var_ptr,
                weight,
                bias,
                total_elements,
                C,
                S,
                eps,
                BLOCK_SIZE=1024,
                HAS_WEIGHT=(weight is not None),
                HAS_BIAS=(bias is not None),
            )
        else:
            batch_norm_fused_kernel_optimized_[grid](
                input,
                y,
                mean_ptr,
                var_ptr,
                weight,
                bias,
                N,
                C,
                S,
                eps,
                momentum,
                IS_TRAINING=training,
                HAS_WEIGHT=(weight is not None),
                HAS_BIAS=(bias is not None),
                HAS_RUNNING_STATS=has_running_stats,
            )

    return y


def batch_norm(
    input: torch.Tensor,
    running_mean: Optional[torch.Tensor],
    running_var: Optional[torch.Tensor],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    return batch_norm_aten(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        torch.backends.cudnn.enabled,
    )
