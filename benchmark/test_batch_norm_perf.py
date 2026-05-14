from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


# Batch Norm 通常在推理阶段 (training=False) 主要表现为 Element-wise 的仿射变换
def torch_batch_norm(x, running_mean, running_var, weight, bias):
    return F.batch_norm(
        x, running_mean, running_var, weight, bias, training=True
    )


def gems_batch_norm_wrapper(x, running_mean, running_var, weight, bias):
    return flag_dnn.ops.batch_norm(
        x, running_mean, running_var, weight, bias, training=True
    )


class BatchNormBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 针对 Batch Norm 定制的真实 CV 业务典型 shape
        shapes = [
            # 1. MLP / 1D 卷积输出 (N, C) 或 (N, C, L)
            (1024, 256),  # 典型 MLP 中间层
            (256, 1024),
            (32, 256, 1024),  # 1D 序列特征
            # 2. ResNet 系列典型特征图 (N, C, H, W)
            (32, 64, 112, 112),  # ResNet 前期特征图 (高分辨率)
            (32, 256, 56, 56),  # ResNet Stage 1 输出
            (32, 512, 28, 28),  # ResNet Stage 2 输出
            (32, 1024, 14, 14),  # ResNet Stage 3 输出
            (32, 2048, 7, 7),  # ResNet Stage 4 输出 (通道数多，空间小)
            # 3. 高分辨率/大 Batch Size 视觉任务 (如分割/检测)
            (8, 64, 512, 512),  # 语义分割典型输入特征
            (1, 16, 2048, 2048),  # 超高分辨率图像前处理
            # 4. 极限压力测试
            (16, 1024, 64, 64),  # 约 67M 元素
            (8, 2048, 64, 64),  # 约 67M 元素，通道极多
        ]
        self.shapes = shapes
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        # 设定单张 Tensor 的安全显存上限 (8 GB)
        MAX_TENSOR_BYTES = 8 * 1024**3

        for shape in self.shapes:
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            # 如果单张 Tensor 超过限制，跳过当前 shape，防止 OOM
            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            # 生成主特征图 x
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue

            # Batch Norm 至少需要 2D 张量，其中 dim=1 是通道维度 (C)
            num_channels = shape[1] if len(shape) > 1 else shape[0]

            # Batch Norm 需要的 4 个参数：均值、方差、缩放(weight)、平移(bias)
            # 为了防止数值溢出或 NAN，给均值和方差赋安全值
            running_mean = torch.zeros(
                num_channels, dtype=cur_dtype, device=self.device
            )
            running_var = torch.ones(
                num_channels, dtype=cur_dtype, device=self.device
            )
            weight = torch.ones(
                num_channels, dtype=cur_dtype, device=self.device
            )
            bias = torch.zeros(
                num_channels, dtype=cur_dtype, device=self.device
            )

            yield inp, running_mean, running_var, weight, bias

    def get_gbps(self, args, latency):
        inp = args[0]
        running_mean = args[1]
        running_var = args[2]
        weight = args[3]
        bias = args[4]

        # GBPS 计算：读入 x + 写出 y + 读入 4 个参数
        io_amount = (
            shape_utils.size_in_bytes(inp) * 2  # 读 x，写 y
            + shape_utils.size_in_bytes(running_mean)
            + shape_utils.size_in_bytes(running_var)
            + shape_utils.size_in_bytes(weight)
            + shape_utils.size_in_bytes(bias)
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.batch_norm
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_batch_norm(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = BatchNormBenchmark(
        op_name="batch_norm",
        torch_op=torch_batch_norm,
        gems_op=gems_batch_norm_wrapper,
        dtypes=[dtype],
    )
    bench.run()
