from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_prelu(x, weight):
    return F.prelu(x, weight)


def gems_prelu_wrapper(x, weight):
    return flag_dnn.ops.prelu(x, weight)


class PreluBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        shapes = [
            # 1. 基础小尺寸 & 1D (测试启动开销)
            (1024,),
            (65536,),
            # 2. NLP / LLM 典型 shape (2D/3D)
            (32, 4096),  # (Batch, Hidden_dim) - Linear 层输出
            (8192, 8192),  # 大模型 FFN 内部全连接
            (16, 2048, 4096),  # (Batch, Seq_len, Hidden_dim)
            # 3. LLM Attention Softmax 典型 shape (4D)
            (8, 32, 2048, 2048),  # (Batch, Num_heads, Seq_len, Seq_len)
            (4, 64, 4096, 4096),  # 更长的 Context length
            # 4. 计算机视觉 (CV) 典型 shape (4D)
            (32, 64, 224, 224),  # ResNet 前期特征图 (N, C, H, W)
            (16, 256, 64, 64),  # ResNet 中期特征图
            (8, 1024, 14, 14),  # ResNet 后期特征图
            # 5. 极限压力测试
            (4, 1024, 1024, 16),  # 约 67M 元素
            (2, 2048, 2048, 16),  # 约 134M 元素
        ]
        self.shapes = shapes
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        # 设定单张 Tensor 的安全显存上限 (8 GB)
        MAX_TENSOR_BYTES = 8 * 1024**3

        for shape in self.shapes:
            # 显存拦截：提前计算当前 shape 和 dtype 下的体积
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            # 如果单张 Tensor 超过限制，跳过当前 shape，防止 OOM
            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp1.numel() == 0:
                continue

            if len(shape) == 1:
                # 1D 张量，PyTorch 视其 channel size 为 1
                num_channels = 1
            else:
                # 2D 及以上张量，PyTorch 默认 dim=1 为通道维度 (N, C, ...)
                num_channels = shape[1]

            # 生成对应通道数的 weight
            weight = torch.full(
                (num_channels,), 0.25, dtype=cur_dtype, device=self.device
            )

            yield inp1, weight

    def get_gbps(self, args, latency):
        inp1 = args[0]
        weight = args[1]

        # GBPS 计算：读入 x + 读入 weight + 写出 y
        io_amount = (
            shape_utils.size_in_bytes(inp1)
            + shape_utils.size_in_bytes(weight)
            + shape_utils.size_in_bytes(inp1)
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.prelu
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_prelu(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = PreluBenchmark(
        op_name="prelu",
        torch_op=torch_prelu,
        gems_op=gems_prelu_wrapper,
        dtypes=[dtype],
    )
    bench.run()
