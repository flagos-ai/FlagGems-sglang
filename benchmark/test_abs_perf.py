from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_abs(x):
    return torch.abs(x)


def gems_abs_wrapper(x):
    return flag_dnn.ops.abs(x)


class AbsBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        configs = [
            # 1. 极小 Shape
            (1,),
            (16,),
            (64,),
            # 2. 奇葩 / 非对齐 / 质数 Shape
            (127,),
            (1023, 1025),
            (7, 31, 109),
            (33, 129, 257),
            # 3. 经典 NLP / LLM 尺寸 (Batch, SeqLen, Hidden)
            (1, 2048, 4096),  # 类似 Llama 推理
            (8, 128, 12288),  # 短序列大宽度的 LLM
            (4, 4096, 4096),  # 长序列压力测试
            # 4. 经典 CV 尺寸 (NCHW格式)
            (1, 3, 224, 224),  # ImageNet 单张输入
            (32, 256, 56, 56),  # 典型特征图
            (16, 1024, 14, 14),  # 深层密集特征图
            # 5. 高维张量测试 (如 5D 的医疗图像或视频 Video 算子)
            (2, 16, 32, 64, 64),
            # 6. 大容量与极限 HBM 带宽压力测试
            (1024 * 256,),  # 256K 常规向量
            (1024 * 1024 * 16,),  # 16M 元素 1D
            (8192, 8192),  # 64M 元素 2D
            (1024 * 1024 * 64,),  # 64M 元素 1D 线性
            (2, 8192, 8192),  # 1.34亿元素
        ]
        self.shapes = [(shape,) for shape in configs]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for (shape_x,) in self.shapes:
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            total_bytes = np.prod(shape_x) * element_size * 2

            if total_bytes > MAX_TENSOR_BYTES:
                continue

            x = torch.randn(shape_x, dtype=cur_dtype, device=self.device)
            if x.numel() == 0:
                continue
            yield (x,)

    def get_gbps(self, args, latency):
        x = args[0]
        io_amount = shape_utils.size_in_bytes(x) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.abs
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_abs(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = AbsBenchmark(
        op_name="abs",
        torch_op=torch_abs,
        gems_op=gems_abs_wrapper,
        dtypes=[dtype],
    )
    bench.run()
