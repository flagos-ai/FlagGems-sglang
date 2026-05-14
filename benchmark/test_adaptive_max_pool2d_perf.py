from typing import Generator

import numpy as np
import pytest
import torch
import torch._C._nn as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_adaptive_max_pool2d(x, output_size):
    return F.adaptive_max_pool2d(x, output_size=output_size)


def gems_adaptive_max_pool2d_wrapper(x, output_size):
    return flag_dnn.ops.adaptive_max_pool2d(x, output_size=output_size)


class AdaptiveMaxPool2dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: ((N, C, H, W), (oH, oW))
        configs = [
            # 1. 全局池化 (Global 2D Pooling)，提取特征图最强特征，如 ResNet 等分类网络最后阶段
            ((32, 2048, 7, 7), (1, 1)),
            ((128, 512, 14, 14), (1, 1)),
            # 2. 空间降维到特定的宽高特征大小，常见于目标检测或特征金字塔
            ((16, 256, 112, 112), (56, 56)),
            ((32, 128, 64, 64), (32, 32)),
            # 3. 针对非对称尺寸的图像输入处理 (如长宽不等的特征图)
            ((8, 32, 128, 256), (32, 64)),
            ((4, 32, 1080, 1920), (270, 480)),
            # 4. 非整除、奇数和边界场景
            ((16, 16, 223, 225), (17, 19)),
            ((8, 27, 111, 109), (14, 13)),
            ((1, 1, 1, 1), (1, 1)),
            ((64, 16, 15, 15), (7, 5)),
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for config in self.shapes:
            shape, output_size = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue

            yield inp, output_size

    def get_gbps(self, args, latency):
        inp, output_size = args

        # 对于 Adaptive Max Pool 2D, 输出的空间维度是 output_size (oH, oW)
        # inp.shape[0] 是 N (batch), inp.shape[1] 是 C (channel)
        out_numel = (
            inp.shape[0] * inp.shape[1] * output_size[0] * output_size[1]
        )

        io_amount = shape_utils.size_in_bytes(inp) + (
            out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.adaptive_max_pool2d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_adaptive_max_pool2d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = AdaptiveMaxPool2dBenchmark(
        op_name="adaptive_max_pool2d",
        torch_op=torch_adaptive_max_pool2d,
        gems_op=gems_adaptive_max_pool2d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
