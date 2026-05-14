from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_adaptive_max_pool1d(x, output_size):
    return torch.adaptive_max_pool1d(x, output_size=output_size)


def gems_adaptive_max_pool1d_wrapper(x, output_size):
    return flag_dnn.ops.adaptive_max_pool1d(x, output_size=output_size)


class AdaptiveMaxPool1dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: (shape, output_size)
        configs = [
            # 1. 全局池化 (Global 1D Pooling)，提取序列最强特征，在分类任务中最常见
            ((32, 256, 1024), 1),
            ((8, 512, 4096), 1),
            # 2. 降维到指定的特征长度
            ((32, 128, 1024), 64),
            ((64, 64, 512), 32),
            # 3. 对极长音频特征或文本序列做粗粒度压缩
            ((4, 128, 16000), 100),
            ((1, 256, 48000), 256),
            # 4. 非整除、非对齐和边界场景
            ((16, 3, 1023), 17),
            ((8, 27, 733), 41),
            ((1, 1, 1), 1),
            ((128, 16, 257), 128),
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

        # 对于 Adaptive Pooling, 输出的最后一个维度就是指定的 output_size
        out_numel = inp.shape[0] * inp.shape[1] * output_size

        io_amount = shape_utils.size_in_bytes(inp) + (
            out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.adaptive_max_pool1d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_adaptive_max_pool1d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = AdaptiveMaxPool1dBenchmark(
        op_name="adaptive_max_pool1d",
        torch_op=torch_adaptive_max_pool1d,
        gems_op=gems_adaptive_max_pool1d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
