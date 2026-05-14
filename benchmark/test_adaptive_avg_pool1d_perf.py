from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_adaptive_avg_pool1d(x, output_size):
    return F.adaptive_avg_pool1d(x, output_size=output_size)


def gems_adaptive_avg_pool1d_wrapper(x, output_size):
    return flag_dnn.ops.adaptive_avg_pool1d(x, output_size=output_size)


class AdaptiveAvgPool1dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: (shape, output_size)
        configs = [
            # --- 基础场景 ---
            ((32, 256, 1024), 1),
            ((8, 512, 4096), 1),
            ((32, 128, 1024), 32),
            ((64, 64, 512), 16),
            ((4, 128, 16000), 100),
            ((1, 256, 48000), 256),
            # --- 非整除场景 ---
            ((32, 128, 1024), 15),
            ((16, 64, 733), 42),
            # --- 内存非对齐/奇数场景 ---
            ((16, 3, 1024), 1),
            ((32, 27, 512), 16),
            ((8, 128, 1023), 32),
            # --- 边界与极端场景 ---
            ((1, 1, 1), 1),  # Launch overhead 测试
            ((128, 16, 32), 32),  # 输入等于输出大小
            ((1024, 64, 64), 8),  # 大 Batch 场景
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

        # Adaptive Pooling 输出直接由指定的 output_size 决定
        out_numel = inp.shape[0] * inp.shape[1] * output_size

        io_amount = shape_utils.size_in_bytes(inp) + (
            out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.adaptive_avg_pool1d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_adaptive_avg_pool1d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = AdaptiveAvgPool1dBenchmark(
        op_name="adaptive_avg_pool1d",
        torch_op=torch_adaptive_avg_pool1d,
        gems_op=gems_adaptive_avg_pool1d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
