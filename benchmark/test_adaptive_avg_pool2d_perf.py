from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_adaptive_avg_pool2d(x, output_size):
    return F.adaptive_avg_pool2d(x, output_size=output_size)


def gems_adaptive_avg_pool2d_wrapper(x, output_size):
    return flag_dnn.ops.adaptive_avg_pool2d(x, output_size=output_size)


class AdaptiveAvgPool2dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: (shape, output_size)
        configs = [
            # --- 基础场景 (常见于CV模型如ResNet等) ---
            ((32, 256, 14, 14), 1),
            ((16, 512, 7, 7), (1, 1)),
            ((32, 128, 28, 28), 14),
            ((8, 64, 224, 224), 7),
            ((4, 128, 500, 500), 10),
            # --- 非整除场景 ---
            ((32, 128, 224, 224), 15),
            ((16, 64, 300, 300), (42, 42)),
            # --- 内存非对齐/奇数场景 ---
            ((16, 3, 224, 224), 1),
            ((32, 27, 112, 112), 16),
            ((8, 128, 223, 223), (14, 14)),
            # --- 边界与极端场景 ---
            ((1, 1, 1, 1), 1),  # Launch overhead 测试
            ((128, 16, 32, 32), 32),  # 输入等于输出大小
            ((1024, 64, 8, 8), 2),  # 大 Batch 场景
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

        # 处理 output_size 是 int 还是 tuple 的情况
        if isinstance(output_size, int):
            out_h, out_w = output_size, output_size
        else:
            out_h, out_w = output_size

        out_numel = inp.shape[0] * inp.shape[1] * out_h * out_w

        io_amount = shape_utils.size_in_bytes(inp) + (
            out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_adaptive_avg_pool2d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = AdaptiveAvgPool2dBenchmark(
        op_name="adaptive_avg_pool2d",
        torch_op=torch_adaptive_avg_pool2d,
        gems_op=gems_adaptive_avg_pool2d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
