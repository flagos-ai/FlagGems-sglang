from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_adaptive_avg_pool3d(x, output_size):
    return F.adaptive_avg_pool3d(x, output_size=output_size)


def gems_adaptive_avg_pool3d_wrapper(x, output_size):
    return flag_dnn.ops.adaptive_avg_pool3d(x, output_size=output_size)


class AdaptiveAvgPool3dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: (shape, output_size)
        configs = [
            # --- 基础场景 (常见于视频/医学图像 3D-CNN) ---
            ((4, 256, 16, 14, 14), 1),
            ((2, 512, 8, 7, 7), (1, 1, 1)),
            ((4, 128, 16, 28, 28), 14),
            ((2, 64, 32, 112, 112), (8, 7, 7)),
            # --- 非整除场景 ---
            ((4, 128, 16, 112, 112), 15),
            ((2, 64, 10, 150, 150), (4, 42, 42)),
            # --- 内存非对齐/奇数场景 ---
            ((4, 3, 16, 112, 112), 1),
            ((8, 27, 8, 56, 56), 7),
            ((2, 128, 15, 111, 111), (7, 14, 14)),
            # --- 边界与极端场景 ---
            ((1, 1, 1, 1, 1), 1),  # Launch overhead 测试
            ((32, 16, 8, 16, 16), 8),  # 输入等于/接近输出大小
            ((128, 64, 4, 4, 4), 2),  # 大 Batch 场景
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
            out_d, out_h, out_w = output_size, output_size, output_size
        else:
            out_d, out_h, out_w = output_size

        out_numel = inp.shape[0] * inp.shape[1] * out_d * out_h * out_w

        io_amount = shape_utils.size_in_bytes(inp) + (
            out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_adaptive_avg_pool3d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = AdaptiveAvgPool3dBenchmark(
        op_name="adaptive_avg_pool3d",
        torch_op=torch_adaptive_avg_pool3d,
        gems_op=gems_adaptive_avg_pool3d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
