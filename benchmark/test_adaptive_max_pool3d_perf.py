from typing import Generator

import numpy as np
import pytest
import torch
import torch._C._nn as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_adaptive_max_pool3d(x, output_size):
    return F.adaptive_max_pool3d(x, output_size=output_size)


def gems_adaptive_max_pool3d_wrapper(x, output_size):
    return flag_dnn.ops.adaptive_max_pool3d(x, output_size=output_size)


class AdaptiveMaxPool3dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: ((N, C, D, H, W), (oD, oH, oW))
        configs = [
            # 1. 全局池化 (Global 3D Pooling)，提取视频/体积数据最强时空特征 (如 I3D 视频分类结尾)
            ((4, 1024, 16, 7, 7), (1, 1, 1)),
            ((8, 512, 8, 14, 14), (1, 1, 1)),
            # 2. 时空特征降维（针对高维度的视频流或 3D 医学影像）
            ((2, 256, 32, 56, 56), (16, 28, 28)),
            ((4, 128, 16, 64, 64), (8, 32, 32)),
            # 3. 仅对空间或时间维度进行压缩 (时间维压缩、空间维保持)
            ((1, 64, 64, 128, 128), (16, 128, 128)),
            ((1, 32, 128, 256, 256), (32, 64, 64)),
            # 4. 非整除、奇数和边界场景
            ((4, 3, 15, 111, 109), (7, 17, 13)),
            ((2, 27, 9, 55, 57), (4, 11, 9)),
            ((1, 1, 1, 1, 1), (1, 1, 1)),
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

        # 对于 Adaptive Max Pool 3D, 输出的空间维度是 output_size (oD, oH, oW)
        # inp.shape[0] 是 N, inp.shape[1] 是 C
        out_numel = (
            inp.shape[0]
            * inp.shape[1]
            * output_size[0]
            * output_size[1]
            * output_size[2]
        )

        io_amount = shape_utils.size_in_bytes(inp) + (
            out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_adaptive_max_pool3d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = AdaptiveMaxPool3dBenchmark(
        op_name="adaptive_max_pool3d",
        torch_op=torch_adaptive_max_pool3d,
        gems_op=gems_adaptive_max_pool3d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
