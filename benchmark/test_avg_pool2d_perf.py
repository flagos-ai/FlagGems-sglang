from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_avg_pool2d(x, kernel_size, stride, padding):
    return F.avg_pool2d(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )


def gems_avg_pool2d_wrapper(x, kernel_size, stride, padding):
    return flag_dnn.ops.avg_pool2d(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )


def _to_tuple2(val):
    """将输入的标量或元组统一转换为长度为 2 的元组 (H, W)"""
    if isinstance(val, int):
        return (val, val)
    return val


class AvgPool2dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 针对 2D Avg Pooling 定制的典型测试集。输入格式为 (Batch, Channels, Height, Width)
        configs = [
            # 1. 经典下采样 (Halving)
            ((128, 64, 56, 56), 2, 2, 0),
            ((64, 128, 56, 56), 2, 2, 0),
            # 2. 模拟 Global Average Pooling (GAP) - 各大经典网络最后的分类前处理
            ((128, 512, 7, 7), 7, 1, 0),  # ResNet-18/34
            ((64, 1024, 7, 7), 7, 1, 0),
            ((32, 1280, 7, 7), 7, 1, 0),  # MobileNetV2
            ((16, 1536, 10, 10), 10, 1, 0),  # InceptionV3
            # 3. 稍微大一点特征图的滑动平均池化 (如 DenseNet 中的 Transition Layer)
            ((32, 128, 56, 56), 2, 2, 0),
            ((16, 256, 56, 56), 2, 2, 0),
            # 4. 极端长宽比 / 大尺寸下采样 (如高分辨率遥感、医学图像切片)
            ((4, 32, 1024, 1024), 4, 4, 0),  # 大 kernel 大 stride
            ((8, 64, 256, 1024), 2, 2, 0),  # 极宽的特征图
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for config in self.shapes:
            shape, kernel_size, stride, padding = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue

            yield inp, kernel_size, stride, padding

    def get_gbps(self, args, latency):
        inp, kernel_size, stride, padding = args

        kh, kw = _to_tuple2(kernel_size)
        sh, sw = _to_tuple2(stride)
        ph, pw = _to_tuple2(padding)

        H_in, W_in = inp.shape[2], inp.shape[3]
        H_out = (H_in + 2 * ph - kh) // sh + 1
        W_out = (W_in + 2 * pw - kw) // sw + 1

        out_numel = inp.shape[0] * inp.shape[1] * H_out * W_out

        io_amount = shape_utils.size_in_bytes(inp) + (
            out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.avg_pool2d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_avg_pool2d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = AvgPool2dBenchmark(
        op_name="avg_pool2d",
        torch_op=torch_avg_pool2d,
        gems_op=gems_avg_pool2d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
