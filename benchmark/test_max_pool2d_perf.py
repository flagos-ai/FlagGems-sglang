from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_max_pool2d(x, kernel_size, stride, padding):
    return F.max_pool2d(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )


def gems_max_pool2d_wrapper(x, kernel_size, stride, padding):
    return flag_dnn.ops.max_pool2d(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )


def _to_tuple2(val):
    """将输入的标量或元组统一转换为长度为 2 的元组 (H, W)"""
    if isinstance(val, int):
        return (val, val)
    return val


class MaxPool2dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 针对 2D Max Pooling 定制的典型测试集。输入格式为 (Batch, Channels, Height, Width)
        # 配置格式为: (shape, kernel_size, stride, padding)
        configs = [
            # 1. 经典 VGG/ResNet 阶段下采样 (Halving, 覆盖浅层到深层)
            ((128, 64, 56, 56), 2, 2, 0),
            ((64, 128, 56, 56), 2, 2, 0),
            ((32, 256, 14, 14), 2, 2, 0),
            ((16, 512, 7, 7), 2, 2, 0),
            # 2. ResNet Stem 阶段的大核带 padding 池化
            ((64, 64, 112, 112), 3, 2, 1),
            # 3. 保持尺寸的重叠池化 (Overlapping pooling, 常见于 Inception/AlexNet)
            ((64, 192, 28, 28), 3, 1, 1),
            ((32, 256, 28, 28), 3, 1, 1),
            # 4. 高分辨率输入 (如目标检测/语义分割 Backbone)
            ((8, 64, 800, 800), 2, 2, 0),  # 常见的目标检测输入尺寸
            ((4, 128, 1080, 1920), 2, 2, 0),  # 1080p FHD 分辨率非对称输入
            # 5. 非对称 Kernel 和 Stride (如语音声学特征频谱图处理、特定遥感图像)
            ((32, 1, 128, 256), (2, 3), (2, 2), (0, 1)),
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

        # 2D Pooling 的输出宽高计算公式
        H_in, W_in = inp.shape[2], inp.shape[3]
        H_out = (H_in + 2 * ph - kh) // sh + 1
        W_out = (W_in + 2 * pw - kw) // sw + 1

        # 输出的 numel = Batch * Channels * H_out * W_out
        out_numel = inp.shape[0] * inp.shape[1] * H_out * W_out

        io_amount = (
            shape_utils.size_in_bytes(inp) + out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.max_pool2d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_max_pool2d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = MaxPool2dBenchmark(
        op_name="max_pool2d",
        torch_op=torch_max_pool2d,
        gems_op=gems_max_pool2d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
