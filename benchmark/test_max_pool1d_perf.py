from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_max_pool1d(x, kernel_size, stride, padding):
    return F.max_pool1d(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )


def gems_max_pool1d_wrapper(x, kernel_size, stride, padding):
    return flag_dnn.ops.max_pool1d(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )


class MaxPool1dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 针对 1D Pooling 定制的典型测试集。输入格式为 (Batch, Channels, Length)
        # 配置格式为: (shape, kernel_size, stride, padding)
        configs = [
            # 1. 典型的下采样降维 (Halving)
            ((32, 128, 1024), 2, 2, 0),
            ((64, 256, 512), 2, 2, 0),
            # 2. 保持原序列长度的滑动窗口 (TextCNN 常用)
            ((32, 64, 4096), 3, 1, 1),
            ((16, 512, 1024), 5, 1, 2),
            # 3. 大核步长 (Aggressive Pooling, 音频特征提取常用)
            ((8, 128, 16000), 10, 5, 0),  # 例如处理 1 秒 16kHz 的音频波形
            ((1, 64, 48000), 100, 50, 0),
            # 4. 非整除、奇数长度和最小边界
            ((16, 3, 1023), 7, 3, 2),
            ((8, 27, 733), 5, 2, 1),
            ((1, 1, 1), 1, 1, 0),
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

        # 1D Pooling 的输出长度计算公式
        L_in = inp.shape[-1]
        L_out = (L_in + 2 * padding - kernel_size) // stride + 1

        # 输出的 numel = Batch * Channels * L_out
        out_numel = inp.shape[0] * inp.shape[1] * L_out

        io_amount = (
            shape_utils.size_in_bytes(inp) + out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.max_pool1d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_max_pool1d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = MaxPool1dBenchmark(
        op_name="max_pool1d",
        torch_op=torch_max_pool1d,
        gems_op=gems_max_pool1d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
