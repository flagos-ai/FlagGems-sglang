from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_max_pool3d(x, kernel_size, stride, padding):
    return F.max_pool3d(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )


def gems_max_pool3d_wrapper(x, kernel_size, stride, padding):
    return flag_dnn.ops.max_pool3d(
        x, kernel_size=kernel_size, stride=stride, padding=padding
    )


def _to_tuple3(val):
    """将输入的标量或元组统一转换为长度为 3 的元组 (D, H, W)"""
    if isinstance(val, int):
        return (val, val, val)
    return val


class MaxPool3dBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 针对 3D Max Pooling 定制的典型测试集。
        # 输入格式为 (Batch, Channels, Depth, Height, Width)
        # 配置格式为: (shape, kernel_size, stride, padding)
        configs = [
            # 1. 典型 3D 降采样 (如 3D U-Net / V-Net 医疗影像)
            ((2, 64, 32, 128, 128), 2, 2, 0),
            ((4, 256, 16, 64, 64), 2, 2, 0),
            ((8, 512, 8, 32, 32), 2, 2, 0),
            # 2. 视频理解与动作识别 (保留时间/深度维度，仅在空间降采样)
            ((8, 64, 16, 112, 112), (1, 2, 2), (1, 2, 2), 0),
            ((8, 128, 16, 56, 56), (2, 2, 2), (2, 2, 2), 0),
            # 3. 边界测试与重叠池化 (Overlapping pooling)
            ((1, 32, 15, 55, 55), 3, 2, 1),
            ((16, 16, 8, 112, 112), (3, 3, 3), 1, 1),
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

        kd, kh, kw = _to_tuple3(kernel_size)
        sd, sh, sw = _to_tuple3(stride)
        pd, ph, pw = _to_tuple3(padding)

        # 3D Pooling 的输出计算公式
        D_in, H_in, W_in = inp.shape[2], inp.shape[3], inp.shape[4]
        D_out = (D_in + 2 * pd - kd) // sd + 1
        H_out = (H_in + 2 * ph - kh) // sh + 1
        W_out = (W_in + 2 * pw - kw) // sw + 1

        # 输出的 numel = Batch * Channels * D_out * H_out * W_out
        out_numel = inp.shape[0] * inp.shape[1] * D_out * H_out * W_out

        io_amount = (
            shape_utils.size_in_bytes(inp) + out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.max_pool3d
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_max_pool3d(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = MaxPool3dBenchmark(
        op_name="max_pool3d",
        torch_op=torch_max_pool3d,
        gems_op=gems_max_pool3d_wrapper,
        dtypes=[dtype],
    )
    bench.run()
