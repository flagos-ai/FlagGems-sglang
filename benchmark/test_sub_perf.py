from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_sub(x, y):
    return torch.sub(x, y)


def gems_sub_wrapper(x, y):
    return flag_dnn.ops.sub(x, y)


class SubBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        configs = [
            # 1. 相同 Shape (纯 Element-wise)
            ((1024, 1024), (1024, 1024)),
            ((32, 256, 1024), (32, 256, 1024)),
            ((32, 64, 112, 112), (32, 64, 112, 112)),
            ((8, 2048, 64, 64), (8, 2048, 64, 64)),
            # 2. 典型的 1D 广播
            ((1024, 256), (256,)),
            ((32, 256, 1024), (256, 1)),
            # 3. CV 中的空间与通道广播 (NCHW 格式)
            ((32, 256, 56, 56), (256, 1, 1)),
            ((32, 256, 56, 56), (1, 256, 1, 1)),
            # 4. 复杂的双向广播
            ((32, 1, 56, 56), (1, 256, 56, 56)),
            ((8, 16, 1, 128), (1, 16, 128, 1)),
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for shape_x, shape_y in self.shapes:
            out_shape = torch.broadcast_shapes(shape_x, shape_y)
            out_numel = np.prod(out_shape)

            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            total_bytes = (
                np.prod(shape_x) + np.prod(shape_y) + out_numel
            ) * element_size

            if total_bytes > MAX_TENSOR_BYTES:
                continue

            x = torch.randn(shape_x, dtype=cur_dtype, device=self.device)
            y = torch.randn(shape_y, dtype=cur_dtype, device=self.device)

            if x.numel() == 0 or y.numel() == 0:
                continue

            yield x, y

    def get_gbps(self, args, latency):
        x = args[0]
        y = args[1]

        out_shape = torch.broadcast_shapes(x.shape, y.shape)
        out_bytes = np.prod(out_shape) * x.element_size()

        io_amount = (
            shape_utils.size_in_bytes(x)
            + shape_utils.size_in_bytes(y)
            + out_bytes
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.sub
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_sub(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = SubBenchmark(
        op_name="sub",
        torch_op=torch_sub,
        gems_op=gems_sub_wrapper,
        dtypes=[dtype],
    )
    bench.run()
