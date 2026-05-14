from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_pow(x, y):
    return torch.pow(x, y)


def gems_pow_wrapper(x, y):
    return flag_dnn.ops.pow(x, y)


class PowBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # Pow 的 Shape 策略：除了常规广播，增加标量/单元素广播（非常高频的场景）
        configs = [
            # 1. 相同 Shape (极限压榨计算密集型算力)
            ((1024, 1024), (1024, 1024)),
            ((32, 256, 1024), (32, 256, 1024)),
            # 2. 标量/单元素广播 (例如 x ** 2.0 或 x ** -0.5)
            ((1024, 1024), (1,)),
            ((32, 256, 1024), (1,)),
            # 3. 典型的 1D 广播 (沿着特定维度进行不同的幂运算)
            ((1024, 256), (256,)),
            ((32, 256, 1024), (256, 1)),
            # 4. CV 中的空间与通道广播
            ((32, 256, 56, 56), (256, 1, 1)),
            ((32, 256, 56, 56), (1, 256, 1, 1)),
            # 5. 复杂的双向广播 (极致的内存偏移寻址压力测试)
            ((32, 1, 56, 56), (1, 256, 56, 56)),
            ((8, 16, 1, 128), (8, 16, 1, 128)),
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

            # 防 NaN 与防溢出
            # 1. 底数 x 必须为正数，且避免接近 0：使用 rand(0~1) + 0.1，范围在 [0.1, 1.1]
            x = torch.rand(shape_x, dtype=cur_dtype, device=self.device) + 0.1

            # 2. 指数 y 必须受限，防止 fp16 下指数爆炸：将指数控制在 [-2.0, 3.0] 之间
            y = torch.empty(
                shape_y, dtype=cur_dtype, device=self.device
            ).uniform_(-2.0, 3.0)

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


@pytest.mark.pow
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_pow(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = PowBenchmark(
        op_name="pow",
        torch_op=torch_pow,
        gems_op=gems_pow_wrapper,
        dtypes=[dtype],
    )
    bench.run()
