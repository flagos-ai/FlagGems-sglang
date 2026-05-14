from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_add(x, y):
    return torch.add(x, y)


def gems_add_wrapper(x, y):
    return flag_dnn.ops.add(x, y)


class AddBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        configs = [
            # 1. 相同 Shape (纯 Element-wise，极限压榨带宽)
            ((1024, 1024), (1024, 1024)),
            ((32, 256, 1024), (32, 256, 1024)),  # 3D 张量
            ((32, 64, 112, 112), (32, 64, 112, 112)),  # ResNet 高维特征
            ((8, 2048, 64, 64), (8, 2048, 64, 64)),  # 超大张量压力测试
            # 2. 典型的 1D 广播 (如加 Bias, MLP 或 1D 卷积后)
            ((1024, 256), (256,)),  # MLP 加 Bias
            ((32, 256, 1024), (256, 1)),  # 沿着通道加 Bias
            # 3. CV 中的空间与通道广播 (NCHW 格式)
            ((32, 256, 56, 56), (256, 1, 1)),  # 常规 Conv2D 加 Bias
            ((32, 256, 56, 56), (1, 256, 1, 1)),  # 显式保留 N 维的 Bias
            # 4. 复杂的双向广播 (x 和 y 都需要 Broadcast 才能得到最终 shape)
            ((32, 1, 56, 56), (1, 256, 56, 56)),  # 输出为 (32, 256, 56, 56)
            ((8, 16, 1, 128), (1, 16, 128, 1)),  # 输出为 (8, 16, 128, 128)
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for shape_x, shape_y in self.shapes:
            # 计算广播后的输出 shape，以便估算内存占用
            out_shape = torch.broadcast_shapes(shape_x, shape_y)
            out_numel = np.prod(out_shape)

            element_size = torch.tensor([], dtype=cur_dtype).element_size()

            # 估算总内存 (x + y + out)，防止 OOM
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

        # 获取输出的 shape 以计算输出写入的数据量
        out_shape = torch.broadcast_shapes(x.shape, y.shape)
        out_bytes = np.prod(out_shape) * x.element_size()

        # GBPS 计算：读 x + 读 y + 写 out
        io_amount = (
            shape_utils.size_in_bytes(x)
            + shape_utils.size_in_bytes(y)
            + out_bytes
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.add
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_add(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = AddBenchmark(
        op_name="add",
        torch_op=torch_add,
        gems_op=gems_add_wrapper,
        dtypes=[dtype],
    )
    bench.run()
