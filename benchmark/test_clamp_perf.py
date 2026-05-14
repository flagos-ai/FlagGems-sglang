from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_clamp(x, min_val, max_val):
    return torch.clamp(x, min_val, max_val)


def gems_clamp_wrapper(x, min_val, max_val):
    return flag_dnn.ops.clamp(x, min_val, max_val)


class ClampBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 格式: (shape_x, shape_min, shape_max)
        # 如果 shape_min 或 shape_max 为 None，则代表该参数传入标量 (Scalar)
        configs = [
            # ==========================================
            # 场景 1: 纯标量截断 (Scalar)
            # ==========================================
            ((1024 * 256,), None, None),
            ((32, 256, 56, 56), None, None),
            ((2, 8192, 8192), None, None),
            # ==========================================
            # 场景 2: 同 Shape 张量截断 (Same Shape)
            # ==========================================
            ((127,), (127,), (127,)),
            ((1024 * 256,), (1024 * 256,), (1024 * 256,)),
            ((1, 2048, 4096), (1, 2048, 4096), (1, 2048, 4096)),
            # ==========================================
            # 场景 3: 广播张量截断 (Broadcasting)
            # ==========================================
            # 3.1 CV 场景: 按 Channel 截断 (NCHW)
            ((32, 256, 56, 56), (256, 1, 1), (256, 1, 1)),
            ((16, 1024, 14, 14), (1024, 1, 1), (1024, 1, 1)),
            # 3.2 NLP 场景: 按 Hidden Dim 截断
            ((8, 128, 12288), (12288,), (12288,)),
            # 3.3 极小 Tensor 广播 (退化为类似 Scalar 但走 Tensor 链路)
            ((1024, 1024), (1,), (1,)),
        ]

        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for shape_x, shape_min, shape_max in self.shapes:
            element_size = torch.tensor([], dtype=cur_dtype).element_size()

            # 动态计算所需的总显存
            bytes_x = np.prod(shape_x) * element_size
            bytes_min = (
                np.prod(shape_min) * element_size
                if shape_min is not None
                else 0
            )
            bytes_max = (
                np.prod(shape_max) * element_size
                if shape_max is not None
                else 0
            )

            # 总内存 = x读 + out写 + min读 + max读
            total_bytes = bytes_x * 2 + bytes_min + bytes_max

            if total_bytes > MAX_TENSOR_BYTES:
                continue

            # 生成主张量
            x = torch.randn(shape_x, dtype=cur_dtype, device=self.device) * 5.0

            if x.numel() == 0:
                continue

            # 生成 min 参数 (如果 shape 为 None，则给 float 标量)
            if shape_min is None:
                min_val = -2.0
            else:
                # 偏移到负数区间，降低与 max_val 冲突的概率
                min_val = (
                    torch.randn(shape_min, dtype=cur_dtype, device=self.device)
                    - 2.0
                )

            # 生成 max 参数 (如果 shape 为 None，则给 float 标量)
            if shape_max is None:
                max_val = 2.0
            else:
                # 偏移到正数区间
                max_val = (
                    torch.randn(shape_max, dtype=cur_dtype, device=self.device)
                    + 2.0
                )

            yield (x, min_val, max_val)

    def get_gbps(self, args, latency):
        x, min_val, max_val = args

        # 基础访存: 读取 x，写入 out
        io_amount = shape_utils.size_in_bytes(x) * 2

        # 如果参数是 Tensor，读取它们也会消耗总线带宽
        if isinstance(min_val, torch.Tensor):
            io_amount += shape_utils.size_in_bytes(min_val)
        if isinstance(max_val, torch.Tensor):
            io_amount += shape_utils.size_in_bytes(max_val)

        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.clamp
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_clamp(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = ClampBenchmark(
        op_name="clamp",
        torch_op=torch_clamp,
        gems_op=gems_clamp_wrapper,
        dtypes=[dtype],
    )
    bench.run()
