from math import prod
from typing import Generator

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark


def torch_glu(x):
    return F.glu(x, dim=-1)


def gems_glu_wrapper(x):
    return flag_dnn.ops.glu(x, dim=-1)


class GluBenchmark(Benchmark):
    # 读取整个输入 + 写出一半大小的输出
    IO_FACTOR = 1.5
    MAX_PEAK_BYTES = 6 * 1024**3

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = [
            (2,),
            (16,),
            (1024,),
            (65536,),
            (17, 32),
            (1023, 1024),
            (7, 31, 110),
            (32, 128, 768),
            (1, 2048, 4096),
            (2, 3, 32, 64),
            (1, 3, 224, 224),
            (8, 64, 56, 56),
            (16, 128, 28, 28),
            (32, 256, 14, 14),
            (1, 8, 16, 32, 32),
        ]
        return None

    @staticmethod
    def _tensor_nbytes(shape, dtype):
        return prod(shape) * torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, dtype):
        input_bytes = self._tensor_nbytes(shape, dtype)
        output_bytes = input_bytes // 2
        return input_bytes + output_bytes

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            if shape[-1] % 2 != 0:
                continue

            if (
                self._estimate_peak_bytes(shape, cur_dtype)
                > self.MAX_PEAK_BYTES
            ):
                continue

            numel = prod(shape)
            if numel == 0:
                continue

            x = torch.empty(
                shape, dtype=cur_dtype, device=self.device
            ).uniform_(-5.0, 5.0)
            yield (x,)

    def get_gbps(self, args, latency):
        x = args[0]
        io_amount = x.numel() * x.element_size() * self.IO_FACTOR
        return io_amount / (latency * 1e-3) / 1e9


@pytest.mark.glu
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_glu(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = GluBenchmark(
        op_name="glu",
        torch_op=torch_glu,
        gems_op=gems_glu_wrapper,
        dtypes=[dtype],
    )
    bench.run()
