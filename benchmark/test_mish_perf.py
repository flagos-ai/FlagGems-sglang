from math import prod
from typing import Generator

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark, ELEMENTWISE_PERF_SHAPES


def torch_mish(x):
    return F.mish(x)


def gems_mish_wrapper(x):
    return flag_dnn.ops.mish(x)


class MishBenchmark(Benchmark):
    IO_FACTOR = 2
    MAX_PEAK_BYTES = 6 * 1024**3

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = [
            shape
            for shape in ELEMENTWISE_PERF_SHAPES
            if prod(shape) != 16 and prod(shape) < 16 * 1024 * 1024
        ]
        return None

    @staticmethod
    def _tensor_nbytes(shape, dtype):
        return prod(shape) * torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, dtype):
        input_bytes = self._tensor_nbytes(shape, dtype)
        return input_bytes * 2

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
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


@pytest.mark.mish
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_mish(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = MishBenchmark(
        op_name="mish",
        torch_op=torch_mish,
        gems_op=gems_mish_wrapper,
        dtypes=[dtype],
    )
    bench.run()
