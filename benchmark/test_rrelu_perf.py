from math import prod
from typing import Generator

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark, ELEMENTWISE_PERF_SHAPES


def torch_rrelu(x, lower, upper, training):
    return F.rrelu(x, lower=lower, upper=upper, training=training)


def gems_rrelu_wrapper(x, lower, upper, training):
    return flag_dnn.ops.rrelu(x, lower, upper, training)


class RReLUBenchmark(Benchmark):
    IO_FACTOR = 2
    MAX_PEAK_BYTES = 6 * 1024**3

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = list(ELEMENTWISE_PERF_SHAPES)
        return None

    @staticmethod
    def _tensor_nbytes(shape, dtype):
        return prod(shape) * torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, dtype):
        input_bytes = self._tensor_nbytes(shape, dtype)
        return input_bytes * 2

    def get_input_iter(self, cur_dtype) -> Generator:
        lower = 1.0 / 8
        upper = 1.0 / 3
        training = False

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
            ).uniform_(-1.0, 1.0)
            yield x, lower, upper, training

    def get_gbps(self, args, latency):
        x = args[0]
        io_amount = x.numel() * x.element_size() * self.IO_FACTOR
        return io_amount / (latency * 1e-3) / 1e9


@pytest.mark.rrelu
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_rrelu(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = RReLUBenchmark(
        op_name="rrelu",
        torch_op=torch_rrelu,
        gems_op=gems_rrelu_wrapper,
        dtypes=[dtype],
    )
    bench.run()
