from math import prod
from typing import Generator

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark


def torch_softmin(x, dim, out_dtype):
    return F.softmin(x, dim=dim, dtype=out_dtype)


def gems_softmin_wrapper(x, dim, out_dtype):
    return flag_dnn.ops.softmin(x, dim=dim, dtype=out_dtype)


class SoftminBenchmark(Benchmark):
    IO_FACTOR = 2
    MAX_PEAK_BYTES = 6 * 1024**3

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = [
            (1024,),
            (17, 31),
            (256, 1000),
            (32, 4096),
            (7, 31, 109),
            (4, 512, 1024),
            (8, 128, 4096),
            (2, 3, 32, 32),
            (8, 64, 56, 56),
            (16, 128, 28, 28),
            (2, 12, 512, 512),
        ]
        return None

    @staticmethod
    def _tensor_nbytes(shape, dtype):
        return prod(shape) * torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, dtype):
        input_bytes = self._tensor_nbytes(shape, dtype)
        return input_bytes * 2

    def _candidate_dims(self, shape):
        ndim = len(shape)
        dims = {0, -1}
        if ndim >= 2:
            dims.add(1)
        if prod(shape) >= 1024 * 1024:
            dims = {-1}
        return sorted(dims, key=lambda x: (x < 0, x))

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

            for dim in self._candidate_dims(shape):
                yield x, dim, None

    def get_gbps(self, args, latency):
        x = args[0]
        io_amount = x.numel() * x.element_size() * self.IO_FACTOR
        return io_amount / (latency * 1e-3) / 1e9


@pytest.mark.softmin
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_softmin(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = SoftminBenchmark(
        op_name="softmin",
        torch_op=torch_softmin,
        gems_op=gems_softmin_wrapper,
        dtypes=[dtype],
    )
    bench.run()
