from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_cummax(x, dim):
    return torch.cummax(x, dim)


def gems_cummax_wrapper(x, dim):
    return flag_dnn.ops.cummax(x, dim)


_CUMMAX_CONFIGS = [
    ((1024,), 0),  # 1D 超长向量 Scan
    ((32, 256, 1024), 2),  # Inner Dim (Row) Scan, 易合并
    ((32, 256, 1024), 0),  # Outer Dim (Column) Scan, 最难合并
    ((32, 256, 1024), 1),  # Middle Dim Scan
    ((1024, 1024), 1),  # 方阵 Inner
    ((1024, 1024), 0),  # 方阵 Outer
    ((32, 256, 56, 56), 3),  # CV 典型 Inner
    ((32, 256, 56, 56), 1),  # CV 典型 Channel Scan
]


class CummaxBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = _CUMMAX_CONFIGS
        return None

    def _gen_inputs(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3  # 8 GiB guard

        is_floating = cur_dtype.is_floating_point
        for shape, dim in self.shapes:
            numel = int(np.prod(shape))
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue
            if numel == 0:
                continue

            if is_floating:
                inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            else:
                # Use a moderate range so there are plenty of ties (stresses
                # tie-break) but values stay in-range for small-width ints.
                iinfo = torch.iinfo(cur_dtype)
                low = max(iinfo.min, -128)
                high = min(iinfo.max, 127) + 1
                inp = torch.randint(
                    low, high, shape, dtype=cur_dtype, device=self.device
                )

            yield inp, dim

    def get_input_iter(self, cur_dtype) -> Generator:
        yield from self._gen_inputs(cur_dtype)

    def get_gbps(self, args, latency):
        inp = args[0]
        numel = inp.numel()

        io_values = shape_utils.size_in_bytes(inp) * 2
        io_indices = numel * 8
        io_amount = io_values + io_indices

        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.cummax
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_cummax(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = CummaxBenchmark(
        op_name="cummax",
        torch_op=torch_cummax,
        gems_op=gems_cummax_wrapper,
        dtypes=[dtype],
    )
    bench.run()
