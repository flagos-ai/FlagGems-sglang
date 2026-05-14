from typing import Generator

import pytest
import torch

import flag_dnn
from benchmark.performance_utils import Benchmark


def torch_mv(mat, vec):
    return torch.mv(mat, vec)


def gems_mv_wrapper(mat, vec):
    return flag_dnn.ops.mv(mat, vec)


class MVBenchmark(Benchmark):
    MAX_PEAK_BYTES = 6 * 1024**3

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        self.shapes = [
            # 2 的幂：基准场景
            (64, 64),
            (256, 256),
            (1024, 1024),
            (4096, 4096),
            # tall-skinny
            (4096, 256),
            (16384, 512),
            # short-fat
            (256, 4096),
            (512, 16384),
            # 非 2 的幂：小中规模
            (63, 65),
            (127, 255),
            (511, 1000),
            # 非 2 的幂：常见工程宽度附近
            (2048, 768),
            (4096, 1536),
            (8192, 3072),
            # 非 2 的幂：tall-skinny / short-fat
            (5000, 384),
            (384, 5000),
            # 更接近真实业务的不规则组合
            (3584, 3584),
            (1892, 3584),
        ]
        return None

    @staticmethod
    def _elem_size(dtype):
        return torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, dtype):
        m, n = shape
        elem_size = self._elem_size(dtype)

        mat_bytes = m * n * elem_size
        vec_bytes = n * elem_size
        out_bytes = m * elem_size

        return mat_bytes + vec_bytes + out_bytes

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            if (
                self._estimate_peak_bytes((m, n), cur_dtype)
                > self.MAX_PEAK_BYTES
            ):
                continue

            mat = torch.empty(
                (m, n), dtype=cur_dtype, device=self.device
            ).uniform_(-1.0, 1.0)
            vec = torch.empty(
                (n,), dtype=cur_dtype, device=self.device
            ).uniform_(-1.0, 1.0)

            yield mat, vec

    def get_gbps(self, args, latency):
        mat, vec = args

        io_amount = (
            mat.numel() * mat.element_size()
            + vec.numel() * vec.element_size()
            + mat.shape[0] * mat.element_size()
        )
        return io_amount / (latency * 1e-3) / 1e9


@pytest.mark.mv
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_mv(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = MVBenchmark(
        op_name="mv",
        torch_op=torch_mv,
        gems_op=gems_mv_wrapper,
        dtypes=[dtype],
    )
    bench.run()
