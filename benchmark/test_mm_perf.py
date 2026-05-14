from typing import Generator

import pytest
import torch

import flag_dnn
from benchmark.performance_utils import Benchmark


def torch_mm(a, b):
    return torch.mm(a, b)


def gems_mm_wrapper(a, b):
    return flag_dnn.ops.mm(a, b)


class MMBenchmark(Benchmark):
    FLOP_FACTOR = 2
    MAX_PEAK_BYTES = 8 * 1024**3

    def set_more_metrics(self):
        return ["gbps", "gflops"]

    def set_more_shapes(self):
        self.shapes = [
            # ------------------------------------------------------------
            # 1. Tiny / correctness-sensitive shapes
            # ------------------------------------------------------------
            (1, 1, 1),
            (1, 16, 16),
            (16, 1, 16),
            (16, 16, 1),
            (8, 8, 8),
            (16, 16, 16),
            (17, 17, 17),
            # ------------------------------------------------------------
            # 2. Small / medium square GEMM
            # ------------------------------------------------------------
            (32, 32, 32),
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
            # ------------------------------------------------------------
            # 3. Rectangular aligned GEMM
            # ------------------------------------------------------------
            (32, 64, 16),
            (64, 128, 32),
            (128, 256, 64),
            (256, 512, 128),
            (512, 1024, 256),
            (1024, 512, 256),
            # ------------------------------------------------------------
            # 4. Non-power-of-two / boundary shapes
            # ------------------------------------------------------------
            (31, 31, 31),
            (33, 65, 17),
            (63, 127, 31),
            (65, 129, 33),
            (127, 65, 33),
            (129, 257, 65),
            (255, 511, 127),
            (513, 1025, 255),
            # ------------------------------------------------------------
            # 5. Skinny M / skinny N / skinny K
            # ------------------------------------------------------------
            # Small M, large N/K
            (1, 1024, 1024),
            (4, 1024, 1024),
            (16, 1024, 1024),
            # Small N, large M/K
            (1024, 1, 1024),
            (1024, 4, 1024),
            (1024, 16, 1024),
            # Small K, large M/N
            (1024, 1024, 1),
            (1024, 1024, 4),
            (1024, 1024, 16),
            (1024, 1024, 32),
            # ------------------------------------------------------------
            # 6. Tall / wide rectangular GEMM
            # ------------------------------------------------------------
            (4096, 256, 512),
            (256, 4096, 512),
            (4096, 512, 256),
            (512, 4096, 256),
            (8192, 256, 512),
            (256, 8192, 512),
            # ------------------------------------------------------------
            # 7. Large K / reduction-heavy GEMM
            # ------------------------------------------------------------
            (512, 512, 1024),
            (512, 512, 2048),
            (1024, 1024, 2048),
            (2048, 2048, 1024),
            # ------------------------------------------------------------
            # 8. Transformer / LLM-style GEMM
            # ------------------------------------------------------------
            # Common hidden sizes: 768, 1024, 2048, 3072, 4096
            # Common FFN sizes: 3072, 4096, 8192, 11008, 14336
            (128, 768, 768),
            (512, 768, 768),
            (2048, 768, 768),
            (128, 3072, 768),
            (512, 3072, 768),
            (2048, 3072, 768),
            (128, 768, 3072),
            (512, 768, 3072),
            (2048, 768, 3072),
            (128, 4096, 4096),
            (512, 4096, 4096),
            (2048, 4096, 4096),
            (128, 11008, 4096),
            (512, 11008, 4096),
            (2048, 11008, 4096),
            (128, 4096, 11008),
            (512, 4096, 11008),
            (2048, 4096, 11008),
            # ------------------------------------------------------------
            # 9. Batch-token flattened GEMM patterns
            # ------------------------------------------------------------
            # e.g. M = batch * seq_len
            (256, 4096, 4096),
            (1024, 4096, 4096),
            (4096, 4096, 4096),
            (8192, 4096, 4096),
            # ------------------------------------------------------------
            # 10. Existing / regression shapes
            # ------------------------------------------------------------
            (1892, 3584, 768),
        ]
        return None

    @staticmethod
    def _elem_size(dtype):
        return torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, dtype):
        m, k, n = shape
        elem_size = self._elem_size(dtype)
        return (m * k + k * n + m * n) * elem_size

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, k, n in self.shapes:
            # Current fp64 path is a correctness/general fallback path.
            # Avoid treating large cuBLAS-dominated fp64 GEMM as active target.
            if cur_dtype == torch.float64 and max(m, k, n) > 128:
                continue

            if (
                self._estimate_peak_bytes((m, k, n), cur_dtype)
                > self.MAX_PEAK_BYTES
            ):
                continue

            a = torch.empty(
                (m, k), dtype=cur_dtype, device=self.device
            ).uniform_(-1.0, 1.0)
            b = torch.empty(
                (k, n), dtype=cur_dtype, device=self.device
            ).uniform_(-1.0, 1.0)
            yield a, b

    def get_gbps(self, args, latency):
        a, b = args
        io_amount = (
            a.numel() * a.element_size()
            + b.numel() * b.element_size()
            + a.shape[0] * b.shape[1] * a.element_size()
        )
        return io_amount / (latency * 1e-3) / 1e9

    def get_gflops(self, args, latency):
        a, b = args
        flops = a.shape[0] * a.shape[1] * b.shape[1] * self.FLOP_FACTOR
        return flops / (latency * 1e-3) / 1e9


@pytest.mark.mm
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_mm(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = MMBenchmark(
        op_name="mm",
        torch_op=torch_mm,
        gems_op=gems_mm_wrapper,
        dtypes=[dtype],
    )
    bench.run()
