from typing import Generator

import pytest
import torch

import flag_dnn
from benchmark.performance_utils import Benchmark


def torch_dot(x, y):
    return torch.dot(x, y)


def gems_dot_wrapper(x, y):
    return flag_dnn.ops.dot(x, y)


class DotBenchmark(Benchmark):
    # dot 主要读两个输入向量，输出只有一个标量，可忽略不计
    IO_FACTOR = 2
    FLOP_FACTOR = 2
    MAX_PEAK_BYTES = 6 * 1024**3

    def set_more_metrics(self):
        return ["gbps", "gflops"]

    def set_more_shapes(self):
        # dot 是 1D 向量归约，这里用常见向量长度来测
        # 兼顾：
        # 1. 小规模
        # 2. 常见 DL hidden size
        # 3. 2 的幂附近
        # 4. 大规模吞吐
        self.shapes = [
            (256,),
            (512,),
            (768,),
            (1024,),
            (2048,),
            (3072,),
            (4096,),
            (6144,),
            (8192,),
            (12288,),
            (24576,),
            (32768,),
            (49152,),
            (65536,),
            (131072,),
            (262144,),
            (524288,),
            (1048576,),
            (2097152,),
            (4194304,),
            (8388608,),
        ]
        return None

    @staticmethod
    def _tensor_nbytes(shape, dtype):
        numel = shape[0]
        return numel * torch.empty((), dtype=dtype).element_size()

    def _estimate_peak_bytes(self, shape, dtype):
        # 两个输入向量
        input_bytes = self._tensor_nbytes(shape, dtype)
        return input_bytes * 2

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            if (
                self._estimate_peak_bytes(shape, cur_dtype)
                > self.MAX_PEAK_BYTES
            ):
                continue

            numel = shape[0]
            if numel == 0:
                continue

            # dot 是乘加归约，范围不宜过大，避免大长度下数值过于发散
            x = torch.empty(
                shape, dtype=cur_dtype, device=self.device
            ).uniform_(-1.0, 1.0)
            y = torch.empty(
                shape, dtype=cur_dtype, device=self.device
            ).uniform_(-1.0, 1.0)

            yield (x, y)

    def get_gbps(self, args, latency):
        x, y = args
        io_amount = x.numel() * x.element_size() + y.numel() * y.element_size()
        return io_amount / (latency * 1e-3) / 1e9

    def get_gflops(self, args, latency):
        x, _ = args
        flops = x.numel() * self.FLOP_FACTOR
        return flops / (latency * 1e-3) / 1e9


@pytest.mark.dot
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_dot(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = DotBenchmark(
        op_name="dot",
        torch_op=torch_dot,
        gems_op=gems_dot_wrapper,
        dtypes=[dtype],
    )
    bench.run()
