from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_mean(x, dim, keepdim):
    if dim is None:
        return torch.mean(x)
    return torch.mean(x, dim=dim, keepdim=keepdim)


def gems_mean_wrapper(x, dim, keepdim):
    if dim is None:
        return flag_dnn.ops.mean(x)
    return flag_dnn.ops.mean(x, dim=dim, keepdim=keepdim)


class MeanBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 格式为: (shape, dim, keepdim)
        configs = [
            ((1024 * 1024 * 16,), None, False),  # 1D 超长向量
            ((32, 256, 1024), None, False),  # 3D 全局归约
            ((1024, 1024 * 16), 1, False),
            ((32, 1024, 1024), 2, False),  # NLP Seq_len 维度归约
            ((8, 128, 4096), 2, False),  # NLP 大词表/长序列
            ((1024 * 16, 1024), 0, False),  # 跨步距 Outer 访存
            ((32, 1024, 1024), 0, False),  # Reduce Batch 维度
            ((32, 256, 56, 56), (2, 3), False),  # 全局平均池化 (GAP) 前置
            ((32, 256, 56, 56), 1, False),  # Reduce 通道维度
            ((1, 16, 2048, 2048), (2, 3), False),  # 高分辨空间维度
            ((64, 512, 512), 2, True),  # keepdim 测试
            ((128, 256, 256), 1, True),
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3
        for config in self.shapes:
            shape, dim, keepdim = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue

            yield inp, dim, keepdim

    def get_gbps(self, args, latency):
        inp = args[0]
        dim = args[1]

        if dim is None:
            out_numel = 1
        else:
            dims = [dim] if isinstance(dim, int) else dim
            out_numel = inp.numel()
            for d in dims:
                out_numel //= inp.shape[d]

        io_amount = (
            shape_utils.size_in_bytes(inp) + out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.mean
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_mean(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = MeanBenchmark(
        op_name="mean",
        torch_op=torch_mean,
        gems_op=gems_mean_wrapper,
        dtypes=[dtype],
    )
    bench.run()
