from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_sum(x, dim, keepdim):
    if dim is None:
        return torch.sum(x)
    return torch.sum(x, dim=dim, keepdim=keepdim)


def gems_sum_wrapper(x, dim, keepdim):
    if dim is None:
        return flag_dnn.ops.sum(x)
    return flag_dnn.ops.sum(x, dim=dim, keepdim=keepdim)


class SumBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 针对 Sum 定制的典型测试集。格式为: (shape, dim, keepdim)
        # 不同的 dim 对 GPU 访存合并 (Memory Coalescing) 考验极大
        configs = [
            # 1. 全局求和 (Global Reduction) - 极度考验原子操作 (Atomic Add) 或全局通信
            ((1024 * 1024 * 16,), None, False),  # 1D 超长向量
            ((32, 256, 1024), None, False),  # 3D 全局归约
            # 2. 沿着最后一个维度求和 (Row/Inner Reduction) - 内存连续，最容易优化
            ((1024, 1024), 1, False),  # 典型方阵
            ((32, 1024, 1024), 2, False),  # NLP 典型的 Seq_len 维度归约
            ((8, 128, 4096), 2, False),  # NLP 大词表/长序列
            # 3. 沿着最前面的维度求和 (Column/Outer Reduction) - 内存不连续，极度考验访存
            ((1024, 1024), 0, False),  # 考验跨步距 (Strided) 访存
            ((32, 1024, 1024), 0, False),  # Reduce Batch 维度
            # 4. 视觉任务 (CV) 中的典型归约
            (
                (32, 256, 56, 56),
                (2, 3),
                False,
            ),  # 类似全局平均池化 (GAP) 的前置求和
            (
                (128, 256, 56, 56),
                1,
                False,
            ),  # Reduce 通道维度 (Channel Reduction)
            ((1, 16, 2048, 2048), (2, 3), False),  # 高分辨率空间维度归约
            # 5. 保留维度 (keepdim=True) 测试
            ((64, 512, 512), 2, True),
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

        # 估算输出大小
        if dim is None:
            out_numel = 1
        else:
            # 支持传入 tuple 的 dim (例如 (2, 3))
            dims = [dim] if isinstance(dim, int) else dim
            out_numel = inp.numel()
            for d in dims:
                out_numel //= inp.shape[d]

        # GBPS 计算：读入输入 x + 写出输出 y
        io_amount = shape_utils.size_in_bytes(inp) + (
            out_numel * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.sum
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_sum(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = SumBenchmark(
        op_name="sum",
        torch_op=torch_sum,
        gems_op=gems_sum_wrapper,
        dtypes=[dtype],
    )
    bench.run()
