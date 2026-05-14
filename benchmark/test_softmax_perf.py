from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


# Softmax 绝大多数场景都是在最后一个维度 (dim=-1) 进行操作
def torch_softmax(x, y=None):
    return F.softmax(x, dim=-1)


def gems_softmax_wrapper(x, y=None):
    return flag_dnn.ops.softmax(x, dim=-1)


class SoftmaxBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 针对 Softmax 定制的真实业务典型 shape
        shapes = [
            # 1. 基础小尺寸 & 1D (测试启动与寻址开销)
            (1024,),
            (65536,),
            # 2. 图像分类/小模型 Logits (Batch, Num_Classes)
            (256, 1000),  # ImageNet 典型分类输出
            (1024, 100),  # CIFAR/通用小规模分类
            # 3. LLM 大语言模型 词表分类 (Batch, Vocab_size)
            (32, 32000),  # LLaMA-1/2 典型词表大小
            (16, 128256),  # LLaMA-3 典型大词表大小
            # 4. Transformer Attention 典型 shape
            # (Batch, Num_heads, Seq_len, Seq_len)
            (16, 12, 1024, 1024),  # BERT-base 注意力分数矩阵
            (8, 32, 2048, 2048),  # LLM 2K Context 注意力
            (4, 32, 4096, 4096),  # LLM 4K Context 注意力
            (1, 32, 8192, 8192),  # LLM 8K 长文本 注意力 (重点考验极限归约能力)
            # 5. 极端长序列/极限压力测试
            (
                2,
                8,
                32768,
                32768,
            ),  # 32K 超长上下文的单次 Attention 矩阵理论大小
        ]
        self.shapes = shapes
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        # 设定单张 Tensor 的安全显存上限 (8 GB)
        # 防止 32K attention 在高精度(FP64/FP32)时直接撑爆显存
        MAX_TENSOR_BYTES = 8 * 1024**3

        for shape in self.shapes:
            # 显存拦截：提前计算当前 shape 和 dtype 下的体积
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            # 如果单张 Tensor 超过限制，跳过当前 shape，防止 OOM
            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp1.numel() == 0:
                continue

            # Softmax 只需要输入 x，不需要 weight
            yield inp1, None

    def get_gbps(self, args, latency):
        inp1 = args[0]

        # 优秀的 Online Softmax Kernel 会融合操作，实现一次读取、一次写入
        # GBPS = (读入 x + 写出 y) / 延迟
        io_amount = shape_utils.size_in_bytes(
            inp1
        ) + shape_utils.size_in_bytes(inp1)
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.softmax
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_softmax(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = SoftmaxBenchmark(
        op_name="softmax",
        torch_op=torch_softmax,
        gems_op=gems_softmax_wrapper,
        dtypes=[dtype],
    )
    bench.run()
