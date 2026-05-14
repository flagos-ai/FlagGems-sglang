from typing import Generator

import numpy as np
import pytest
import torch

import flag_dnn

from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_sqrt(x):
    return torch.sqrt(x)


def gems_sqrt_wrapper(x):
    return flag_dnn.ops.sqrt(x)


class SqrtBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        configs = [
            # 1. 极小 Shape (测 Python 调度开销和 Kernel Launch 延迟)
            # 在这种尺寸下，GPU 计算时间几乎为 0，比拼的全是外层框架的皮薄不薄
            (1,),
            (64,),
            (128,),
            # 2. 非对齐 Shape
            (127,),  # 接近 128
            (1023, 1025),  # 不对齐
            (7, 31, 109),  # 质数
            # 3. 经典 NLP / LLM 尺寸 (Batch, SeqLen, Hidden)
            (1, 2048, 4096),  # Llama 7B 推理常见的上下文尺寸
            (4, 4096, 4096),  # 加大 Batch 和 Sequence Length
            (8, 128, 12288),  # 类似 GPT-3 的短序列、大隐藏层
            # 4. 经典 CV 尺寸 (NCHW)
            (1, 3, 224, 224),  # 经典 ImageNet 输入 (元素极少)
            (32, 256, 56, 56),  # ResNet 中间层特征图
            (16, 1024, 14, 14),  # 深层通道密集型特征图
            # 5. 常规大尺寸
            (1024 * 256,),  # 标准一维大向量
            (1024, 1024),  # 1M 元素
            (4096, 4096),  # 16M 元素
            # 6. 极限吞吐 / HBM 带宽压力测试
            (1024 * 1024 * 64,),  # 纯 1D 线性爆炸，6400 万元素
            (2, 8192, 8192),  # 约 1.34 亿元素
        ]

        # 将配置包装成 tuple 以兼容 Benchmark 基类 (x_shape, )
        self.shapes = [(shape,) for shape in configs]
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for (shape_x,) in self.shapes:
            element_size = torch.tensor([], dtype=cur_dtype).element_size()

            # 估算总内存 (x + out)，防止 OOM
            total_bytes = np.prod(shape_x) * element_size * 2

            if total_bytes > MAX_TENSOR_BYTES:
                continue

            # 防 NaN
            # sqrt 的输入必须 >= 0，这里使用 rand(0~1) 并乘上一个标量，范围控制在 [0, 10.0]
            x = torch.rand(shape_x, dtype=cur_dtype, device=self.device) * 10.0

            if x.numel() == 0:
                continue

            yield (x,)

    def get_gbps(self, args, latency):
        x = args[0]

        # 一元算子的 GBPS 计算：读 x + 写 out (两者 size 完全一样)
        io_amount = shape_utils.size_in_bytes(x) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.sqrt
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_sqrt(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = SqrtBenchmark(
        op_name="sqrt",
        torch_op=torch_sqrt,
        gems_op=gems_sqrt_wrapper,
        dtypes=[dtype],
    )
    bench.run()
