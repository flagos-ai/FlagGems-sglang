from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_rms_norm(x, normalized_shape):
    return F.rms_norm(x, normalized_shape)


def gems_rms_norm_wrapper(x, normalized_shape):
    return flag_dnn.ops.rms_norm(x, normalized_shape)


class RmsNormBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        configs = [
            # 1. LLM 推理阶段 (Decode Phase / 自回归生成)
            # 特点：SeqLen 永远为 1，Batch Size 动态变化。
            ((256, 1, 2048), (2048,)),  # 小型模型高并发解码 (如 Gemma-2B)
            ((128, 1, 4096), (4096,)),  # LLaMA-3-8B / Mistral-7B 批量解码
            (
                (32, 1, 8192),
                (8192,),
            ),  # 70B/72B 级别大模型 (如 LLaMA-3-70B, Qwen2-72B) 的解码
            ((1, 1, 4096), (4096,)),  # 极低延迟的单用户流式响应
            # 2. Flattened Tokens
            # (连续批处理 Continuous Batching
            # / PageAttention 场景)
            # 特点：现代推理框架为了消除 Padding，
            # 通常会将 (Batch, SeqLen, Hidden)
            # 展平为 (Total_Tokens, Hidden)。
            ((16384, 4096), (4096,)),  # 约 16K 个 Token 并发 (一维展平)
            ((65536, 4096), (4096,)),  # 大规模吞吐下的展平 Token 处理
            ((100000, 8192), (8192,)),  # 大模型极致吞吐量测试 (10万 Token)
            # 3. 非 2 的幂次 / 特殊 Hidden Size
            # 特点：很多模型的 Hidden Size
            # 不是完美的 1024, 2048, 4096，
            # 这非常考验 Kernel 内的向量化读取
            # (Vectorized Load) 和边界处理。
            ((16, 2048, 1536), (1536,)),  # Qwen2-1.5B (Hidden = 1536)
            ((16, 2048, 3072), (3072,)),  # Phi-3-mini (Hidden = 3072)
            ((8, 4096, 5120), (5120,)),  # LLaMA-13B
            (
                (8, 4096, 3584),
                (3584,),
            ),  # LLaMA-3-8B 的 FFN 层有时会涉及非标准维度的投影归一化
            # 4. 端侧与轻量级模型 (Mobile / Edge)
            # 特点：为手机端或端侧 NPU 优化的极小模型，Hidden Size 很小。
            ((1, 512, 1024), (1024,)),  # 端侧 1B 级别以下模型预填充
            ((1, 1, 1024), (1024,)),  # 端侧逐字生成
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for config in self.shapes:
            shape, normalized_shape = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue

            yield inp, normalized_shape

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.rms_norm
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_rms_norm(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = RmsNormBenchmark(
        op_name="rms_norm",
        torch_op=torch_rms_norm,
        gems_op=gems_rms_norm_wrapper,
        dtypes=[dtype],
    )
    bench.run()
