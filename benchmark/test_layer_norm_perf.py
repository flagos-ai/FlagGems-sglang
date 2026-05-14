from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_layer_norm(x, normalized_shape):
    return F.layer_norm(x, normalized_shape)


def gems_layer_norm_wrapper(x, normalized_shape):
    return flag_dnn.ops.layer_norm(x, normalized_shape)


class LayerNormBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: (shape, normalized_shape)
        # normalized_shape 必须匹配输入张量最后几个维度
        configs = [
            # 1. 典型 NLP/LLM 模型中的 Hidden Size 归一化
            ((32, 1024, 1024), (1024,)),  # (Batch, SeqLen, Hidden)
            ((16, 4096, 4096), (4096,)),  # 大模型长序列
            ((8, 8192, 4096), (4096,)),  # 极长上下文
            # 2. 视觉任务 (CV) 中的 LayerNorm (如 Vision Transformer)
            ((128, 197, 768), (768,)),  # ViT Base
            ((64, 1370, 1024), (1024,)),  # ViT Large
            # 3. 跨多维度归一化
            ((32, 256, 56, 56), (256, 56, 56)),  # InstanceNorm 的等价变体
            ((32, 256, 56, 56), (56, 56)),  # 对空间维度归一化
            # 4. LLM 推理阶段 (Decode Phase)
            # 特点：SeqLen = 1，Batch Size 较大
            ((128, 1, 4096), (4096,)),  # Llama-7B/8B, Qwen-7B
            (
                (256, 1, 2048),
                (2048,),
            ),  # 较小模型 (如 Gemma-2B) 的高吞吐批量解码
            ((32, 1, 8192), (8192,)),  # Llama-70B 级别大模型的解码
            # 5. 主流开源 LLM / 基础模型 (Prefill / Training 阶段)
            (
                (4, 8192, 2048),
                (2048,),
            ),  # 小规模模型长文本预填充 (如 Qwen 1.5B, 8K 上下文)
            (
                (2, 32768, 4096),
                (4096,),
            ),  # 长上下文场景 (如 Llama-3-8B 处理 32K Token)
            (
                (1, 128000, 4096),
                (4096,),
            ),  # 极长上下文场景 (128K Token，此时 Batch 往往受限于显存只能为 1)
            ((8, 4096, 8192), (8192,)),  # 百亿级参数模型 (70B+) 标准预训练尺寸
            # 6. 现代视觉与生成式模型 (Swin, ConvNeXt, DiT)
            (
                (16, 256, 1152),
                (1152,),
            ),  # DiT-XL (Diffusion Transformer)
            (
                (2048, 49, 96),
                (96,),
            ),  # Swin Transformer
            (
                (32, 56, 56, 96),
                (96,),
            ),  # ConvNeXt-Tiny: 典型的 (B, H, W, C) 格式，仅在通道维度 C 上做 LayerNorm
            # 7. 语音识别与音频处理 (Speech / Audio)
            # 特点：序列长度代表时间帧，通常比普通文本 NLP 更长，但 Hidden 较小。
            ((16, 1500, 512), (512,)),  # Whisper Base / Wav2Vec 2.0
            ((8, 3000, 1280), (1280,)),  # Whisper Large
            # 8. 边缘计算 / 端侧小模型 (Mobile/Edge)
            ((1, 512, 256), (256,)),  # 移动端轻量级 NLP 模型 (如 MobileBERT)
            ((1, 1, 256), (256,)),  # 端侧流式推理，极小 Shape
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
        # LayerNorm 读取一次输入 x，写出一次输出 y (暂时忽略极小的 weight/bias)
        io_amount = shape_utils.size_in_bytes(inp) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.layer_norm
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_layer_norm(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = LayerNormBenchmark(
        op_name="layer_norm",
        torch_op=torch_layer_norm,
        gems_op=gems_layer_norm_wrapper,
        dtypes=[dtype],
    )
    bench.run()
