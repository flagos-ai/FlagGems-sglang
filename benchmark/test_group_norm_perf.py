from typing import Generator

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from benchmark.performance_utils import Benchmark
from flag_dnn.utils import shape_utils


def torch_group_norm(x, num_groups):
    return F.group_norm(x, num_groups)


def gems_group_norm_wrapper(x, num_groups):
    return flag_dnn.ops.group_norm(x, num_groups)


class GroupNormBenchmark(Benchmark):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # 配置格式为: (shape, num_groups)
        # 必须保证 shape[1] (Channels) % num_groups == 0
        configs = [
            # 1. 典型 2D 卷积/视觉模型中的特征图 (Batch, Channels, H, W)
            ((32, 256, 56, 56), 32),  # 常见配置: 32个组，每组 8 个通道
            ((16, 512, 28, 28), 32),
            ((8, 1024, 14, 14), 32),
            # 2. Diffusion Models (如 SD) 中的典型 UNet 分辨率
            ((4, 320, 64, 64), 32),
            ((2, 1280, 16, 16), 32),
            # 3. 1D 时序/音频特征 (Batch, Channels, Length)
            ((32, 128, 1024), 8),
            ((16, 256, 4096), 16),
            # 4. 高分辨率目标检测与图像分割 (如 Mask R-CNN, FPN 结构)
            # 特点：Batch Size 极小 (通常为 1-2)，分辨率大，常常是非方形特征图。
            ((2, 256, 256, 256), 32),  # FPN 浅层高分辨特征图
            (
                (2, 256, 800, 1088),
                32,
            ),  # 真实场景中常见的不规则高分辨率输入 (H=800, W=1088)
            ((1, 256, 128, 128), 32),  # 极端小 Batch 推理
            # 5. 现代高清生成模型 (如 Stable Diffusion XL / 视频生成)
            # 特点：相比于基础 SD，SDXL 的起始 Latent 分辨率更大，通道数更深。
            (
                (2, 320, 128, 128),
                32,
            ),  # SDXL Base 分辨率 (生成 1024x1024 图像时的 Latent 尺寸)
            ((1, 2560, 16, 16), 32),  # UNet 最深层极宽的通道数验证
            # 6. 3D 视觉 / 视频理解 / 医疗影像 (Batch, Channels, Depth, Height, Width)
            (
                (2, 128, 16, 56, 56),
                32,
            ),  # 视频理解网络 (如 SlowFast/I3D)，16帧视频序列
            (
                (1, 32, 128, 128, 128),
                8,
            ),  # 3D 医疗影像分割 (如 V-Net 处理 CT/MRI 扫描)
            # 7. GroupNorm 的等价极端情况 (Edge Cases)
            (
                (8, 64, 112, 112),
                64,
            ),  # 当 num_groups == Channels 时，等价于 InstanceNorm
            (
                (8, 64, 112, 112),
                1,
            ),  # 当 num_groups == 1 时，等价于对 (C, H, W) 做的 LayerNorm
            ((128, 16, 8, 8), 4),  # 小通道、小尺寸特征图
        ]
        self.shapes = configs
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        MAX_TENSOR_BYTES = 8 * 1024**3

        for config in self.shapes:
            shape, num_groups = config
            numel = np.prod(shape)
            element_size = torch.tensor([], dtype=cur_dtype).element_size()
            tensor_bytes = numel * element_size

            if tensor_bytes > MAX_TENSOR_BYTES:
                continue

            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            if inp.numel() == 0:
                continue

            yield inp, num_groups

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) * 2
        return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.group_norm
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_perf_group_norm(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    bench = GroupNormBenchmark(
        op_name="group_norm",
        torch_op=torch_group_norm,
        gems_op=gems_group_norm_wrapper,
        dtypes=[dtype],
    )
    bench.run()
