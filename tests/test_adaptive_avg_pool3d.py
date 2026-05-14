import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# adaptive_avg_pool3d 参数格式：(shape, output_size)
PARAMS = [
    ((2, 3, 8, 16, 16), (4, 8, 8)),  # 标准 3D 降采样
    ((1, 8, 5, 14, 14), 7),  # 输出尺寸为单 int
    ((2, 4, 7, 15, 15), (3, 5, 7)),  # 三个维度不同尺寸
    ((1, 2, 4, 8, 8), (5, 10, 10)),  # 上采样 (输出尺寸大于输入)
    ((4, 5, 10, 20, 20), 1),  # 3D 全局平均池化 (Global Average Pooling)
    ((3, 8, 14, 14), (4, 7, 7)),  # 4D 张量输入 (无 Batch 维度 N)
    ((1, 2, 8, 8, 8), (8, 8, 8)),  # output == input (原样输出)
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _as_triple(output_size):
    return (
        (output_size, output_size, output_size)
        if isinstance(output_size, int)
        else output_size
    )


def _adaptive_reduce_dim(shape, output_size):
    out_d, out_h, out_w = _as_triple(output_size)
    in_d, in_h, in_w = shape[-3], shape[-2], shape[-1]
    return max(
        ((in_d + out_d - 1) // out_d)
        * ((in_h + out_h - 1) // out_h)
        * ((in_w + out_w - 1) // out_w),
        1,
    )


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, output_size", PARAMS)
def test_accuracy_adaptive_avg_pool3d(dtype, shape, output_size):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 生成测试数据
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.adaptive_avg_pool3d(ref_x, output_size)

    with flag_dnn.use_dnn():
        out = F.adaptive_avg_pool3d(x, output_size)

    utils.gems_assert_close(
        out,
        ref_out,
        dtype,
        reduce_dim=_adaptive_reduce_dim(shape, output_size),
    )


@pytest.mark.adaptive_avg_pool3d
@pytest.mark.parametrize(
    "dtype",
    [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float16],
)
def test_accuracy_adaptive_avg_pool3d_empty_tensor(dtype):
    # D, H, W 至少一个维度的尺寸导致输出 M=0 的情况
    shape = (0, 3, 4, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.adaptive_avg_pool3d(ref_x, 2)
    with flag_dnn.use_dnn():
        out = F.adaptive_avg_pool3d(x, 2)

    assert out.shape == ref_out.shape
    assert out.numel() == 0
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=512)
