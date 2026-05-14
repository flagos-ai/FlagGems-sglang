import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# adaptive_avg_pool1d 参数格式：(shape, output_size)
PARAMS = [
    ((2, 3, 32), 16),  # 标准降采样
    ((1, 8, 14), 14),  # output == input (原样输出)
    ((2, 4, 15), 7),  # 不规则的奇数下采样
    ((1, 2, 8), 12),  # 上采样 (输出尺寸大于输入，PyTorch 是支持的)
    ((4, 5, 20), 1),  # 全局平均池化 (Global Average Pooling)
    ((16, 14), 5),  # 2D 张量输入 (无 Batch(N) 维度)
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _adaptive_reduce_dim(shape, output_size):
    input_size = shape[-1]
    return max((input_size + output_size - 1) // output_size, 1)


@pytest.mark.adaptive_avg_pool1d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, output_size", PARAMS)
def test_accuracy_adaptive_avg_pool1d(dtype, shape, output_size):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 生成测试数据
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.adaptive_avg_pool1d(ref_x, output_size)

    with flag_dnn.use_dnn():
        out = F.adaptive_avg_pool1d(x, output_size)

    utils.gems_assert_close(
        out,
        ref_out,
        dtype,
        reduce_dim=_adaptive_reduce_dim(shape, output_size),
    )


@pytest.mark.adaptive_avg_pool1d
@pytest.mark.parametrize(
    "dtype",
    [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float16],
)
def test_accuracy_adaptive_avg_pool1d_empty_tensor(dtype):
    shape = (0, 3, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.adaptive_avg_pool1d(ref_x, 2)
    with flag_dnn.use_dnn():
        out = F.adaptive_avg_pool1d(x, 2)

    assert out.shape == ref_out.shape
    assert out.numel() == 0
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=16)
