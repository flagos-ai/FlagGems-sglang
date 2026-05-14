import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# adaptive_max_pool1d 参数格式：(shape, output_size)
PARAMS = [
    ((2, 3, 32), 16),  # 标准降采样
    ((1, 8, 14), 14),  # output == input (原样输出)
    ((2, 4, 15), 7),  # 不规则的奇数下采样
    ((1, 2, 8), 12),  # 上采样 (输出尺寸大于输入)
    ((4, 5, 20), 1),  # 全局最大池化 (Global Max Pooling)
    ((16, 14), 5),  # 2D 张量输入 (无 Batch 维度)
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


@pytest.mark.adaptive_max_pool1d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, output_size", PARAMS)
def test_accuracy_adaptive_max_pool1d(dtype, shape, output_size):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 生成测试数据
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.adaptive_max_pool1d(ref_x, output_size)

    with flag_dnn.use_dnn():
        out = torch.adaptive_max_pool1d(x, output_size)

    out_vals, out_indices = out
    ref_vals, ref_indices = ref_out

    # 验证数值正确性
    utils.gems_assert_close(out_vals, ref_vals, dtype, atol=0)
    # 验证索引正确性 (必须完全一致)
    utils.gems_assert_equal(out_indices, ref_indices)


@pytest.mark.adaptive_max_pool1d
@pytest.mark.parametrize(
    "dtype",
    [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float16],
)
def test_accuracy_adaptive_max_pool1d_empty_tensor(dtype):
    # W 维度尺寸导致输出 M=0 的情况
    shape = (0, 3, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.adaptive_max_pool1d(ref_x, 2)
    with flag_dnn.use_dnn():
        out = torch.adaptive_max_pool1d(x, 2)

    out_vals, out_indices = out
    ref_vals, ref_indices = ref_out
    assert out_vals.shape == ref_vals.shape
    assert out_indices.shape == ref_indices.shape
    assert out_vals.numel() == 0
    assert out_indices.numel() == 0
    utils.gems_assert_close(out_vals, ref_vals, dtype, atol=0)
    utils.gems_assert_equal(out_indices, ref_indices)
