import pytest
import torch
import torch._C._nn as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# (shape, output_size)
PARAMS = [
    ((2, 3, 32, 32), (1, 1)),  # 全局最大池化
    ((1, 16, 28, 28), 14),  # 降维到 14x14
    ((4, 8, 15, 15), (7, 5)),  # 非对称目标尺寸
    ((2, 4, 32, 32), (16, 16)),  # 保持 H 尺寸不变
    ((16, 14, 14), (2, 2)),  # 3D 张量输入
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


@pytest.mark.adaptive_max_pool2d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, output_size", PARAMS)
def test_accuracy_adaptive_max_pool2d(dtype, shape, output_size):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # MaxPool 本身不做乘加，直接比较，所以可以要求绝对的零误差
    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.adaptive_max_pool2d(ref_x, output_size)
    with flag_dnn.use_dnn():
        out = F.adaptive_max_pool2d(x, output_size)

    ref_y, ref_idx = ref_out
    y, idx = out
    utils.gems_assert_close(y, ref_y, dtype, atol=0)
    utils.gems_assert_equal(idx, ref_idx)


@pytest.mark.adaptive_max_pool2d
@pytest.mark.parametrize(
    "dtype",
    [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float16],
)
def test_accuracy_adaptive_max_pool2d_empty_tensor(dtype):
    shape = (0, 3, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.adaptive_max_pool2d(ref_x, (2, 2))
    with flag_dnn.use_dnn():
        out = F.adaptive_max_pool2d(x, (2, 2))

    ref_y, ref_idx = ref_out
    y, idx = out
    assert y.shape == ref_y.shape
    assert y.numel() == 0
    assert idx.shape == ref_idx.shape
    assert idx.numel() == 0
    utils.gems_assert_close(y, ref_y, dtype, atol=0)
    utils.gems_assert_equal(idx, ref_idx)
