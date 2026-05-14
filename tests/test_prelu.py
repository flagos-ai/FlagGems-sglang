import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# 专门为 PReLU 扩展了多维 Shape，以测试通道维度 (dim=1)
SHAPES = [(32,), (1024,), (2, 16), (4, 8, 32), (2, 4, 16, 16)]
MODES = ["single", "channel"]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    MIXED_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    MIXED_DTYPES = [torch.float32, torch.float64]


@pytest.mark.prelu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_prelu(dtype, shape, mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 如果是逐通道模式，但维度不够（只有 1 维），则跳过当前测试组合
    if mode == "channel" and len(shape) < 2:
        pytest.skip("Channel mode requires at least 2 dimensions")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 根据模式初始化 weight 参数
    if mode == "single":
        num_parameters = 1
    else:
        num_parameters = shape[1]  # PyTorch 约定 dim=1 是通道维度

    weight = torch.full(
        (num_parameters,), 0.25, dtype=dtype, device=flag_dnn.device
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_y = F.prelu(ref_x, ref_weight)
    with flag_dnn.use_dnn():
        y = F.prelu(x, weight)

    utils.gems_assert_close(y, ref_y, dtype)


@pytest.mark.prelu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_prelu_empty_tensor(dtype, mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量，为了能测试 channel 模式，给一个含 0 的多维 shape
    shape = (0, 4, 16) if mode == "channel" else (0,)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    num_parameters = shape[1] if mode == "channel" else 1
    weight = torch.full(
        (num_parameters,), 0.25, dtype=dtype, device=flag_dnn.device
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_y = F.prelu(ref_x, ref_weight)
    with flag_dnn.use_dnn():
        y = F.prelu(x, weight)

    assert y.shape == shape
    assert y.dtype == dtype
    assert y.device == x.device
    utils.gems_assert_close(y, ref_y, dtype)


@pytest.mark.prelu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_prelu_negative_values(dtype, mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (4, 8, 16)  # 固定一个多维形状方便测试
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) - 2.0

    num_parameters = shape[1] if mode == "channel" else 1
    # 随机生成 weight 而不是全 0.25，更能测出计算的准确性
    weight = (
        torch.randn(num_parameters, dtype=dtype, device=flag_dnn.device) * 0.1
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_y = F.prelu(ref_x, ref_weight)
    with flag_dnn.use_dnn():
        y = F.prelu(x, weight)

    utils.gems_assert_close(y, ref_y, dtype)


@pytest.mark.prelu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_prelu_positive_values(dtype, mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (4, 8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) + 2.0

    num_parameters = shape[1] if mode == "channel" else 1
    weight = (
        torch.randn(num_parameters, dtype=dtype, device=flag_dnn.device) * 0.1
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_y = F.prelu(ref_x, ref_weight)
    with flag_dnn.use_dnn():
        y = F.prelu(x, weight)

    utils.gems_assert_close(y, ref_y, dtype)


@pytest.mark.prelu
@pytest.mark.parametrize("dtype", MIXED_DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_accuracy_prelu_mixed_values(dtype, mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (4, 8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    num_parameters = shape[1] if mode == "channel" else 1
    weight = (
        torch.randn(num_parameters, dtype=dtype, device=flag_dnn.device) * 0.1
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_y = F.prelu(ref_x, ref_weight)
    with flag_dnn.use_dnn():
        y = F.prelu(x, weight)

    utils.gems_assert_close(y, ref_y, dtype)
