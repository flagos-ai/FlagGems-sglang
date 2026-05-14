import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


SHAPES = utils.POINTWISE_SHAPES

NEGATIVE_SLOPES = [0.01, 0.2]  # 测试默认斜率和较大的斜率


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu(dtype, shape, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 必须 clone，防止 inplace=True 时原生算子破坏输入数据
    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)
    test_x = x.clone()

    with flag_dnn.use_dnn():
        y = F.leaky_relu(
            test_x, negative_slope=negative_slope, inplace=inplace
        )

    utils.gems_assert_close(y, ref_y, dtype)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_empty_tensor(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)
    test_x = x.clone()

    with flag_dnn.use_dnn():
        y = F.leaky_relu(
            test_x, negative_slope=negative_slope, inplace=inplace
        )

    assert y.shape == (0,)
    assert y.dtype == dtype
    assert y.device == test_x.device
    utils.gems_assert_close(y, ref_y, dtype)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_negative_values(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device) - 2.0

    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)
    test_x = x.clone()

    with flag_dnn.use_dnn():
        y = F.leaky_relu(
            test_x, negative_slope=negative_slope, inplace=inplace
        )

    utils.gems_assert_close(y, ref_y, dtype)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_positive_values(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device) + 2.0

    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)
    test_x = x.clone()

    with flag_dnn.use_dnn():
        y = F.leaky_relu(
            test_x, negative_slope=negative_slope, inplace=inplace
        )

    utils.gems_assert_close(y, ref_y, dtype)


@pytest.mark.leaky_relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_mixed_values(dtype, inplace, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    ref_y = F.leaky_relu(ref_x, negative_slope=negative_slope, inplace=inplace)
    test_x = x.clone()

    with flag_dnn.use_dnn():
        y = F.leaky_relu(
            test_x, negative_slope=negative_slope, inplace=inplace
        )

    utils.gems_assert_close(y, ref_y, dtype)
