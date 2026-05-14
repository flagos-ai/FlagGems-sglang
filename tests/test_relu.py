import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INTEGER_DTYPES = [torch.int32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    INTEGER_DTYPES = utils.ALL_INT_DTYPES


SHAPES = utils.POINTWISE_SHAPES


@pytest.mark.relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu(dtype, shape, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # Inplace 测试必须隔离显存
    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    # ReLU 无精度损失，直接卡死容差
    utils.gems_assert_close(y, ref_y, dtype)
    if inplace:
        assert (
            y.data_ptr() == test_x.data_ptr()
        ), "Inplace operation failed to reuse memory pointer."


@pytest.mark.relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu_empty_tensor(dtype, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试多维度的空张量
    x = torch.empty((2, 0, 3), dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    assert y.shape == (2, 0, 3)
    assert y.dtype == dtype
    assert y.device == test_x.device
    utils.gems_assert_close(y, ref_y, dtype)


@pytest.mark.relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu_negative_values(dtype, shape, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 纯负数测试
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) - 2.0

    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    utils.gems_assert_close(y, ref_y, dtype)
    if inplace:
        assert y.data_ptr() == test_x.data_ptr()


@pytest.mark.relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu_positive_values(dtype, shape, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 纯正数测试
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) + 2.0

    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    utils.gems_assert_close(y, ref_y, dtype)
    if inplace:
        assert y.data_ptr() == test_x.data_ptr()


@pytest.mark.relu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu_mixed_values(dtype, shape, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 混合正负数测试
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    utils.gems_assert_close(y, ref_y, dtype)
    if inplace:
        assert y.data_ptr() == test_x.data_ptr()


@pytest.mark.relu
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_accuracy_relu_integer_dtype(dtype, shape, inplace):
    x = torch.randint(-5, 6, shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    test_x = x.clone()

    ref_y = torch.nn.functional.relu(ref_x, inplace=inplace)
    with flag_dnn.use_dnn():
        y = torch.nn.functional.relu(test_x, inplace=inplace)

    assert y.dtype == dtype
    utils.gems_assert_equal(y, ref_y)
    if inplace:
        assert y.data_ptr() == test_x.data_ptr()
