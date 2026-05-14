import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    INT_DTYPES = utils.ALL_INT_DTYPES


SHAPES = utils.POINTWISE_SHAPES


@pytest.mark.neg
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_neg(dtype, shape):
    """最基础的全域测试"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.neg(ref_x)
    with flag_dnn.use_dnn():
        out = torch.neg(x)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.neg
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_neg_mixed_values(dtype, shape):
    """细粒度测试：显式测试包含正负数的混合情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.neg(ref_x)
    with flag_dnn.use_dnn():
        out = torch.neg(x)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.neg
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_neg_positive_values(dtype, shape):
    """细粒度测试：纯正数的情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = (
        torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        + 0.1
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.neg(ref_x)
    with flag_dnn.use_dnn():
        out = torch.neg(x)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.neg
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_neg_negative_values(dtype, shape):
    """细粒度测试：纯负数的情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = (
        -torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        - 0.1
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.neg(ref_x)
    with flag_dnn.use_dnn():
        out = torch.neg(x)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.neg
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_neg_empty_tensor(dtype):
    """边界情况：空张量测试"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.neg(ref_x)
    with flag_dnn.use_dnn():
        out = torch.neg(x)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.neg
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_neg_integer(dtype, shape):
    x = torch.randint(-9, 10, shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.neg(ref_x)
    with flag_dnn.use_dnn():
        out = torch.neg(x)

    assert out.dtype == ref_out.dtype
    utils.gems_assert_equal(out, ref_out)
