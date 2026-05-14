import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    NON_FLOAT_DTYPES = [torch.bool, torch.int32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    NON_FLOAT_DTYPES = utils.BOOL_TYPES + utils.ALL_INT_DTYPES


SHAPES = utils.POINTWISE_SHAPES


def _get_non_negative_tensor(shape, dtype, device):
    """
    生成非负张量，避免对负数求平方根产生 NaN，从而破坏容差对比。
    """
    return torch.abs(torch.randn(shape, dtype=dtype, device=device))


@pytest.mark.sqrt
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_sqrt(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 确保输入全为非负数
    x = _get_non_negative_tensor(shape, dtype, flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")

    ref_out = torch.sqrt(ref_x)
    with flag_dnn.use_dnn():
        out = torch.sqrt(x)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.sqrt
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sqrt_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量 (shape 为 0)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.sqrt(ref_x)
    with flag_dnn.use_dnn():
        out = torch.sqrt(x)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.sqrt
@pytest.mark.parametrize("dtype", NON_FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_sqrt_integral_and_bool(dtype, shape):
    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=dtype, device=flag_dnn.device)
    else:
        base = torch.randint(0, 4, shape, dtype=dtype, device=flag_dnn.device)
        x = base * base

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.sqrt(ref_x)
    with flag_dnn.use_dnn():
        out = torch.sqrt(x)

    assert out.dtype == torch.float32
    utils.gems_assert_equal(out, ref_out)
