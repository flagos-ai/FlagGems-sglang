import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
    BOOL_DTYPES = [torch.bool]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    INT_DTYPES = utils.ALL_INT_DTYPES
    BOOL_DTYPES = utils.BOOL_TYPES


ABS_SHAPES = utils.POINTWISE_SHAPES


@pytest.mark.abs
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", ABS_SHAPES)
def test_accuracy_abs(dtype, shape):
    """最基础的全域测试"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    inp = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.abs(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.abs(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.abs
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", ABS_SHAPES)
def test_accuracy_abs_integer(dtype, shape):
    inp = torch.randint(-9, 10, shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.abs(ref_inp)
    with flag_dnn.use_dnn():
        res_out = torch.abs(inp)

    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_equal(res_out, ref_out)
