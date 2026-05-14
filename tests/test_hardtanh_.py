import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INTEGER_DTYPES = [torch.int32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    INTEGER_DTYPES = utils.ALL_INT_DTYPES


HARDTANH_CASES = [
    *[(shape, -1.0, 1.0) for shape in [(0,), (0, 3), *utils.POINTWISE_SHAPES]],
    ((1024,), -2.0, 2.0),
    ((4, 8, 16), -0.5, 0.5),
    ((1, 128, 64, 64), -3.0, 3.0),
]

INTEGER_HARDTANH_CASES = [
    *[(shape, -1, 1) for shape in [(0, 3), *utils.POINTWISE_SHAPES]],
    ((17,), -2, 2),
    ((17, 31), -3, 3),
]


@pytest.mark.hardtanh_
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, min_val, max_val", HARDTANH_CASES)
def test_accuracy_hardtanh_(dtype, shape, min_val, max_val):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = F.hardtanh_(ref_x, min_val=min_val, max_val=max_val)

    with flag_dnn.use_dnn():
        out_custom = F.hardtanh_(x_custom, min_val=min_val, max_val=max_val)

    utils.gems_assert_close(out_custom, out_ref, dtype)

    assert out_custom.data_ptr() == x_custom.data_ptr(), (
        "output is not modifying " "the input tensor directly."
    )
    utils.gems_assert_close(x_custom, out_ref, dtype)


@pytest.mark.hardtanh_
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("shape, min_val, max_val", INTEGER_HARDTANH_CASES)
def test_accuracy_hardtanh__integer_dtype(dtype, shape, min_val, max_val):
    x = torch.randint(-5, 6, shape, dtype=dtype, device=flag_dnn.device)

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = F.hardtanh_(ref_x, min_val=min_val, max_val=max_val)
    with flag_dnn.use_dnn():
        out_custom = F.hardtanh_(x_custom, min_val=min_val, max_val=max_val)

    assert out_custom.dtype == dtype
    assert out_custom.data_ptr() == x_custom.data_ptr()
    utils.gems_assert_equal(out_custom, out_ref)
    utils.gems_assert_equal(x_custom, out_ref)
