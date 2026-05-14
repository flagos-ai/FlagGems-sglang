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
    *[
        (shape, -1.0, 1.0, inplace)
        for shape in [(0,), (0, 3), *utils.POINTWISE_SHAPES]
        for inplace in [False, True]
    ],
    ((1024,), -2.0, 2.0, False),
    ((1024,), -2.0, 2.0, True),
    ((4, 8, 16), -0.5, 0.5, False),
    ((4, 8, 16), -0.5, 0.5, True),
    ((1, 128, 64, 64), -3.0, 3.0, False),
    ((1, 128, 64, 64), -3.0, 3.0, True),
]

INTEGER_HARDTANH_CASES = [
    *[
        (shape, -1, 1, inplace)
        for shape in [(0, 3), *utils.POINTWISE_SHAPES]
        for inplace in [False, True]
    ],
    ((17,), -2, 2, True),
    ((17, 31), -3, 3, False),
]


@pytest.mark.hardtanh
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, min_val, max_val, inplace", HARDTANH_CASES)
def test_accuracy_hardtanh(dtype, shape, min_val, max_val, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = F.hardtanh(
        ref_x, min_val=min_val, max_val=max_val, inplace=inplace
    )

    with flag_dnn.use_dnn():
        out_custom = F.hardtanh(
            x_custom, min_val=min_val, max_val=max_val, inplace=inplace
        )

    utils.gems_assert_close(out_custom, out_ref, dtype)

    if inplace:
        assert out_custom.data_ptr() == x_custom.data_ptr(), (
            "Inplace flag is True, but output is not modifying "
            "the input tensor directly."
        )
        utils.gems_assert_close(x_custom, out_ref, dtype)
    else:
        if x.numel() > 0:
            assert out_custom.data_ptr() != x_custom.data_ptr(), (
                "Inplace flag is False, but output is modifying "
                "the input tensor memory."
            )
        torch.testing.assert_close(x_custom, x, rtol=0, atol=0)


@pytest.mark.hardtanh
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize(
    "shape, min_val, max_val, inplace", INTEGER_HARDTANH_CASES
)
def test_accuracy_hardtanh_integer_dtype(
    dtype, shape, min_val, max_val, inplace
):
    x = torch.randint(-5, 6, shape, dtype=dtype, device=flag_dnn.device)

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = F.hardtanh(
        ref_x, min_val=min_val, max_val=max_val, inplace=inplace
    )
    with flag_dnn.use_dnn():
        out_custom = F.hardtanh(
            x_custom, min_val=min_val, max_val=max_val, inplace=inplace
        )

    assert out_custom.dtype == dtype
    utils.gems_assert_equal(out_custom, out_ref)

    if inplace:
        assert out_custom.data_ptr() == x_custom.data_ptr()
        utils.gems_assert_equal(x_custom, out_ref)
    else:
        if x.numel() > 0:
            assert out_custom.data_ptr() != x_custom.data_ptr()
        torch.testing.assert_close(x_custom, x, rtol=0, atol=0)
