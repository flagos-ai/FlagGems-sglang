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


SELU_CASES = [
    (shape, inplace)
    for shape in [(0,), *utils.POINTWISE_SHAPES]
    for inplace in [False, True]
]


@pytest.mark.selu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, inplace", SELU_CASES)
def test_accuracy_selu(dtype, shape, inplace):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = F.selu(ref_x, inplace=inplace)

    with flag_dnn.use_dnn():
        out_custom = F.selu(x_custom, inplace=inplace)

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
