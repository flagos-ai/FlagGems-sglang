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


SOFTSHRINK_CASES = [
    *[(shape, 0.5) for shape in [(0,), *utils.POINTWISE_SHAPES]],
    ((1024,), 0.0),
    ((1024,), 1.0),
    ((2, 3, 32, 32), 1.5),
]


@pytest.mark.softshrink
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, lambd", SOFTSHRINK_CASES)
def test_accuracy_softshrink(dtype, shape, lambd):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = F.softshrink(ref_x, lambd=lambd)

    with flag_dnn.use_dnn():
        out_custom = F.softshrink(x_custom, lambd=lambd)

    utils.gems_assert_close(out_custom, out_ref, dtype)

    if x.numel() > 0:
        assert (
            out_custom.data_ptr() != x_custom.data_ptr()
        ), "softshrink should be out-of-place, but output shares input memory."
        torch.testing.assert_close(x_custom, x, rtol=0, atol=0)


@pytest.mark.softshrink
def test_softshrink_invalid_lambd():
    x = torch.randn((16,), dtype=torch.float32, device=flag_dnn.device)

    with flag_dnn.use_dnn():
        with pytest.raises(ValueError):
            F.softshrink(x, lambd=-0.5)
