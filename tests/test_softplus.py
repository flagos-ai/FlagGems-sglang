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


SOFTPLUS_CASES = [
    *[(shape, 1.0, 20.0) for shape in [(0,), *utils.POINTWISE_SHAPES]],
    ((1024,), 0.5, 20.0),
    ((1024,), 2.0, 20.0),
    ((1024,), 1.0, 10.0),
    ((1024,), 1.0, 30.0),
    ((2, 3, 32, 32), 0.5, 10.0),
    ((2, 3, 32, 32), 2.0, 30.0),
]


@pytest.mark.softplus
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, beta, threshold", SOFTPLUS_CASES)
def test_accuracy_softplus(dtype, shape, beta, threshold):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = F.softplus(ref_x, beta=beta, threshold=threshold)

    with flag_dnn.use_dnn():
        out_custom = F.softplus(x_custom, beta=beta, threshold=threshold)

    utils.gems_assert_close(out_custom, out_ref, dtype)

    if x.numel() > 0:
        assert (
            out_custom.data_ptr() != x_custom.data_ptr()
        ), "softplus should be out-of-place, but output shares input memory."
        torch.testing.assert_close(x_custom, x, rtol=0, atol=0)


@pytest.mark.softplus
def test_softplus_invalid_beta():
    x = torch.randn((16,), dtype=torch.float32, device=flag_dnn.device)

    with flag_dnn.use_dnn():
        with pytest.raises(ValueError):
            F.softplus(x, beta=0.0, threshold=20.0)
