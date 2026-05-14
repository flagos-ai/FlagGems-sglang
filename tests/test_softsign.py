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


SOFTSIGN_CASES = [(0,), *utils.POINTWISE_SHAPES]


@pytest.mark.softsign
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SOFTSIGN_CASES)
def test_accuracy_softsign(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = F.softsign(ref_x)

    with flag_dnn.use_dnn():
        out_custom = F.softsign(x_custom)

    utils.gems_assert_close(out_custom, out_ref, dtype)

    if x.numel() > 0:
        assert (
            out_custom.data_ptr() != x_custom.data_ptr()
        ), "softsign should be out-of-place, but output shares input memory."
        torch.testing.assert_close(x_custom, x, rtol=0, atol=0)
