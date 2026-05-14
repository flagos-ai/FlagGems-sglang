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


TANH_CASES = [(0,), *utils.POINTWISE_SHAPES]


@pytest.mark.tanh
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", TANH_CASES)
def test_accuracy_tanh(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    ref_x = utils.to_reference(x, ref_kind="compute")
    out_ref = F.tanh(ref_x)

    with flag_dnn.use_dnn():
        out_custom = F.tanh(x.clone())

    utils.gems_assert_close(out_custom, out_ref, dtype)
