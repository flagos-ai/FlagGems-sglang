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


ELU_CASES = [
    *[(shape, 1.0) for shape in [(0,), (0, 3), *utils.POINTWISE_SHAPES]],
    ((1024,), 0.5),
    ((1024,), 2.0),
    ((1, 128, 64, 64), 0.75),
]


@pytest.mark.elu_
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, alpha", ELU_CASES)
def test_accuracy_elu_(dtype, shape, alpha):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 3.0

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = F.elu_(ref_x, alpha=alpha)

    with flag_dnn.use_dnn():
        out_custom = F.elu_(x_custom, alpha=alpha)

    utils.gems_assert_close(out_custom, out_ref, dtype)

    assert out_custom.data_ptr() == x_custom.data_ptr(), (
        "output is not modifying " "the input tensor directly."
    )
    utils.gems_assert_close(x_custom, out_ref, dtype)
