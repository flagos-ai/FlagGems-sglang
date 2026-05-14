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


GLU_CASES = [
    ((2, 4), 1),
    ((3, 8), -1),
    ((2, 4, 8), 1),
    ((2, 4, 8), 2),
    ((2, 4, 8), -1),
    ((1, 6, 7, 8), 1),
    ((1, 6, 7, 8), -1),
    ((2, 8, 16, 32), 2),
    ((2, 8, 16, 32), -1),
]


@pytest.mark.glu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, dim", GLU_CASES)
def test_accuracy_glu(dtype, shape, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    ref_x = utils.to_reference(x, ref_kind="compute")
    out_ref = F.glu(ref_x, dim=dim)

    with flag_dnn.use_dnn():
        out_custom = F.glu(x.clone(), dim=dim)

    utils.gems_assert_close(out_custom, out_ref, dtype)


@pytest.mark.glu
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_glu_invalid_odd_dim(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn((2, 5, 8), dtype=dtype, device=flag_dnn.device)

    with pytest.raises(RuntimeError):
        with flag_dnn.use_dnn():
            F.glu(x, dim=1)
