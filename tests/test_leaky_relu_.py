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


SHAPES = utils.POINTWISE_SHAPES
NEGATIVE_SLOPES = [0.01, 0.2]  # 测试默认斜率和较大的斜率


@pytest.mark.leaky_relu_
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("negative_slope", NEGATIVE_SLOPES)
def test_accuracy_leaky_relu_(dtype, shape, negative_slope):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x.clone(), ref_kind="compute")
    ref_y = F.leaky_relu_(ref_x, negative_slope=negative_slope)
    test_x = x.clone()

    with flag_dnn.use_dnn():
        y = F.leaky_relu_(test_x, negative_slope=negative_slope)

    utils.gems_assert_close(y, ref_y, dtype)

    assert y.data_ptr() == test_x.data_ptr(), (
        "output is not modifying " "the input tensor directly."
    )
    utils.gems_assert_close(test_x, ref_y, dtype)
