import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# (
#   input_shape,
#   weight_shape,
#   has_bias,
#   stride,
#   padding,
#   dilation,
#   groups,
# )
CONV2D_CASES = [
    # 普通卷积，最基础场景
    ((1, 3, 8, 8), (4, 3, 3, 3), True, 1, 0, 1, 1),
    ((2, 3, 16, 16), (8, 3, 3, 3), False, 1, 1, 1, 1),
    ((2, 3, 15, 17), (6, 3, 5, 5), True, 2, 2, 1, 1),
    ((1, 3, 13, 11), (5, 3, 1, 1), True, 1, 0, 1, 1),
    # dilation 场景
    ((1, 3, 17, 17), (4, 3, 3, 3), True, 1, 2, 2, 1),
    ((2, 3, 19, 21), (7, 3, 3, 3), False, 2, 2, 2, 1),
    # groups = 2
    ((1, 4, 12, 12), (8, 2, 3, 3), True, 1, 1, 1, 2),
    ((2, 4, 18, 14), (6, 2, 5, 5), False, 2, 2, 1, 2),
    # depthwise 卷积
    ((1, 4, 10, 11), (4, 1, 3, 3), True, 1, 1, 1, 4),
    ((2, 8, 16, 16), (8, 1, 5, 5), False, 1, 2, 1, 8),
    # padding='same'
    ((1, 2, 9, 9), (4, 2, 3, 3), True, 1, "same", 1, 1),
    ((1, 4, 13, 13), (8, 2, 3, 3), False, 1, "same", 2, 2),
    # padding='valid'
    ((1, 3, 12, 12), (6, 3, 3, 3), True, 1, "valid", 1, 1),
    ((2, 4, 20, 18), (8, 2, 5, 5), False, 2, "valid", 1, 2),
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _conv_reduce_dim(weight_shape):
    return max(weight_shape[1] * weight_shape[2] * weight_shape[3], 1)


@pytest.mark.conv2d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "input_shape, weight_shape, has_bias, stride, padding, dilation, groups",
    CONV2D_CASES,
)
def test_accuracy_conv2d(
    dtype,
    input_shape,
    weight_shape,
    has_bias,
    stride,
    padding,
    dilation,
    groups,
):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(input_shape, dtype=dtype, device=flag_dnn.device)
    w = torch.randn(weight_shape, dtype=dtype, device=flag_dnn.device)
    b = (
        torch.randn(weight_shape[0], dtype=dtype, device=flag_dnn.device)
        if has_bias
        else None
    )

    x_ref = utils.to_reference(x.clone(), ref_kind="compute")
    w_ref = utils.to_reference(w.clone(), ref_kind="compute")
    b_ref = (
        utils.to_reference(b.clone(), ref_kind="compute")
        if b is not None
        else None
    )

    x_custom = x.clone()
    w_custom = w.clone()
    b_custom = b.clone() if b is not None else None

    out_ref = F.conv2d(
        x_ref,
        w_ref,
        b_ref,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    with flag_dnn.use_dnn():
        out_custom = F.conv2d(
            x_custom,
            w_custom,
            b_custom,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    utils.gems_assert_close(
        out_custom,
        out_ref,
        dtype,
        reduce_dim=_conv_reduce_dim(weight_shape),
        atol=2e-2,
    )
