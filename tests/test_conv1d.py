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
#   unbatched,
#   noncontiguous,
# )
CONV1D_CASES = [
    ((2, 3, 16), (4, 3, 3), True, 1, 1, 1, 1, False, False),
    ((1, 3, 17), (5, 3, 1), True, 1, 0, 1, 1, False, False),
    ((2, 4, 31), (8, 4, 5), False, 2, 2, 1, 1, False, False),
    ((1, 3, 29), (4, 3, 3), True, 1, 2, 2, 1, False, False),
    ((2, 4, 23), (6, 2, 3), True, 1, 1, 1, 2, False, False),
    ((2, 8, 33), (8, 1, 5), False, 1, 2, 1, 8, False, False),
    ((1, 2, 19), (4, 2, 4), True, 1, "same", 1, 1, False, False),
    ((2, 4, 25), (8, 2, 3), False, 1, "same", 2, 2, False, False),
    ((2, 3, 21), (5, 3, 3), True, 2, "valid", 1, 1, False, False),
    ((3, 13), (4, 3, 3), True, 1, 1, 1, 1, True, False),
    ((2, 3, 18), (4, 3, 3), True, 1, 1, 1, 1, False, True),
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _conv_reduce_dim(weight_shape):
    return max(weight_shape[1] * weight_shape[2], 1)


def _make_tensor(shape, dtype, noncontiguous=False):
    if not noncontiguous:
        return torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    widened = tuple(shape[:-1]) + (shape[-1] * 2,)
    base = torch.randn(widened, dtype=dtype, device=flag_dnn.device)
    return base[..., ::2]


@pytest.mark.conv1d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    (
        "input_shape, weight_shape, has_bias, stride, padding, dilation, "
        "groups, unbatched, noncontiguous"
    ),
    CONV1D_CASES,
)
def test_accuracy_conv1d(
    dtype,
    input_shape,
    weight_shape,
    has_bias,
    stride,
    padding,
    dilation,
    groups,
    unbatched,
    noncontiguous,
):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = _make_tensor(input_shape, dtype, noncontiguous)
    w = _make_tensor(weight_shape, dtype, noncontiguous)
    b = (
        _make_tensor((weight_shape[0] * 2,), dtype)[::2]
        if has_bias and noncontiguous
        else (
            torch.randn(weight_shape[0], dtype=dtype, device=flag_dnn.device)
            if has_bias
            else None
        )
    )

    x_ref = utils.to_reference(x.detach(), ref_kind="compute")
    w_ref = utils.to_reference(w.detach(), ref_kind="compute")
    b_ref = (
        utils.to_reference(b.detach(), ref_kind="compute")
        if b is not None
        else None
    )

    out_ref = F.conv1d(
        x_ref,
        w_ref,
        b_ref,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    with flag_dnn.use_dnn():
        out_custom = F.conv1d(
            x,
            w,
            b,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    assert out_custom.dim() == (2 if unbatched else 3)
    utils.gems_assert_close(
        out_custom,
        out_ref,
        dtype,
        reduce_dim=_conv_reduce_dim(weight_shape),
        atol=2e-2,
    )


@pytest.mark.conv1d
def test_conv1d_unsupported_complex():
    x = torch.randn((1, 2, 8), dtype=torch.complex64, device=flag_dnn.device)
    w = torch.randn((4, 2, 3), dtype=torch.complex64, device=flag_dnn.device)
    with flag_dnn.use_dnn(include=["conv1d"]):
        with pytest.raises(NotImplementedError):
            F.conv1d(x, w, padding=1)


@pytest.mark.conv1d
def test_conv1d_requires_grad_is_unsupported():
    x = torch.randn(
        (1, 2, 8),
        dtype=torch.float32,
        device=flag_dnn.device,
        requires_grad=True,
    )
    w = torch.randn(
        (4, 2, 3),
        dtype=torch.float32,
        device=flag_dnn.device,
    )
    with flag_dnn.use_dnn(include=["conv1d"]):
        with pytest.raises(NotImplementedError):
            F.conv1d(x, w, padding=1)


@pytest.mark.conv1d
def test_conv1d_same_padding_stride_error():
    x = torch.randn((1, 2, 8), dtype=torch.float32, device=flag_dnn.device)
    w = torch.randn((4, 2, 3), dtype=torch.float32, device=flag_dnn.device)
    with flag_dnn.use_dnn(include=["conv1d"]):
        with pytest.raises(RuntimeError):
            F.conv1d(x, w, stride=2, padding="same")
