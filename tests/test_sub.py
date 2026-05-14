import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INTEGER_DTYPES = [torch.int32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    INTEGER_DTYPES = utils.ALL_INT_DTYPES


SHAPES = utils.POINTWISE_SHAPES

BROADCAST_SHAPES = [
    ((4, 4), (4,)),  # 1D broadcast to 2D
    ((2, 3, 4), (3, 1)),  # 内部维度广播
    ((1, 5), (5, 5)),  # 单一维度扩展
    ((2, 1, 4, 1), (1, 3, 1, 5)),  # 复杂高维双向广播
    ((), (17, 31)),  # 标量 Tensor 广播到矩阵
]


@pytest.mark.sub
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_sub(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.sub(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.sub(x, y)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.sub
@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_sub_integer_dtype(dtype, shape):
    x = torch.randint(-5, 6, shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randint(-5, 6, shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.sub(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.sub(x, y)

    assert out.dtype == dtype
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.sub
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量 (shape 为 0)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.sub(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.sub(x, y)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.sub
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub_scalar(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    scalar = 3.14

    ref_x = utils.to_reference(x, ref_kind="compute")

    ref_out = torch.sub(ref_x, scalar)
    with flag_dnn.use_dnn():
        out = torch.sub(x, scalar)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.sub
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_sub_alpha(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    alpha = 2.5

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.sub(ref_x, ref_y, alpha=alpha)
    with flag_dnn.use_dnn():
        out = torch.sub(x, y, alpha=alpha)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.sub
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("input_shape, other_shape", BROADCAST_SHAPES)
def test_accuracy_sub_broadcast(dtype, input_shape, other_shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(input_shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(other_shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.sub(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.sub(x, y)

    utils.gems_assert_close(out, ref_out, dtype)
