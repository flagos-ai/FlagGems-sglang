import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    NON_FLOAT_DTYPES = [torch.bool, torch.int32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    NON_FLOAT_DTYPES = utils.BOOL_TYPES + utils.ALL_INT_DTYPES


SHAPES = utils.POINTWISE_SHAPES

BROADCAST_SHAPES = [
    ((4, 4), (4,)),  # 1D broadcast to 2D
    ((2, 3, 4), (3, 1)),  # 内部维度广播
    ((1, 5), (5, 5)),  # 单一维度扩展
    ((2, 1, 4, 1), (1, 3, 1, 5)),  # 复杂高维双向广播
    ((), (17, 31)),  # 标量 Tensor 广播到矩阵
]


def _rand_non_float(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_dnn.device)
    return torch.randint(-5, 6, shape, dtype=dtype, device=flag_dnn.device)


@pytest.mark.add
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_add(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.add(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.add(x, y)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量 (shape 为 0)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.add(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.add(x, y)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_scalar(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    scalar = 3.14

    ref_x = utils.to_reference(x, ref_kind="compute")

    ref_out = torch.add(ref_x, scalar)
    with flag_dnn.use_dnn():
        out = torch.add(x, scalar)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_add_alpha(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    alpha = 2.5

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.add(ref_x, ref_y, alpha=alpha)
    with flag_dnn.use_dnn():
        out = torch.add(x, y, alpha=alpha)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.add
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("input_shape, other_shape", BROADCAST_SHAPES)
def test_accuracy_add_broadcast(dtype, input_shape, other_shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(input_shape, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(other_shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.add(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.add(x, y)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.add
def test_accuracy_add_bool_tensor():
    x = torch.tensor(
        [True, False, True], dtype=torch.bool, device=flag_dnn.device
    )
    y = torch.tensor(
        [True, True, False], dtype=torch.bool, device=flag_dnn.device
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.add(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.add(x, y)

    assert out.dtype == torch.bool
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.add
@pytest.mark.parametrize("dtype", NON_FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_add_non_float_dtype(dtype, shape):
    x = _rand_non_float(shape, dtype)
    y = _rand_non_float(shape, dtype)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.add(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.add(x, y)

    assert out.dtype == ref_out.dtype
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.add
def test_accuracy_add_integer_alpha():
    x = torch.randint(-5, 6, (128,), dtype=torch.int32, device=flag_dnn.device)
    y = torch.randint(-5, 6, (128,), dtype=torch.int32, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.add(ref_x, ref_y, alpha=2)
    with flag_dnn.use_dnn():
        out = torch.add(x, y, alpha=2)

    assert out.dtype == torch.int32
    utils.gems_assert_equal(out, ref_out)
