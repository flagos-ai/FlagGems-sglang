import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


SHAPES = utils.POINTWISE_SHAPES

BROADCAST_SHAPES = [
    ((4, 4), (4,)),  # 1D broadcast to 2D
    ((2, 3, 4), (3, 1)),  # 内部维度广播
    ((1, 5), (5, 5)),  # 单一维度扩展
    ((2, 1, 4, 1), (1, 3, 1, 5)),  # 复杂高维双向广播
    ((), (17, 31)),  # 标量 Tensor 广播到矩阵
]


def _get_positive_tensor(shape, dtype, device):
    """
    生成严格为正数的张量。
    对于 pow 运算，如果底数为负数且指数为小数，会产生 NaN 或复数。
    我们限制底数为正，确保能够进行稳定的精度比对。
    """
    return torch.abs(torch.randn(shape, dtype=dtype, device=device)) + 0.5


@pytest.mark.pow
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_pow_tensor(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 底数必须为正数
    x = _get_positive_tensor(shape, dtype, flag_dnn.device)
    # 指数可以用普通随机数 (负指数即为取倒数)
    y = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.pow(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.pow(x, y)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.pow
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量 (shape 为 0)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.pow(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.pow(x, y)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.pow
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_scalar_exponent(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # Tensor base, scalar exponent
    x = _get_positive_tensor((100,), dtype, flag_dnn.device)
    scalar_exp = 2.5

    ref_x = utils.to_reference(x, ref_kind="compute")

    ref_out = torch.pow(ref_x, scalar_exp)
    with flag_dnn.use_dnn():
        out = torch.pow(x, scalar_exp)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.pow
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_pow_scalar_base(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # Scalar base, tensor exponent
    scalar_base = 3.14
    y = torch.randn(100, dtype=dtype, device=flag_dnn.device)

    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.pow(scalar_base, ref_y)
    with flag_dnn.use_dnn():
        out = torch.pow(scalar_base, y)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.pow
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("input_shape, other_shape", BROADCAST_SHAPES)
def test_accuracy_pow_broadcast(dtype, input_shape, other_shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = _get_positive_tensor(input_shape, dtype, flag_dnn.device)
    y = torch.randn(other_shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.pow(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.pow(x, y)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.pow
def test_accuracy_pow_integer_and_bool_dtype():
    x_int = torch.tensor([2, 3, 4], dtype=torch.int32, device=flag_dnn.device)
    x_bool = torch.tensor(
        [True, False, True], dtype=torch.bool, device=flag_dnn.device
    )

    ref_x_int = utils.to_reference(x_int, ref_kind="compute")
    ref_x_bool = utils.to_reference(x_bool, ref_kind="compute")

    ref_int = torch.pow(ref_x_int, 2.0)
    ref_bool = torch.pow(ref_x_bool, 2)
    with flag_dnn.use_dnn():
        out_int = torch.pow(x_int, 2.0)
        out_bool = torch.pow(x_bool, 2)

    assert out_int.dtype == torch.float32
    assert out_bool.dtype == torch.int64
    utils.gems_assert_equal(out_int, ref_int)
    utils.gems_assert_equal(out_bool, ref_bool)
