import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    INT_DTYPES = utils.ALL_INT_DTYPES


SHAPES = utils.POINTWISE_SHAPES
TENSOR_BOUND_DTYPES = (
    [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float16]
)

# 测试组合：(min_val, max_val)
CLAMP_BOUNDS = [
    (-0.5, 0.5),  # 正常双边界
    (0.0, None),  # 只有下界 (类似于 ReLU)
    (None, 0.0),  # 只有上界
    (0.5, -0.5),  # 异常边界：min > max，预期全部被 clamp 到 max (-0.5)
]


@pytest.mark.clamp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("min_val, max_val", CLAMP_BOUNDS)
def test_accuracy_clamp(dtype, shape, min_val, max_val):
    """最基础的全域测试"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.clamp(ref_x, min=min_val, max=max_val)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_val, max=max_val)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.clamp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("min_val, max_val", CLAMP_BOUNDS)
def test_accuracy_clamp_mixed_values(dtype, shape, min_val, max_val):
    """细粒度测试：显式测试包含正负数的混合情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.clamp(ref_x, min=min_val, max=max_val)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_val, max=max_val)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.clamp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "min_val, max_val", [(0.1, 0.5), (0.5, None), (None, 0.2)]
)
def test_accuracy_clamp_positive_values(dtype, shape, min_val, max_val):
    """细粒度测试：纯正数的情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = (
        torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        + 0.1
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.clamp(ref_x, min=min_val, max=max_val)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_val, max=max_val)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.clamp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "min_val, max_val", [(-0.5, -0.1), (-0.5, None), (None, -0.2)]
)
def test_accuracy_clamp_negative_values(dtype, shape, min_val, max_val):
    """细粒度测试：纯负数的情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = (
        -torch.abs(torch.randn(shape, dtype=dtype, device=flag_dnn.device))
        - 0.1
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.clamp(ref_x, min=min_val, max=max_val)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_val, max=max_val)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.clamp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("min_val, max_val", CLAMP_BOUNDS)
def test_accuracy_clamp_empty_tensor(dtype, min_val, max_val):
    """边界情况：空张量测试"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.clamp(ref_x, min=min_val, max=max_val)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_val, max=max_val)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.clamp
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_clamp_tensor_bounds_same_shape(dtype, shape):
    """测试边界为相同形状 Tensor 的情况"""
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 构造同形状的 min 和 max Tensor
    min_t = torch.randn(shape, dtype=dtype, device=flag_dnn.device) - 1.0
    max_t = min_t + 2.0  # 确保 max > min

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_min_t = utils.to_reference(min_t, ref_kind="compute")
    ref_max_t = utils.to_reference(max_t, ref_kind="compute")

    ref_out = torch.clamp(ref_x, min=ref_min_t, max=ref_max_t)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_t, max=max_t)

    utils.gems_assert_close(out, ref_out, dtype)

    # 测试仅有 Tensor min
    ref_out_min = torch.clamp(ref_x, min=ref_min_t)
    with flag_dnn.use_dnn():
        out_min = torch.clamp(x, min=min_t)

    utils.gems_assert_close(out_min, ref_out_min, dtype)


@pytest.mark.clamp
@pytest.mark.parametrize("dtype", TENSOR_BOUND_DTYPES)
def test_accuracy_clamp_tensor_bounds_broadcast(dtype):
    """测试边界为需要广播的 Tensor (例如标量 Tensor 或 1D Tensor)"""
    shape = (4, 16, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 1. 标量 Tensor 广播
    min_scalar_t = torch.tensor(-0.5, dtype=dtype, device=flag_dnn.device)
    max_scalar_t = torch.tensor(0.5, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_min_scalar_t = utils.to_reference(min_scalar_t, ref_kind="compute")
    ref_max_scalar_t = utils.to_reference(max_scalar_t, ref_kind="compute")

    ref_out = torch.clamp(ref_x, min=ref_min_scalar_t, max=ref_max_scalar_t)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_scalar_t, max=max_scalar_t)

    utils.gems_assert_close(out, ref_out, dtype)

    # 2. 尾部维度广播 (例如 1D Tensor [32] 广播到 [4, 16, 32])
    min_1d_t = torch.randn(32, dtype=dtype, device=flag_dnn.device) - 1.0
    max_1d_t = min_1d_t + 2.0

    ref_min_1d_t = utils.to_reference(min_1d_t, ref_kind="compute")
    ref_max_1d_t = utils.to_reference(max_1d_t, ref_kind="compute")

    ref_out = torch.clamp(ref_x, min=ref_min_1d_t, max=ref_max_1d_t)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=min_1d_t, max=max_1d_t)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.clamp
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_clamp_integer_input(dtype, shape):
    x = torch.randint(-9, 10, shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.clamp(ref_x, min=-2, max=3)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=-2, max=3)

    assert out.dtype == dtype
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.clamp
def test_accuracy_clamp_mixed_dtype_integer_input():
    x = torch.tensor([-2, 1, 3], dtype=torch.int32, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.clamp(ref_x, min=0.5, max=2.5)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=0.5, max=2.5)

    assert out.dtype == torch.float32
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.clamp
def test_accuracy_clamp_bool_with_integer_bounds():
    x = torch.tensor(
        [True, False, True], dtype=torch.bool, device=flag_dnn.device
    )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = torch.clamp(ref_x, min=0, max=1)
    with flag_dnn.use_dnn():
        out = torch.clamp(x, min=0, max=1)

    assert out.dtype == torch.int64
    utils.gems_assert_equal(out, ref_out)
