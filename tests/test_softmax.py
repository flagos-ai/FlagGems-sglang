import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


SHAPES = [(32,), (1024,), (2, 16), (4, 8, 32), (2, 4, 16, 16)]
DIMS = [-1, 0, 1, 2]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _softmax_reduce_dim(shape, dim):
    dim_norm = dim if dim >= 0 else dim + len(shape)
    return max(shape[dim_norm], 1)


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIMS)
def test_accuracy_softmax(dtype, shape, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 如果指定的 dim 超出了当前 shape 的维度范围，则跳过
    if dim is not None and (dim >= len(shape) or dim < -len(shape)):
        pytest.skip(f"Dimension {dim} is out of bounds for shape {shape}")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = F.softmax(ref_x, dim=dim)
    with flag_dnn.use_dnn():
        y = F.softmax(x, dim=dim)

    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_softmax_reduce_dim(shape, dim)
    )


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0])
def test_accuracy_softmax_empty_tensor(dtype, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量
    shape = (0, 4, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = F.softmax(ref_x, dim=dim)
    with flag_dnn.use_dnn():
        y = F.softmax(x, dim=dim)

    assert y.shape == shape
    assert y.dtype == dtype
    assert y.device == x.device
    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_softmax_reduce_dim(shape, dim)
    )


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 1])
def test_accuracy_softmax_large_values(dtype, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 专门测试大数值 (如 100 左右)，验证算子内部减去最大值的防溢出机制是否生效
    shape = (4, 8, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 10.0 + 100.0

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = F.softmax(ref_x, dim=dim)
    with flag_dnn.use_dnn():
        y = F.softmax(x, dim=dim)

    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_softmax_reduce_dim(shape, dim)
    )


@pytest.mark.softmax
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_accuracy_softmax_mixed_values(dtype, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 混合常规数据测试
    shape = (4, 8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = F.softmax(ref_x, dim=dim)
    with flag_dnn.use_dnn():
        y = F.softmax(x, dim=dim)

    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_softmax_reduce_dim(shape, dim)
    )
