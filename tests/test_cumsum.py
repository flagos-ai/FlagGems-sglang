import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
    BOOL_DTYPES = [torch.bool]
    DIM_LIST = [0]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    INT_DTYPES = utils.ALL_INT_DTYPES
    BOOL_DTYPES = utils.BOOL_TYPES
    DIM_LIST = [0, 1]


CUMSUM_SHAPES = utils.REDUCTION_SHAPES + [
    (128, 256),
    (2, 5000),
]
CUMSUM_CASES = [
    (shape, dim)
    for shape in CUMSUM_SHAPES
    for dim in DIM_LIST
    if dim < len(shape)
]


@pytest.mark.cumsum
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, dim", CUMSUM_CASES)
def test_accuracy_cumsum(dtype, shape, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 生成数据：浮点数稍微小点，防止累计求和时溢出
    if dtype.is_floating_point:
        inp = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 0.1
    else:
        inp = torch.randint(-5, 5, shape, dtype=dtype, device=flag_dnn.device)

    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.cumsum(ref_inp, dim=dim)
    with flag_dnn.use_dnn():
        out = torch.cumsum(inp, dim=dim)

    # 整型的累加必须 100% 精确
    if not dtype.is_floating_point:
        utils.gems_assert_equal(out, ref_out)
    else:
        utils.gems_assert_close(out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.cumsum
def test_accuracy_cumsum_out_param():
    """测试原地的 out 参数覆盖"""
    inp = torch.randn((10, 20), dtype=torch.float32, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp, ref_kind=None)
    out = torch.empty((10, 20), dtype=torch.float32, device=flag_dnn.device)

    ref_out = torch.cumsum(ref_inp, dim=0)
    with flag_dnn.use_dnn():
        torch.cumsum(inp, dim=0, out=out)

    utils.gems_assert_close(out, ref_out, torch.float32)


@pytest.mark.cumsum
def test_accuracy_cumsum_empty():
    """测试极其刁钻的空张量边界情况"""
    inp = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp, ref_kind=None)
    ref_out = torch.cumsum(ref_inp, dim=1)
    with flag_dnn.use_dnn():
        out = torch.cumsum(inp, dim=1)
    utils.gems_assert_close(out, ref_out, torch.float32)


@pytest.mark.cumsum
@pytest.mark.parametrize("dtype", BOOL_DTYPES + INT_DTYPES)
def test_accuracy_cumsum_default_integer_output_dtype(dtype):
    if dtype == torch.bool:
        inp = torch.tensor(
            [[True, False], [True, True]],
            dtype=dtype,
            device=flag_dnn.device,
        )
    else:
        inp = torch.tensor(
            [[1, 2], [3, 4]], dtype=dtype, device=flag_dnn.device
        )

    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.cumsum(ref_inp, dim=1)
    with flag_dnn.use_dnn():
        out = torch.cumsum(inp, dim=1)

    assert out.dtype == torch.int64
    utils.gems_assert_equal(out, ref_out)
