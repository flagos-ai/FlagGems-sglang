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


CUMPROD_SHAPES = utils.REDUCTION_SHAPES + [
    (128, 256),
    (128, 5000),
]
CUMPROD_CASES = [
    (shape, dim)
    for shape in CUMPROD_SHAPES
    for dim in DIM_LIST
    if dim < len(shape)
]


@pytest.mark.cumprod
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, dim", CUMPROD_CASES)
def test_accuracy_cumprod(dtype, shape, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 防溢出策略
    if dtype.is_floating_point:
        # 围绕 1.0 波动
        inp = (
            torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 0.1 + 1.0
        )
    else:
        # 只有 1 和 -1，避开 int64 溢出
        inp = (
            torch.randint(0, 2, shape, dtype=dtype, device=flag_dnn.device) * 2
            - 1
        )

    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.cumprod(ref_inp, dim=dim)

    with flag_dnn.use_dnn():
        out = torch.cumprod(inp, dim=dim)

    if not dtype.is_floating_point:
        utils.gems_assert_equal(out, ref_out)
    else:
        utils.gems_assert_close(out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.cumprod
def test_accuracy_cumprod_empty():
    inp = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp, ref_kind=None)
    ref_out = torch.cumprod(ref_inp, dim=1)
    with flag_dnn.use_dnn():
        out = torch.cumprod(inp, dim=1)
    utils.gems_assert_close(out, ref_out, torch.float32)


@pytest.mark.cumprod
@pytest.mark.parametrize("dtype", BOOL_DTYPES + INT_DTYPES)
def test_accuracy_cumprod_default_integer_output_dtype(dtype):
    if dtype == torch.bool:
        inp = torch.tensor(
            [[True, True], [False, True]],
            dtype=dtype,
            device=flag_dnn.device,
        )
    else:
        inp = torch.tensor(
            [[1, 2], [3, 4]], dtype=dtype, device=flag_dnn.device
        )

    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.cumprod(ref_inp, dim=1)
    with flag_dnn.use_dnn():
        out = torch.cumprod(inp, dim=1)

    assert out.dtype == torch.int64
    utils.gems_assert_equal(out, ref_out)
