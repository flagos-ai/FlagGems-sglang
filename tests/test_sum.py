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
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    INT_DTYPES = utils.ALL_INT_DTYPES
    BOOL_DTYPES = utils.BOOL_TYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


SUM_SHAPES = utils.REDUCTION_SHAPES + [
    (128, 256),
    (10, 20, 30),
]

SUM_CASES = [
    (shape, dim, keepdim)
    for shape in SUM_SHAPES
    for dim in DIM_LIST
    if dim < len(shape)
    for keepdim in KEEPDIM
]


@pytest.mark.sum
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, dim, keepdim", SUM_CASES)
def test_accuracy_sum(dtype, shape, dim, keepdim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    inp = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    with flag_dnn.use_dnn():
        res_out = torch.sum(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.sum
@pytest.mark.parametrize("dtype", BOOL_DTYPES + INT_DTYPES)
def test_accuracy_sum_default_integer_output_dtype(dtype):
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

    ref_out = torch.sum(ref_inp, dim=1)
    with flag_dnn.use_dnn():
        res_out = torch.sum(inp, dim=1)

    assert res_out.dtype == torch.int64
    utils.gems_assert_equal(res_out, ref_out)
