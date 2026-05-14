import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
    BOOL_DTYPES = [torch.bool]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    INT_DTYPES = utils.ALL_INT_DTYPES
    BOOL_DTYPES = utils.BOOL_TYPES


EQ_CASES = [
    *[(shape, shape) for shape in utils.POINTWISE_SHAPES],
    ((128, 256), 0.5),
    ((10, 10), 0),
    ((10, 1), (1, 20)),
    ((2, 3, 4), (4,)),
    ((1, 3, 1, 5), (2, 1, 4, 1)),
    ((), (17, 31)),
]

EQ_DTYPES = FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES


def _rand_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_dnn.device)
    if not dtype.is_floating_point:
        return torch.randint(-5, 5, shape, dtype=dtype, device=flag_dnn.device)
    return torch.randn(shape, dtype=dtype, device=flag_dnn.device)


@pytest.mark.eq
@pytest.mark.parametrize("dtype", EQ_DTYPES)
@pytest.mark.parametrize("input_shape, other_spec", EQ_CASES)
def test_accuracy_eq(dtype, input_shape, other_spec):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    inp = _rand_input(input_shape, dtype)

    # 初始化 other
    if isinstance(other_spec, tuple):
        other = _rand_input(other_spec, dtype)
        # 为了制造相等的条件，随机将一部分 y 赋值为 x 的对应切片 (如果形状允许)
        if input_shape == other_spec:
            mask = torch.rand(input_shape, device=flag_dnn.device) > 0.5
            other = torch.where(mask, inp, other)
    else:
        other = torch.tensor(other_spec, dtype=dtype).item()

    ref_inp = utils.to_reference(inp, ref_kind="logical")
    ref_other = utils.to_reference(other, ref_kind="logical")

    ref_out = torch.eq(ref_inp, ref_other)
    with flag_dnn.use_dnn():
        res_out = torch.eq(inp, other)

    # eq 操作返回必须是 bool 类型
    assert res_out.dtype == torch.bool
    utils.gems_assert_equal(res_out, ref_out)
