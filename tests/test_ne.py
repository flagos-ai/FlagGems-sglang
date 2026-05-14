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


NE_CASES = [
    *[(shape, shape) for shape in utils.POINTWISE_SHAPES],
    ((128, 256), 0.5),
    ((10, 10), 0),
    ((10, 1), (1, 20)),
    ((2, 3, 4), (4,)),
    ((1, 3, 1, 5), (2, 1, 4, 1)),
    ((), (17, 31)),
]

NE_DTYPES = FLOAT_DTYPES + INT_DTYPES + BOOL_DTYPES


def _rand_input(shape, dtype):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=flag_dnn.device)
    if not dtype.is_floating_point:
        return torch.randint(-5, 5, shape, dtype=dtype, device=flag_dnn.device)
    return torch.randn(shape, dtype=dtype, device=flag_dnn.device)


@pytest.mark.ne
@pytest.mark.parametrize("dtype", NE_DTYPES)
@pytest.mark.parametrize("input_shape, other_spec", NE_CASES)
def test_accuracy_ne(dtype, input_shape, other_spec):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    inp = _rand_input(input_shape, dtype)

    # 初始化 other
    if isinstance(other_spec, tuple):
        other = _rand_input(other_spec, dtype)
        # 为了制造相等的条件，随机将一部分 y 赋值为 x 的对应切片 (从而让 != 产生 False)
        if input_shape == other_spec:
            mask = torch.rand(input_shape, device=flag_dnn.device) > 0.5
            other = torch.where(mask, inp, other)
    else:
        other = torch.tensor(other_spec, dtype=dtype).item()

    ref_inp = utils.to_reference(inp, ref_kind="logical")
    ref_other = utils.to_reference(other, ref_kind="logical")

    ref_out = torch.ne(ref_inp, ref_other)
    with flag_dnn.use_dnn():
        res_out = torch.ne(inp, other)

    # ne 操作返回必须是 bool 类型
    assert res_out.dtype == torch.bool
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.ne
def test_accuracy_ne_with_out_param():
    """测试带有 out 参数的原地写入"""
    x = torch.tensor([1.0, 2.0, 3.0], device=flag_dnn.device)
    y = torch.tensor([1.0, 0.0, 3.0], device=flag_dnn.device)

    # 预分配
    ref_out = torch.empty(
        (3,), dtype=torch.bool, device="cpu" if cfg.TO_CPU else flag_dnn.device
    )
    custom_out = torch.empty((3,), dtype=torch.bool, device=flag_dnn.device)

    # 填充脏数据
    custom_out.fill_(False)

    ref_x = utils.to_reference(x, ref_kind="logical")
    ref_y = utils.to_reference(y, ref_kind="logical")
    torch.ne(ref_x, ref_y, out=ref_out)
    with flag_dnn.use_dnn():
        torch.ne(x, y, out=custom_out)

    utils.gems_assert_equal(custom_out, ref_out)


@pytest.mark.ne
def test_accuracy_ne_dtype_promotion():
    """测试数据类型提升 (Type Promotion)"""
    x = torch.tensor([1, 2, 3], dtype=torch.int32, device=flag_dnn.device)
    y = torch.tensor(
        [1.0, 2.5, 3.0], dtype=torch.float32, device=flag_dnn.device
    )

    ref_x = utils.to_reference(x, ref_kind="logical")
    ref_y = utils.to_reference(y, ref_kind="logical")
    ref_out = torch.ne(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.ne(x, y)

    utils.gems_assert_equal(out, ref_out)


@pytest.mark.ne
def test_accuracy_ne_empty_tensor():
    """边界测试：空张量的广播与比较"""
    x = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    y = torch.empty((1, 0, 1), dtype=torch.float32, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="logical")
    ref_y = utils.to_reference(y, ref_kind="logical")
    ref_out = torch.ne(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.ne(x, y)

    assert out.shape == ref_out.shape
    assert out.shape == (2, 0, 3)
    utils.gems_assert_equal(out, ref_out)
