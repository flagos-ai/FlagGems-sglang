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


PROD_SHAPES = utils.REDUCTION_SHAPES + [
    (128, 256),
    (10, 20, 30),
]
PROD_CASES = [
    (shape, dim, keepdim)
    for shape in PROD_SHAPES
    for dim in DIM_LIST
    if dim < len(shape)
    for keepdim in KEEPDIM
]
PROD_CASES += [
    ((1024,), None, False),
    ((2, 3, 4, 5), None, True),
]


@pytest.mark.prod
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, dim, keepdim", PROD_CASES)
def test_accuracy_prod(dtype, shape, dim, keepdim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 乘积操作很容易发生指数级溢出/下溢，所以用较小的方差生成数据
    inp = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 0.5 + 1.0
    ref_inp = utils.to_reference(inp, ref_kind="compute")

    # 绕过 PyTorch 原生 prod API 对 dim=None 和 keepdim 的限制
    if dim is None:
        ref_out = torch.prod(ref_inp)  # PyTorch 全局求积只能这么调用
        with flag_dnn.use_dnn():
            res_out = torch.prod(inp)

        if keepdim:
            ref_out = ref_out.view([1] * inp.ndim)  # 手动补齐 keepdim 的形状
            res_out = res_out.view([1] * inp.ndim)
        reduce_dim = inp.numel()
    else:
        ref_out = torch.prod(ref_inp, dim=dim, keepdim=keepdim)
        with flag_dnn.use_dnn():
            res_out = torch.prod(inp, dim=dim, keepdim=keepdim)
        reduce_dim = shape[dim % inp.ndim]

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.prod
def test_accuracy_prod_empty_tensor():
    """边界测试：空张量的乘积必须产生 1 (而不是 0)"""
    inp = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.prod(ref_inp, dim=1)
    with flag_dnn.use_dnn():
        out = torch.prod(inp, dim=1)

    utils.gems_assert_close(out, ref_out, torch.float32)


@pytest.mark.prod
@pytest.mark.parametrize("dtype", BOOL_DTYPES + INT_DTYPES)
def test_accuracy_prod_default_integer_output_dtype(dtype):
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

    ref_out = torch.prod(ref_inp, dim=1)
    with flag_dnn.use_dnn():
        out = torch.prod(inp, dim=1)

    assert out.dtype == torch.int64
    utils.gems_assert_equal(out, ref_out)
