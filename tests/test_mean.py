import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


MeanDim = int | tuple[int, ...] | None

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


MEAN_SHAPES = utils.REDUCTION_SHAPES + [
    (128, 256),
    (10, 20, 30),
]
MEAN_CASES: list[tuple[tuple[int, ...], MeanDim, bool]] = [
    (shape, dim, keepdim)
    for shape in MEAN_SHAPES
    for dim in DIM_LIST
    if dim < len(shape)
    for keepdim in KEEPDIM
]
MEAN_CASES += [
    ((1024,), None, False),
    ((2, 3, 4, 5), None, True),
    ((2, 3, 4, 5), (1, 3), False),
    ((10, 20, 30), (0, 1), True),
]


@pytest.mark.mean
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, dim, keepdim", MEAN_CASES)
def test_accuracy_mean(dtype, shape, dim, keepdim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    inp = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.mean(ref_inp, dim=dim, keepdim=keepdim)
    with flag_dnn.use_dnn():
        res_out = torch.mean(inp, dim=dim, keepdim=keepdim)

    reduce_dim = inp.numel() if dim is None else 1
    if dim is not None:
        dims = [dim] if isinstance(dim, int) else list(dim)
        for d in dims:
            reduce_dim *= shape[d % inp.ndim]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.mean
def test_accuracy_mean_with_out_param():
    """测试带有 out 参数的原地写入"""
    inp = torch.randn((10, 20), dtype=torch.float32, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp, ref_kind="compute")

    # 预先分配好一个符合预期的 out tensor
    ref_out = torch.empty(
        (10,),
        dtype=torch.float64 if cfg.TO_CPU else torch.float32,
        device="cpu" if cfg.TO_CPU else flag_dnn.device,
    )
    custom_out = torch.empty(
        (10,), dtype=torch.float32, device=flag_dnn.device
    )

    # 填充一些脏数据验证是否会被覆盖
    custom_out.fill_(-999.0)

    torch.mean(ref_inp, dim=1, out=ref_out)
    with flag_dnn.use_dnn():
        torch.mean(inp, dim=1, out=custom_out)

    utils.gems_assert_close(custom_out, ref_out, torch.float32, reduce_dim=20)


@pytest.mark.mean
@pytest.mark.parametrize(
    "input_dtype, out_dtype",
    [
        (torch.float16, torch.float32),
        (torch.int32, torch.float32),  # 整数求均值，必须指定浮点 dtype
    ],
)
def test_accuracy_mean_dtype_promotion(input_dtype, out_dtype):
    """测试带有 dtype 参数的计算"""
    if input_dtype.is_floating_point:
        inp = torch.randn((10, 20), dtype=input_dtype, device=flag_dnn.device)
    else:
        inp = torch.randint(
            -10, 10, (10, 20), dtype=input_dtype, device=flag_dnn.device
        )

    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.mean(ref_inp, dim=1, dtype=out_dtype)
    with flag_dnn.use_dnn():
        out = torch.mean(inp, dim=1, dtype=out_dtype)

    assert out.dtype == out_dtype
    utils.gems_assert_close(out, ref_out, out_dtype, reduce_dim=20)


@pytest.mark.mean
def test_accuracy_mean_empty_tensor():
    """边界测试：空张量必须产生 NaN，且不能崩溃"""
    # 针对被规约的维度是 0 的情况
    inp = torch.empty((2, 0, 3), dtype=torch.float32, device=flag_dnn.device)
    ref_inp = utils.to_reference(inp, ref_kind="compute")

    ref_out = torch.mean(ref_inp, dim=1)
    with flag_dnn.use_dnn():
        out = torch.mean(inp, dim=1)

    # 注意这里必须加 equal_nan=True，否则 NaN != NaN 会导致测试失败
    utils.gems_assert_close(out, ref_out, torch.float32, equal_nan=True)

    # 针对规约别的维度，但某个不相关维度是 0 的情况
    ref_out_2 = torch.mean(ref_inp, dim=2)
    with flag_dnn.use_dnn():
        out_2 = torch.mean(inp, dim=2)
    utils.gems_assert_close(out_2, ref_out_2, torch.float32, equal_nan=True)
