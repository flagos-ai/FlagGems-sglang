import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# (shape, num_groups) 必须保证 shape[1] % num_groups == 0
SHAPES_AND_GROUPS = [
    ((2, 32), 4),  # 2D 张量，类似全连接后的归一化
    ((2, 32, 16), 8),  # 3D 张量
    ((4, 16, 32, 32), 4),  # 4D 张量，标准 CV 特征图
    ((2, 8, 16, 16, 16), 2),  # 5D 张量，如 3D 卷积输出
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _group_reduce_dim(shape, num_groups):
    reduce_dim = shape[1] // num_groups
    for dim_size in shape[2:]:
        reduce_dim *= dim_size
    return max(reduce_dim, 1)


@pytest.mark.group_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, num_groups", SHAPES_AND_GROUPS)
@pytest.mark.parametrize("affine", [False, True])
def test_accuracy_group_norm(dtype, shape, num_groups, affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    weight, bias = None, None
    C = shape[1]
    if affine:
        # GroupNorm 的参数维度等于 Channel 数量
        weight = torch.randn(C, dtype=dtype, device=flag_dnn.device)
        bias = torch.randn(C, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_bias = utils.to_reference(bias, ref_kind="compute")
    ref_y = F.group_norm(ref_x, num_groups, weight=ref_weight, bias=ref_bias)
    with flag_dnn.use_dnn():
        y = F.group_norm(x, num_groups, weight=weight, bias=bias)

    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_group_reduce_dim(shape, num_groups)
    )


@pytest.mark.group_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [False, True])
def test_accuracy_group_norm_empty_tensor(dtype, affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # N=0, 空张量测试
    shape = (0, 16, 8, 8)
    num_groups = 4
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    C = shape[1]
    weight, bias = None, None
    if affine:
        weight = torch.randn(C, dtype=dtype, device=flag_dnn.device)
        bias = torch.randn(C, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_bias = utils.to_reference(bias, ref_kind="compute")
    ref_y = F.group_norm(ref_x, num_groups, weight=ref_weight, bias=ref_bias)
    with flag_dnn.use_dnn():
        y = F.group_norm(x, num_groups, weight=weight, bias=bias)

    assert y.shape == shape
    assert y.dtype == dtype
    assert y.device == x.device
    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_group_reduce_dim(shape, num_groups)
    )


@pytest.mark.group_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [False, True])
def test_accuracy_group_norm_large_values(dtype, affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (2, 32, 16, 16)
    num_groups = 8
    C = shape[1]

    # 控制大数值溢出范围
    if dtype in [torch.float16, torch.bfloat16]:
        x = (
            torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 10.0
            + 50.0
        )
    else:
        x = (
            torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 500.0
            + 5000.0
        )

    weight, bias = None, None
    if affine:
        if dtype in [torch.float16, torch.bfloat16]:
            weight = torch.randn(C, dtype=dtype, device=flag_dnn.device) * 1.5
            bias = torch.randn(C, dtype=dtype, device=flag_dnn.device) * 8.0
        else:
            weight = torch.randn(C, dtype=dtype, device=flag_dnn.device) * 10.0
            bias = torch.randn(C, dtype=dtype, device=flag_dnn.device) * 100.0

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_bias = utils.to_reference(bias, ref_kind="compute")
    ref_y = F.group_norm(ref_x, num_groups, weight=ref_weight, bias=ref_bias)
    with flag_dnn.use_dnn():
        y = F.group_norm(x, num_groups, weight=weight, bias=bias)

    atol = 7e-2 if dtype == torch.bfloat16 and affine else 5e-2
    if dtype not in (torch.float16, torch.bfloat16):
        atol = 1e-4
    utils.gems_assert_close(
        y,
        ref_y,
        dtype,
        reduce_dim=_group_reduce_dim(shape, num_groups),
        atol=atol,
    )


@pytest.mark.group_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [False, True])
def test_accuracy_group_norm_mixed_values(dtype, affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 常规边界/混合尺寸测试
    shape = (4, 16, 7, 7)
    num_groups = 4
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    C = shape[1]
    weight, bias = None, None
    if affine:
        weight = torch.randn(C, dtype=dtype, device=flag_dnn.device)
        bias = torch.randn(C, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_bias = utils.to_reference(bias, ref_kind="compute")
    ref_y = F.group_norm(ref_x, num_groups, weight=ref_weight, bias=ref_bias)
    with flag_dnn.use_dnn():
        y = F.group_norm(x, num_groups, weight=weight, bias=bias)

    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_group_reduce_dim(shape, num_groups)
    )
