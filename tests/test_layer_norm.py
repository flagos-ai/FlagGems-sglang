import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


SHAPES_AND_NORM_SHAPES = utils.NORM_SHAPES


def _norm_reduce_dim(normalized_shape):
    return max(int(torch.tensor(normalized_shape).prod().item()), 1)


@pytest.mark.layer_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, normalized_shape", SHAPES_AND_NORM_SHAPES)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_layer_norm(
    dtype, shape, normalized_shape, elementwise_affine
):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    # 构建仿射变换参数
    weight, bias = None, None
    if elementwise_affine:
        weight = torch.randn(
            normalized_shape, dtype=dtype, device=flag_dnn.device
        )
        bias = torch.randn(
            normalized_shape, dtype=dtype, device=flag_dnn.device
        )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_bias = utils.to_reference(bias, ref_kind="compute")
    ref_y = F.layer_norm(
        ref_x, normalized_shape, weight=ref_weight, bias=ref_bias
    )
    with flag_dnn.use_dnn():
        y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)

    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_norm_reduce_dim(normalized_shape)
    )


@pytest.mark.layer_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_layer_norm_empty_tensor(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量
    shape = (0, 4, 16)
    normalized_shape = (16,)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    weight, bias = None, None
    if elementwise_affine:
        weight = torch.randn(
            normalized_shape, dtype=dtype, device=flag_dnn.device
        )
        bias = torch.randn(
            normalized_shape, dtype=dtype, device=flag_dnn.device
        )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_bias = utils.to_reference(bias, ref_kind="compute")
    ref_y = F.layer_norm(
        ref_x, normalized_shape, weight=ref_weight, bias=ref_bias
    )
    with flag_dnn.use_dnn():
        y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)

    assert y.shape == shape
    assert y.dtype == dtype
    assert y.device == x.device
    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_norm_reduce_dim(normalized_shape)
    )


@pytest.mark.layer_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_layer_norm_large_values(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (4, 8, 32)
    normalized_shape = (32,)

    # 专门测试大数值，验证方差计算时是否发生了严重的精度丢失
    # 由于 16 位浮点数的尾数限制，这里的大数值设计需要针对 dtype 区分
    if dtype in [torch.float16, torch.bfloat16]:
        x = (
            torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 10.0
            + 100.0
        )
    else:
        x = (
            torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 100.0
            + 1000.0
        )

    weight, bias = None, None
    if elementwise_affine:
        # 权重和偏置也需要适配大数值场景，避免乘加溢出
        if dtype in [torch.float16, torch.bfloat16]:
            weight = (
                torch.randn(
                    normalized_shape, dtype=dtype, device=flag_dnn.device
                )
                * 2.0
            )
            bias = (
                torch.randn(
                    normalized_shape, dtype=dtype, device=flag_dnn.device
                )
                * 10.0
            )
        else:
            weight = (
                torch.randn(
                    normalized_shape, dtype=dtype, device=flag_dnn.device
                )
                * 10.0
            )
            bias = (
                torch.randn(
                    normalized_shape, dtype=dtype, device=flag_dnn.device
                )
                * 100.0
            )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_bias = utils.to_reference(bias, ref_kind="compute")
    ref_y = F.layer_norm(
        ref_x, normalized_shape, weight=ref_weight, bias=ref_bias
    )
    with flag_dnn.use_dnn():
        y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)

    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 5e-4
    utils.gems_assert_close(
        y,
        ref_y,
        dtype,
        reduce_dim=_norm_reduce_dim(normalized_shape),
        atol=atol,
    )


@pytest.mark.layer_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_layer_norm_mixed_values(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 混合常规数据测试
    shape = (4, 8, 16)
    normalized_shape = (8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    weight, bias = None, None
    if elementwise_affine:
        weight = torch.randn(
            normalized_shape, dtype=dtype, device=flag_dnn.device
        )
        bias = torch.randn(
            normalized_shape, dtype=dtype, device=flag_dnn.device
        )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_bias = utils.to_reference(bias, ref_kind="compute")
    ref_y = F.layer_norm(
        ref_x, normalized_shape, weight=ref_weight, bias=ref_bias
    )
    with flag_dnn.use_dnn():
        y = F.layer_norm(x, normalized_shape, weight=weight, bias=bias)

    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_norm_reduce_dim(normalized_shape)
    )
