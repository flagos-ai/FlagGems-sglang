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


@pytest.mark.rms_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, normalized_shape", SHAPES_AND_NORM_SHAPES)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_rms_norm(dtype, shape, normalized_shape, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    weight = None
    if elementwise_affine:
        weight = torch.randn(
            normalized_shape, dtype=dtype, device=flag_dnn.device
        )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_y = F.rms_norm(ref_x, normalized_shape, weight=ref_weight)

    # 自定义算子调用
    with flag_dnn.use_dnn():
        y = F.rms_norm(x, normalized_shape, weight=weight)

    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_norm_reduce_dim(normalized_shape)
    )


@pytest.mark.rms_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_rms_norm_empty_tensor(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (0, 4, 16)
    normalized_shape = (16,)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    weight = None
    if elementwise_affine:
        weight = torch.randn(
            normalized_shape, dtype=dtype, device=flag_dnn.device
        )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_y = F.rms_norm(ref_x, normalized_shape, weight=ref_weight)
    with flag_dnn.use_dnn():
        y = F.rms_norm(x, normalized_shape, weight=weight)

    assert y.shape == shape
    assert y.dtype == dtype
    assert y.device == x.device
    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_norm_reduce_dim(normalized_shape)
    )


@pytest.mark.rms_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_rms_norm_large_values(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    shape = (4, 8, 32)
    normalized_shape = (32,)

    # 相比于 LayerNorm 的均值平移，RMSNorm 对大数值的平方和极度敏感
    # 在 FP16/BF16 下更容易溢出，因此需严控数据范围
    if dtype in [torch.float16, torch.bfloat16]:
        x = (
            torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0
            + 20.0
        )
    else:
        x = (
            torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 100.0
            + 1000.0
        )

    weight = None
    if elementwise_affine:
        if dtype in [torch.float16, torch.bfloat16]:
            weight = (
                torch.randn(
                    normalized_shape, dtype=dtype, device=flag_dnn.device
                )
                * 2.0
            )
        else:
            weight = (
                torch.randn(
                    normalized_shape, dtype=dtype, device=flag_dnn.device
                )
                * 10.0
            )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_y = F.rms_norm(ref_x, normalized_shape, weight=ref_weight)
    with flag_dnn.use_dnn():
        y = F.rms_norm(x, normalized_shape, weight=weight)

    atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    utils.gems_assert_close(
        y,
        ref_y,
        dtype,
        reduce_dim=_norm_reduce_dim(normalized_shape),
        atol=atol,
    )


@pytest.mark.rms_norm
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("elementwise_affine", [False, True])
def test_accuracy_rms_norm_mixed_values(dtype, elementwise_affine):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 混合常规数据测试，并测试多维度的 normalized_shape
    shape = (4, 8, 16)
    normalized_shape = (8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    weight = None
    if elementwise_affine:
        weight = torch.randn(
            normalized_shape, dtype=dtype, device=flag_dnn.device
        )

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_weight = utils.to_reference(weight, ref_kind="compute")
    ref_y = F.rms_norm(ref_x, normalized_shape, weight=ref_weight)
    with flag_dnn.use_dnn():
        y = F.rms_norm(x, normalized_shape, weight=weight)

    utils.gems_assert_close(
        y, ref_y, dtype, reduce_dim=_norm_reduce_dim(normalized_shape)
    )
