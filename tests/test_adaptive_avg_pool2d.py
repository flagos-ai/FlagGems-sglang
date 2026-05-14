import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# (shape, output_size)
PARAMS = [
    ((2, 3, 32, 32), (1, 1)),  # 全局平均池化 (Global Average Pooling)
    ((1, 16, 28, 28), 14),  # 降维到 14x14 (输入单整数)
    ((4, 8, 15, 15), (7, 5)),  # 非对称目标尺寸
    ((2, 4, 32, 32), (None, 16)),  # 保持 H 尺寸不变，W 降到 16
    ((16, 14, 14), (2, 2)),  # 3D 张量输入
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _as_pair(output_size):
    return (
        (output_size, output_size)
        if isinstance(output_size, int)
        else output_size
    )


def _adaptive_reduce_dim(shape, output_size):
    output_h, output_w = _as_pair(output_size)
    input_h, input_w = shape[-2], shape[-1]
    output_h = input_h if output_h is None else output_h
    output_w = input_w if output_w is None else output_w
    return max(
        ((input_h + output_h - 1) // output_h)
        * ((input_w + output_w - 1) // output_w),
        1,
    )


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, output_size", PARAMS)
def test_accuracy_adaptive_avg_pool2d(dtype, shape, output_size):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = F.adaptive_avg_pool2d(ref_x, output_size)
    with flag_dnn.use_dnn():
        y = F.adaptive_avg_pool2d(x, output_size)

    utils.gems_assert_close(
        y,
        ref_y,
        dtype,
        reduce_dim=_adaptive_reduce_dim(shape, output_size),
    )


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize(
    "dtype",
    [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float16],
)
def test_accuracy_adaptive_avg_pool2d_empty_tensor(dtype):
    shape = (0, 3, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = F.adaptive_avg_pool2d(ref_x, (2, 2))
    with flag_dnn.use_dnn():
        y = F.adaptive_avg_pool2d(x, (2, 2))

    assert y.shape == ref_y.shape
    assert y.numel() == 0
    utils.gems_assert_close(y, ref_y, dtype, reduce_dim=256)


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize(
    "dtype",
    (
        [torch.float32]
        if cfg.QUICK_MODE
        else [torch.float32, torch.float16, torch.bfloat16]
    ),
)
def test_accuracy_adaptive_avg_pool2d_large_values(dtype):
    shape = (2, 3, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 1000.0

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = F.adaptive_avg_pool2d(ref_x, (2, 2))
    with flag_dnn.use_dnn():
        y = F.adaptive_avg_pool2d(x, (2, 2))

    utils.gems_assert_close(y, ref_y, dtype, reduce_dim=256, atol=1e-1)


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize(
    "dtype",
    (
        [torch.float32]
        if cfg.QUICK_MODE
        else [torch.float32, torch.float16, torch.bfloat16]
    ),
)
def test_accuracy_adaptive_avg_pool2d_mixed_values(dtype):
    shape = (2, 3, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    x[..., ::2, ::2] *= 1000.0
    x[..., 1::2, 1::2] *= 0.001

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = F.adaptive_avg_pool2d(ref_x, (3, 3))
    with flag_dnn.use_dnn():
        y = F.adaptive_avg_pool2d(x, (3, 3))

    utils.gems_assert_close(y, ref_y, dtype, reduce_dim=121, atol=1e-1)
