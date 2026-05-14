import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    NON_FLOAT_DTYPES = [torch.bool, torch.int32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES
    NON_FLOAT_DTYPES = utils.BOOL_TYPES + utils.ALL_INT_DTYPES


SHAPES = utils.POINTWISE_SHAPES

BROADCAST_SHAPES = [
    ((4, 4), (4,)),  # 1D broadcast to 2D
    ((2, 3, 4), (3, 1)),  # 内部维度广播
    ((1, 5), (5, 5)),  # 单一维度扩展
    ((2, 1, 4, 1), (1, 3, 1, 5)),  # 复杂高维双向广播
    ((), (17, 31)),  # 标量 Tensor 广播到矩阵
]


def _get_safe_divisor(shape, dtype, device):
    """
    生成安全的除数张量，避免随机出极小的值（如 1e-4）导致除法结果过大，
    从而破坏 FP16/BF16 的相对/绝对容差验证。
    """
    y = torch.randn(shape, dtype=dtype, device=device)
    # 将绝对值小于 0.1 的元素强制设为 1.0 (保留原本的符号会更好，这里为简单稳定直接赋固定值或带符号偏移)
    y[y.abs() < 0.1] = 1.0
    return y


@pytest.mark.div
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_div(dtype, shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    y = _get_safe_divisor(shape, dtype, flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.div(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.div(x, y)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.div
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_empty_tensor(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 测试空张量 (shape 为 0)
    x = torch.randn(0, dtype=dtype, device=flag_dnn.device)
    y = torch.randn(0, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind=None)
    ref_y = utils.to_reference(y, ref_kind=None)

    ref_out = torch.div(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.div(x, y)

    assert out.shape == (0,)
    assert out.dtype == dtype
    assert out.device == x.device
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.div
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_div_scalar(dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(100, dtype=dtype, device=flag_dnn.device)
    scalar = 3.14  # 使用足够大的固定安全除数

    ref_x = utils.to_reference(x, ref_kind="compute")

    ref_out = torch.div(ref_x, scalar)
    with flag_dnn.use_dnn():
        out = torch.div(x, scalar)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.div
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("rounding_mode", [None, "trunc", "floor"])
def test_accuracy_div_rounding_mode(dtype, rounding_mode):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    numel = 100

    if rounding_mode in ("trunc", "floor"):
        xs = []
        ys = []

        # trunc / floor 对整数边界非常敏感，尤其是 fp16 / bf16。
        # 这里过滤掉 x / y 接近整数的 case，避免参考 FP64 和设备低精度计算差 1。
        while sum(t.numel() for t in xs) < numel:
            cand_x = torch.randn(
                numel * 4, dtype=dtype, device=flag_dnn.device
            )
            cand_y = _get_safe_divisor((numel * 4,), dtype, flag_dnn.device)

            ref_cand_x = utils.to_reference(cand_x, ref_kind="compute")
            ref_cand_y = utils.to_reference(cand_y, ref_kind="compute")

            q = ref_cand_x / ref_cand_y

            # 距离最近整数太近的样本容易因为精度差异导致 trunc/floor 差 1
            boundary_eps = 1e-2
            valid_mask = torch.abs(q - torch.round(q)) > boundary_eps

            valid_idx = valid_mask.nonzero(as_tuple=True)[0]

            if valid_idx.numel() == 0:
                continue

            valid_idx = valid_idx.to(cand_x.device)
            xs.append(cand_x[valid_idx])
            ys.append(cand_y[valid_idx])

        x = torch.cat(xs)[:numel]
        y = torch.cat(ys)[:numel]

    else:
        x = torch.randn(numel, dtype=dtype, device=flag_dnn.device)
        y = _get_safe_divisor((numel,), dtype, flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.div(ref_x, ref_y, rounding_mode=rounding_mode)
    with flag_dnn.use_dnn():
        out = torch.div(x, y, rounding_mode=rounding_mode)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.div
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("input_shape, other_shape", BROADCAST_SHAPES)
def test_accuracy_div_broadcast(dtype, input_shape, other_shape):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(input_shape, dtype=dtype, device=flag_dnn.device)
    y = _get_safe_divisor(other_shape, dtype, flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.div(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.div(x, y)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.div
def test_accuracy_div_integer_dtype():
    x = torch.tensor([2, 4, 8], dtype=torch.int32, device=flag_dnn.device)
    y = torch.tensor([2, 2, 4], dtype=torch.int32, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.div(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.div(x, y)

    assert out.dtype == torch.float32
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.div
@pytest.mark.parametrize("dtype", NON_FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_accuracy_div_non_float_dtype(dtype, shape):
    if dtype == torch.bool:
        x = torch.randint(0, 2, shape, dtype=dtype, device=flag_dnn.device)
        y = torch.ones(shape, dtype=dtype, device=flag_dnn.device)
    else:
        y = torch.randint(1, 5, shape, dtype=dtype, device=flag_dnn.device)
        scale = torch.randint(
            -4, 5, shape, dtype=dtype, device=flag_dnn.device
        )
        x = y * scale

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_y = utils.to_reference(y, ref_kind="compute")

    ref_out = torch.div(ref_x, ref_y)
    with flag_dnn.use_dnn():
        out = torch.div(x, y)

    assert out.dtype == torch.float32
    utils.gems_assert_equal(out, ref_out)
