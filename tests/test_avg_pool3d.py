import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# 3D 参数格式：(shape, kernel_size, stride, padding)
PARAMS = [
    ((2, 3, 8, 32, 32), 2, 2, 0),  # 标准 2x2x2 降采样
    ((1, 8, 4, 16, 16), 3, 1, 1),  # 保持原图尺寸 (Padding=1)
    ((2, 4, 5, 15, 15), 3, 2, 1),  # 奇数尺寸的步长跨越
    ((1, 2, 8, 16, 16), (2, 3, 5), (1, 2, 1), 0),  # 不对称的 Kernel 和 Stride
    ((2, 3, 6, 10, 10), 3, 1, 1),  # 强边缘补零测试
    ((4, 5, 14, 14), 2, 2, 0),  # 4D 张量输入 (无 Batch 维度 N)
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _pool_reduce_dim(kernel_size):
    if isinstance(kernel_size, int):
        return kernel_size**3
    return kernel_size[0] * kernel_size[1] * kernel_size[2]


@pytest.mark.avg_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, kernel_size, stride, padding", PARAMS)
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [False, True])
@pytest.mark.parametrize("divisor_override", [None, 42])  # 验证不传和强行传参
def test_accuracy_avg_pool3d(
    dtype,
    shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 生成测试数据
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.avg_pool3d(
        ref_x,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    with flag_dnn.use_dnn():
        out = F.avg_pool3d(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )

    reduce_dim = divisor_override or _pool_reduce_dim(kernel_size)
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize(
    "dtype",
    [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float16],
)
def test_accuracy_avg_pool3d_empty_tensor(dtype):
    # D, H, W 至少一个维度的尺寸导致输出 M=0 的情况
    shape = (0, 3, 4, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.avg_pool3d(ref_x, 2, 2)
    with flag_dnn.use_dnn():
        out = F.avg_pool3d(x, 2, 2)

    assert out.shape == ref_out.shape
    assert out.numel() == 0
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=8)
