import pytest
import torch
import torch.nn.functional as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# avg_pool1d 参数格式：(shape, kernel_size, stride, padding)
PARAMS = [
    ((2, 3, 32), 2, 2, 0),  # 标准 2 降采样
    ((1, 16, 28), 3, 1, 1),  # 保持原图尺寸 (Padding=1)
    ((4, 8, 15), 3, 2, 1),  # 奇数尺寸的步长跨越
    ((2, 4, 32), 5, 1, 0),  # 较大的一维卷积核测试
    ((2, 3, 10), 3, 1, 1),  # 强边缘补零测试
    ((16, 14), 2, 2, 0),  # 2D 张量输入 (无 Batch(N) 维度)
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _pool_reduce_dim(kernel_size):
    return kernel_size if isinstance(kernel_size, int) else kernel_size[0]


@pytest.mark.avg_pool1d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, kernel_size, stride, padding", PARAMS)
@pytest.mark.parametrize("ceil_mode", [False, True])
@pytest.mark.parametrize("count_include_pad", [False, True])
def test_accuracy_avg_pool1d(
    dtype, shape, kernel_size, stride, padding, ceil_mode, count_include_pad
):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 使用 randn 生成测试数据
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.avg_pool1d(
        ref_x,
        kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    with flag_dnn.use_dnn():
        out = F.avg_pool1d(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    utils.gems_assert_close(
        out, ref_out, dtype, reduce_dim=_pool_reduce_dim(kernel_size)
    )


@pytest.mark.avg_pool1d
@pytest.mark.parametrize(
    "dtype",
    [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float16],
)
def test_accuracy_avg_pool1d_empty_tensor(dtype):
    shape = (0, 3, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.avg_pool1d(ref_x, 2, 2)
    with flag_dnn.use_dnn():
        out = F.avg_pool1d(x, 2, 2)

    assert out.shape == ref_out.shape
    assert out.numel() == 0
    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=2)
