import pytest
import torch
import torch._C._nn as F
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# adaptive_max_pool3d 参数格式：(shape, output_size)
PARAMS = [
    ((2, 3, 8, 16, 16), (4, 8, 8)),  # 标准 3D 降采样
    ((1, 8, 5, 14, 14), 7),  # 输出尺寸为单 int
    ((2, 4, 7, 15, 15), (3, 5, 7)),  # 三个维度不同尺寸
    ((1, 2, 4, 8, 8), (5, 10, 10)),  # 上采样 (输出尺寸大于输入)
    ((4, 5, 10, 20, 20), 1),  # 3D 全局池化 (Global Max Pooling)
    ((3, 8, 14, 14), (4, 7, 7)),  # 4D 张量输入 (无 Batch 维度 N)
    ((1, 2, 8, 8, 8), (8, 8, 8)),  # output == input (原样输出)
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def _gather_from_adaptive_max_pool3d_indices(
    x: torch.Tensor, indices: torch.Tensor
):
    """
    根据 adaptive_max_pool3d 返回的 indices，从输入 x 中取回对应值。
    indices 是相对于每个 (N, C) 或每个 C 的 D*H*W 展平索引。
    """
    if x.ndim == 5:
        # x: (N, C, D, H, W)
        n, c, _, _, _ = x.shape
        x_flat = x.reshape(n, c, -1)
        idx_flat = indices.reshape(n, c, -1).long()
        gathered = torch.gather(x_flat, 2, idx_flat)
        return gathered.reshape_as(indices)

    if x.ndim == 4:
        # x: (C, D, H, W)
        c, _, _, _ = x.shape
        x_flat = x.reshape(c, -1)
        idx_flat = indices.reshape(c, -1).long()
        gathered = torch.gather(x_flat, 1, idx_flat)
        return gathered.reshape_as(indices)

    raise AssertionError(f"Unsupported input ndim={x.ndim}, expected 4 or 5")


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, output_size", PARAMS)
def test_accuracy_adaptive_max_pool3d(dtype, shape, output_size):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    torch.manual_seed(0)

    # 随机输入下，低精度 dtype 可能出现并列最大值，
    # 此时 values 正确，但 indices 不一定和 PyTorch 完全一致。
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_vals, ref_indices = F.adaptive_max_pool3d(ref_x, output_size)

    with flag_dnn.use_dnn():
        out_vals, out_indices = F.adaptive_max_pool3d(x, output_size)

    # Max Pool 只是选取输入元素，values 应与参考结果一致
    utils.gems_assert_close(out_vals, ref_vals, dtype, atol=0)

    # 不直接要求随机输入下的 indices 与 PyTorch 完全一致；
    # 只要求这些 indices 能从输入中取回正确的输出值
    gathered_vals = _gather_from_adaptive_max_pool3d_indices(x, out_indices)
    ref_out_vals = utils.to_reference(out_vals, ref_kind=None)
    utils.gems_assert_close(gathered_vals, ref_out_vals, dtype, atol=0)

    # 对高精度随机输入，通常不会出现 tie，可额外严格比对索引
    if dtype in [torch.float32, torch.float64]:
        utils.gems_assert_equal(out_indices, ref_indices)


@pytest.mark.adaptive_max_pool3d
def test_accuracy_adaptive_max_pool3d_indices_unique_max():
    """
    单独构造“唯一最大值”的输入，严格验证 indices 与 PyTorch 完全一致。
    这里用 float32，避免低精度量化引入并列值。
    """
    shape = (2, 3, 4, 5, 6)
    output_size = (2, 3, 4)

    numel = 1
    for s in shape:
        numel *= s

    x = torch.arange(
        numel, dtype=torch.float32, device=flag_dnn.device
    ).reshape(shape)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_vals, ref_indices = F.adaptive_max_pool3d(ref_x, output_size)

    with flag_dnn.use_dnn():
        out_vals, out_indices = F.adaptive_max_pool3d(x, output_size)

    utils.gems_assert_close(out_vals, ref_vals, torch.float32, atol=0)
    utils.gems_assert_equal(out_indices, ref_indices)


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize(
    "dtype",
    [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float16],
)
def test_accuracy_adaptive_max_pool3d_empty_tensor(dtype):
    # D, H, W 至少一个维度的尺寸导致输出 M=0 的情况
    shape = (0, 3, 4, 32, 32)
    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    ref_out = F.adaptive_max_pool3d(ref_x, 2)
    with flag_dnn.use_dnn():
        out = F.adaptive_max_pool3d(x, 2)

    out_vals, out_indices = out
    ref_vals, ref_indices = ref_out
    assert out_vals.shape == ref_vals.shape
    assert out_indices.shape == ref_indices.shape
    assert out_vals.numel() == 0
    assert out_indices.numel() == 0
    utils.gems_assert_close(out_vals, ref_vals, dtype, atol=0)
    utils.gems_assert_equal(out_indices, ref_indices)
