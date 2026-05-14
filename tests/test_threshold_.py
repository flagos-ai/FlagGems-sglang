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


THRESHOLD_CASES = [
    *[(shape, 0.0, 0.0) for shape in [(0,), *utils.POINTWISE_SHAPES]],
    # 1. 基础与经典形状
    ((1024,), 0.0, 0.0),  # 经典 ReLU 行为
    ((1024,), 1.5, 99.0),
    # 2. 非对齐的形状
    ((1023,), -1.0, 5.0),
    ((17, 31), 0.5, -2.5),
    ((13, 17, 19, 23), 0.0, 42.0),  # 4D 全质数形状，绝对的非 2 的幂次方
    # 3. 多维与高维张量
    ((4, 8, 16), 2.0, 0.0),
    ((2, 3, 4, 5), -0.5, 10.0),
    (
        (1, 1, 1, 1, 1),
        0.5,
        -0.5,
    ),  # 5D 张量，测试维度降维成 1D 时的正确性
    # 4. 大尺度张量 (Stress Testing)
    ((1024 * 1024,), 1.0, -1.0),  # 百万级一维元素，测试 Grid 拆分机制
    (
        (64, 128, 256),
        0.5,
        0.0,
    ),  # 大体积 3D 张量，测试大吞吐量
    # 5. 极端条件判断 (All or Nothing)
    ((64, 64), 100.0, -1.0),  # 阈值极大 (100.0)：几乎所有元素都会被替换
    (
        (64, 64),
        -100.0,
        -1.0,
    ),  # 阈值极小 (-100.0)：几乎没有任何元素会被替换
    # 6. 特殊替换值
    (
        (256, 256),
        0.0,
        float("-inf"),
    ),  # 替换为负无穷 (常用于 Attention Masking)
    ((256, 256), 0.0, float("inf")),  # 替换为正无穷
    # 7. 极端边界情况
    ((1,), 0.0, 1.0),  # 单元素
    (
        (0,),
        0.0,
        0.0,
    ),  # 空张量 (Empty Tensor)，测试 Kernel 是否正确 skip
]


@pytest.mark.threshold_
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, threshold_val, value_val", THRESHOLD_CASES)
def test_accuracy_threshold_(dtype, shape, threshold_val, value_val):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = F.threshold_(ref_x, threshold_val, value_val)

    with flag_dnn.use_dnn():
        out_custom = F.threshold_(x_custom, threshold_val, value_val)

    utils.gems_assert_close(out_custom, out_ref, dtype, equal_nan=True)

    assert out_custom.data_ptr() == x_custom.data_ptr(), (
        "output is not modifying " "the input tensor directly."
    )
    utils.gems_assert_close(x_custom, out_ref, dtype, equal_nan=True)
