import pytest
import torch
import flag_dnn

CUMMAX_CASES = [
    # (shape, dim)
    ((1024,), 0),  # 1D 长向量
    ((64, 1024), 1),  # 2D 扫最后一维（K==1）
    ((64, 1024), 0),  # 2D 扫首维（K>1 外扫描）
    ((32, 128, 64), 1),  # 3D 扫中间维
    ((32, 128, 64), 2),  # 3D 扫最后维
    ((4, 16, 32, 32), 2),  # 4D
    ((4, 16, 32, 32), -1),  # 负 dim
    ((2, 8, 16, 16, 16), 3),  # 5D
    ((8, 1, 16), 1),  # N==1 快速路径
    ((2, 5000), 1),  # N 大到跨多个 BLOCK_N
    ((2, 1048576), 1),  # 大 N 边界测试
]

DTYPES = [
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]


@pytest.mark.cummax
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape, dim", CUMMAX_CASES)
def test_accuracy_cummax(dtype, shape, dim):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    # 生成数据
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device=flag_dnn.device)
    elif dtype == torch.bool:
        x = torch.randint(
            0, 2, shape, dtype=torch.bool, device=flag_dnn.device
        )
    else:
        # 整型：取 [-100, 100] 避免极端值，uint8 会自动转化为 [0, 100]
        x = torch.randint(-100, 100, shape, device=flag_dnn.device).to(dtype)

    # 获取 PyTorch 参考结果（如果原生 CUDA 不支持该 dtype 则跳过）
    try:
        ref_v, ref_i = torch.cummax(x, dim=dim)
    except RuntimeError as e:
        pytest.skip(f"torch.cummax does not support {dtype} on CUDA: {e}")

    with flag_dnn.use_dnn():
        out_v, out_i = torch.cummax(x, dim=dim)

    assert out_v.dtype == dtype
    assert out_i.dtype == torch.int64
    assert out_v.shape == ref_v.shape

    if dtype.is_floating_point:
        rtol, atol = 1e-5, 1e-5
        if dtype == torch.float16:
            rtol, atol = 1e-3, 1e-3
        elif dtype == torch.bfloat16:
            rtol, atol = 1.6e-2, 1e-2

        torch.testing.assert_close(
            out_v, ref_v, rtol=rtol, atol=atol, equal_nan=True
        )
        # 浮点 Tie-break 可能不同，验证 indices 提取的值是否等价
        gathered = torch.gather(x, dim, out_i)
        torch.testing.assert_close(
            gathered, out_v, rtol=0, atol=0, equal_nan=True
        )
    else:
        # 整型和布尔型要求完全一致
        torch.testing.assert_close(out_v, ref_v, rtol=0, atol=0)
        torch.testing.assert_close(out_i, ref_i, rtol=0, atol=0)


@pytest.mark.cummax
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_accuracy_cummax_out_param(dtype):
    shape, dim = (4, 128, 16), 1
    x = torch.randn(shape, device=flag_dnn.device).to(dtype)
    out_v = torch.empty((1,), dtype=dtype, device=flag_dnn.device)
    out_i = torch.empty((1,), dtype=torch.int64, device=flag_dnn.device)

    ref_v, ref_i = torch.cummax(x, dim=dim)
    with flag_dnn.use_dnn():
        ret = torch.cummax(x, dim=dim, out=(out_v, out_i))

    assert ret[0].data_ptr() == out_v.data_ptr()
    torch.testing.assert_close(out_v, ref_v)
    torch.testing.assert_close(out_i, ref_i)


@pytest.mark.cummax
def test_accuracy_cummax_edge_cases():
    """集中测试 NaN、Inf、空张量、非连续张量等边界情况"""
    dim = 1

    # 1. NaN & Inf (fp32)
    x_nan = torch.randn((4, 32), dtype=torch.float32, device=flag_dnn.device)
    x_nan[0, 5], x_nan[1, 0] = float("nan"), float("nan")
    x_nan[2, 10], x_nan[3, 7] = float("inf"), float("-inf")
    ref_v, ref_i = torch.cummax(x_nan, dim=dim)
    with flag_dnn.use_dnn():
        out_v, out_i = torch.cummax(x_nan, dim=dim)
    torch.testing.assert_close(out_v, ref_v, equal_nan=True)

    # 2. 空张量
    x_empty = torch.empty(
        (4, 0, 8), dtype=torch.float32, device=flag_dnn.device
    )
    ref_v, ref_i = torch.cummax(x_empty, dim=dim)
    with flag_dnn.use_dnn():
        out_v, out_i = torch.cummax(x_empty, dim=dim)
    torch.testing.assert_close(out_v, ref_v)

    # 3. 非连续张量
    x_non_contig = torch.randn((4, 32, 16), device=flag_dnn.device).transpose(
        1, 2
    )
    ref_v, ref_i = torch.cummax(x_non_contig, dim=dim)
    with flag_dnn.use_dnn():
        out_v, out_i = torch.cummax(x_non_contig, dim=dim)
    torch.testing.assert_close(out_v, ref_v)
