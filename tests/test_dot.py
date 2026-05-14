import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# (numel, use_out) 的组合测试用例
DOT_CASES = [
    # 空张量
    (0, False),
    (0, True),
    (1, False),
    (1, True),
    (2, False),
    (2, True),
    (64, False),
    (64, True),
    (65, False),
    (65, True),
    (127, False),
    (127, True),
    (128, False),
    (128, True),
    (255, True),
    (256, False),
    (256, True),
    (1023, False),
    (1023, True),
    (1024, False),
    (1024, True),
    (1000, False),
    (1000, True),
    (2048, False),
    (2048, True),
    (8191, False),
    (16384, False),
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


@pytest.mark.dot
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("numel, use_out", DOT_CASES)
def test_accuracy_dot(dtype, numel, use_out):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn((numel,), dtype=dtype, device=flag_dnn.device)
    y = torch.randn((numel,), dtype=dtype, device=flag_dnn.device)

    x_ref = x.clone()
    y_ref = y.clone()
    x_custom = x.clone()
    y_custom = y.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    ref_y = utils.to_reference(y_ref, ref_kind="compute")
    out_ref = torch.dot(ref_x, ref_y)

    if use_out:
        out_buf = torch.empty((), dtype=dtype, device=flag_dnn.device)
        with flag_dnn.use_dnn():
            out_custom = torch.dot(x_custom, y_custom, out=out_buf)

        # 返回值应当复用 out buffer
        assert (
            out_custom.data_ptr() == out_buf.data_ptr()
        ), "use_out=True, but returned tensor does not share memory with out."
    else:
        with flag_dnn.use_dnn():
            out_custom = torch.dot(x_custom, y_custom)

    # dot 返回标量张量
    assert out_custom.dim() == 0
    assert out_ref.dim() == 0

    utils.gems_assert_close(out_custom, out_ref, dtype, reduce_dim=numel)

    # dot 不应修改输入
    torch.testing.assert_close(x_custom, x, rtol=0, atol=0)
    torch.testing.assert_close(y_custom, y, rtol=0, atol=0)
