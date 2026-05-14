import pytest
import torch
import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


# (matrix_shape, use_out) 的组合测试用例
MV_CASES = [
    ((1, 1), False),
    ((4, 4), False),
    ((16, 32), False),
    ((32, 16), False),
    ((128, 64), False),
    ((128, 63), False),
    ((64, 128), False),
    ((63, 128), False),
    ((63, 127), False),
    ((2, 3), False),
    ((3, 2), False),
    ((0, 4), False),
    ((4, 0), False),
    ((1, 128), False),
    ((128, 1), False),
    ((1, 1), True),
    ((4, 4), True),
    ((16, 32), True),
    ((32, 16), True),
    ((128, 64), True),
    ((64, 128), True),
    ((2, 3), True),
    ((3, 2), True),
    ((0, 4), True),
    ((4, 0), True),
    ((1, 128), True),
    ((128, 1), True),
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


@pytest.mark.mv
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("matrix_shape, use_out", MV_CASES)
def test_accuracy_mv(dtype, matrix_shape, use_out):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    m, n = matrix_shape

    mat = torch.randn(matrix_shape, dtype=dtype, device=flag_dnn.device) * 5.0
    vec = torch.randn((n,), dtype=dtype, device=flag_dnn.device) * 5.0

    mat_ref = mat.clone()
    vec_ref = vec.clone()
    mat_custom = mat.clone()
    vec_custom = vec.clone()

    ref_mat = utils.to_reference(mat_ref, ref_kind="compute")
    ref_vec = utils.to_reference(vec_ref, ref_kind="compute")
    out_ref = torch.mv(ref_mat, ref_vec)

    if use_out:
        out_custom_buf = torch.empty((m,), dtype=dtype, device=flag_dnn.device)

        with flag_dnn.use_dnn():
            out_custom = torch.mv(mat_custom, vec_custom, out=out_custom_buf)

        utils.gems_assert_close(out_custom, out_ref, dtype, reduce_dim=n)

        assert out_custom.data_ptr() == out_custom_buf.data_ptr(), (
            "out is provided, but returned tensor does not share "
            "the output buffer memory."
        )
    else:
        with flag_dnn.use_dnn():
            out_custom = torch.mv(mat_custom, vec_custom)

        utils.gems_assert_close(out_custom, out_ref, dtype, reduce_dim=n)

        if out_custom.numel() > 0:
            assert (
                out_custom.data_ptr() != mat_custom.data_ptr()
            ), "Output unexpectedly shares memory with input matrix."
            assert (
                out_custom.data_ptr() != vec_custom.data_ptr()
            ), "Output unexpectedly shares memory with input vector."

    torch.testing.assert_close(mat_custom, mat, rtol=0, atol=0)
    torch.testing.assert_close(vec_custom, vec, rtol=0, atol=0)
