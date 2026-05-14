import warnings

import pytest
import torch
import torch.nn.functional as F

import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


SOFTMIN_CASES = [
    ((1,), 0, None),
    ((17,), 0, None),
    ((2, 3), 0, None),
    ((2, 3), 1, None),
    ((4, 5, 6), 0, None),
    ((4, 5, 6), 1, None),
    ((4, 5, 6), 2, None),
    ((4, 5, 6), -1, None),
    ((2, 3, 32, 32), 1, None),
    ((2, 3, 32, 32), -1, None),
    ((8, 16), None, None),
    ((4, 5, 6), None, None),
    ((8, 16), 1, torch.float32),
    ((4, 5, 6), -1, torch.float32),
]
if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.ALL_FLOAT_DTYPES


def get_tol(dtype, out_dtype=None):
    target_dtype = out_dtype if out_dtype is not None else dtype
    if target_dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)
    if target_dtype == torch.bfloat16:
        return dict(rtol=1e-2, atol=1e-2)
    if target_dtype == torch.float32:
        return dict(rtol=1e-6, atol=1e-6)
    return dict(rtol=1e-12, atol=1e-12)


def _sum_atol(dtype, out_dtype=None):
    target_dtype = out_dtype if out_dtype is not None else dtype
    if target_dtype == torch.bfloat16:
        return 4e-3
    if target_dtype == torch.float16:
        return 6e-4
    return 1e-4


def softmin_no_dim_warning(x, dim=None, dtype=None):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Implicit dimension choice for softmin has been deprecated.*"
            ),
            category=UserWarning,
        )
        return F.softmin(x, dim=dim, dtype=dtype)


@pytest.mark.softmin
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("shape, dim, out_dtype", SOFTMIN_CASES)
def test_accuracy_softmin(dtype, shape, dim, out_dtype):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    if out_dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    x = torch.randn(shape, dtype=dtype, device=flag_dnn.device) * 5.0

    x_ref = x.clone()
    x_custom = x.clone()

    ref_x = utils.to_reference(x_ref, ref_kind="compute")
    out_ref = softmin_no_dim_warning(ref_x, dim=dim, dtype=out_dtype)

    with flag_dnn.use_dnn():
        out_custom = softmin_no_dim_warning(x_custom, dim=dim, dtype=out_dtype)

    target_dtype = out_dtype if out_dtype is not None else dtype
    reduce_dim = 1
    if dim is not None:
        dim_norm = dim if dim >= 0 else dim + len(shape)
        reduce_dim = max(shape[dim_norm], 1)
    utils.gems_assert_close(
        out_custom, out_ref, target_dtype, reduce_dim=reduce_dim
    )

    if x.numel() > 0:
        assert (
            out_custom.data_ptr() != x_custom.data_ptr()
        ), "softmin should be out-of-place, but output shares input memory."
        torch.testing.assert_close(x_custom, x, rtol=0, atol=0)

    if dim is not None and out_custom.numel() > 0:
        dim_norm = dim if dim >= 0 else dim + out_custom.dim()

        sums_custom = out_custom.float().sum(dim=dim_norm)
        sums_ref = out_ref.float().sum(dim=dim_norm)

        utils.gems_assert_close(
            sums_custom,
            sums_ref,
            torch.float32,
            reduce_dim=reduce_dim,
            atol=_sum_atol(dtype, out_dtype=out_dtype),
        )


@pytest.mark.softmin
def test_softmin_dim_none_matches_pytorch_default():
    x = torch.randn((4, 5, 6), dtype=torch.float32, device=flag_dnn.device)

    ref_x = utils.to_reference(x, ref_kind="compute")
    out_ref = softmin_no_dim_warning(ref_x, dim=None)

    with flag_dnn.use_dnn():
        out_custom = softmin_no_dim_warning(x, dim=None)

    utils.gems_assert_close(out_custom, out_ref, torch.float32)
