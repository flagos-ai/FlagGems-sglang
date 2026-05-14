import pytest
import torch

import flag_dnn
from . import accuracy_utils as utils
from . import conftest as cfg


MM_CASES = [
    (0, 3, 4),
    (2, 0, 4),
    (2, 3, 0),
    (2, 3, 4),
    (7, 5, 3),
    (16, 17, 15),
    (32, 64, 16),
]
if cfg.QUICK_MODE:
    MM_DTYPES = [torch.float32]
else:
    MM_DTYPES = [*utils.ALL_FLOAT_DTYPES, torch.complex64]


def _assert_mm_close(res, ref, dtype, reduce_dim=1, equal_nan=False):
    utils.gems_assert_close(
        res, ref, dtype, reduce_dim=reduce_dim, equal_nan=equal_nan
    )


def make_tensor(shape, dtype):
    if dtype in (torch.complex64, torch.complex128):
        real_dtype = (
            torch.float32 if dtype == torch.complex64 else torch.float64
        )
        real = torch.empty(shape, dtype=real_dtype, device=flag_dnn.device)
        imag = torch.empty(shape, dtype=real_dtype, device=flag_dnn.device)
        real.uniform_(-1.0, 1.0)
        imag.uniform_(-1.0, 1.0)
        return torch.complex(real, imag)

    return torch.empty(shape, dtype=dtype, device=flag_dnn.device).uniform_(
        -1.0, 1.0
    )


@pytest.mark.mm
@pytest.mark.parametrize("dtype", MM_DTYPES)
@pytest.mark.parametrize("m, k, n", MM_CASES)
@pytest.mark.parametrize("use_out", [False, True])
def test_accuracy_mm(dtype, m, k, n, use_out):
    if dtype == torch.float64 and not flag_dnn.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")

    a = make_tensor((m, k), dtype)
    b = make_tensor((k, n), dtype)
    a_ref = a.clone()
    b_ref = b.clone()
    a_custom = a.clone()
    b_custom = b.clone()

    ref_a = utils.to_reference(a_ref, ref_kind="compute")
    ref_b = utils.to_reference(b_ref, ref_kind="compute")
    out_ref = torch.mm(ref_a, ref_b)

    if use_out:
        out_buf = torch.empty((m, n), dtype=dtype, device=flag_dnn.device)
        with flag_dnn.use_dnn():
            out_custom = torch.mm(a_custom, b_custom, out=out_buf)

        assert out_custom.data_ptr() == out_buf.data_ptr()
        _assert_mm_close(out_buf, out_ref, dtype, reduce_dim=k)
    else:
        with flag_dnn.use_dnn():
            out_custom = torch.mm(a_custom, b_custom)

    _assert_mm_close(out_custom, out_ref, dtype, reduce_dim=k)
    torch.testing.assert_close(a_custom, a, rtol=0, atol=0)
    torch.testing.assert_close(b_custom, b, rtol=0, atol=0)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_out_dtype_fp32(dtype):
    a = make_tensor((8, 9), dtype)
    b = make_tensor((9, 7), dtype)
    out = torch.empty((8, 7), dtype=torch.float32, device=flag_dnn.device)

    ref_a = utils.to_reference(a, ref_kind="compute")
    ref_b = utils.to_reference(b, ref_kind="compute")
    ref = torch.mm(ref_a, ref_b).to(torch.float32)
    got = flag_dnn.ops.mm(a, b, out_dtype=torch.float32, out=out)

    assert got.data_ptr() == out.data_ptr()
    assert got.dtype == torch.float32
    utils.gems_assert_close(got, ref, torch.float32, reduce_dim=9)


@pytest.mark.mm
def test_mm_non_contiguous_inputs_and_out():
    dtype = torch.float32
    a = torch.randn((5, 7), dtype=dtype, device=flag_dnn.device).t()
    b = torch.randn((3, 5), dtype=dtype, device=flag_dnn.device).t()
    out_base = torch.empty((3, 7), dtype=dtype, device=flag_dnn.device)
    out = out_base.t()

    ref_a = utils.to_reference(a, ref_kind="compute")
    ref_b = utils.to_reference(b, ref_kind="compute")
    ref = torch.mm(ref_a, ref_b)
    with flag_dnn.use_dnn():
        got = torch.mm(a, b, out=out)

    assert got.data_ptr() == out.data_ptr()
    assert not out.is_contiguous()
    utils.gems_assert_close(got, ref, dtype, reduce_dim=5)


@pytest.mark.mm
def test_mm_nan_inf_equal_nan():
    a = torch.tensor(
        [[float("nan"), 1.0], [float("inf"), -0.0]],
        dtype=torch.float32,
        device=flag_dnn.device,
    )
    b = torch.tensor(
        [[2.0, -1.0], [3.0, float("inf")]],
        dtype=torch.float32,
        device=flag_dnn.device,
    )

    ref_a = utils.to_reference(a, ref_kind="compute")
    ref_b = utils.to_reference(b, ref_kind="compute")
    ref = torch.mm(ref_a, ref_b)
    with flag_dnn.use_dnn():
        got = torch.mm(a, b)

    utils.gems_assert_close(
        got, ref, torch.float32, reduce_dim=2, equal_nan=True
    )


@pytest.mark.mm
def test_mm_invalid_inputs():
    a = torch.randn((2, 3), device=flag_dnn.device)
    b = torch.randn((2, 4), device=flag_dnn.device)
    with flag_dnn.use_dnn():
        with pytest.raises(RuntimeError):
            torch.mm(a, b)

    a3 = torch.randn((1, 2, 3), device=flag_dnn.device)
    with flag_dnn.use_dnn():
        with pytest.raises(RuntimeError):
            torch.mm(a3, torch.randn((3, 4), device=flag_dnn.device))

    int_a = torch.randint(0, 4, (2, 3), device=flag_dnn.device)
    int_b = torch.randint(0, 4, (3, 4), device=flag_dnn.device)
    with flag_dnn.use_dnn():
        with pytest.raises(NotImplementedError):
            torch.mm(int_a, int_b)


@pytest.mark.mm
def test_mm_mismatched_dtype():
    a = torch.randn((2, 3), dtype=torch.float16, device=flag_dnn.device)
    b = torch.randn((3, 4), dtype=torch.float32, device=flag_dnn.device)
    with flag_dnn.use_dnn():
        with pytest.raises(RuntimeError):
            torch.mm(a, b)


@pytest.mark.mm
@pytest.mark.parametrize("m, k, n", [(1, 1, 1), (16, 16, 1)])
def test_mm_iluvatar_small_shape_direct_call(m, k, n):
    if flag_dnn.runtime.device.vendor_name != "iluvatar":
        pytest.skip("Iluvatar-specific regression test")

    dtype = torch.float16
    a = make_tensor((m, k), dtype)
    b = make_tensor((k, n), dtype)

    ref_a = utils.to_reference(a, ref_kind="compute")
    ref_b = utils.to_reference(b, ref_kind="compute")
    ref = torch.mm(ref_a, ref_b)
    got = flag_dnn.ops.mm(a, b)

    _assert_mm_close(got, ref, dtype, reduce_dim=k)


@pytest.mark.mm
def test_mm_iluvatar_small_shape_out_dtype_fp32():
    if flag_dnn.runtime.device.vendor_name != "iluvatar":
        pytest.skip("Iluvatar-specific regression test")

    a = make_tensor((16, 16), torch.float16)
    b = make_tensor((16, 1), torch.float16)
    out = torch.empty((16, 1), dtype=torch.float32, device=flag_dnn.device)

    ref_a = utils.to_reference(a, ref_kind="compute")
    ref_b = utils.to_reference(b, ref_kind="compute")
    ref = torch.mm(ref_a, ref_b).to(torch.float32)
    got = flag_dnn.ops.mm(a, b, out_dtype=torch.float32, out=out)

    assert got.data_ptr() == out.data_ptr()
    assert got.dtype == torch.float32
    utils.gems_assert_close(got, ref, torch.float32, reduce_dim=16)


@pytest.mark.mm
@pytest.mark.parametrize("m, k, n", [(256, 4, 256), (256, 16, 256)])
def test_mm_iluvatar_fp32_skinny_k_large_output(m, k, n):
    if flag_dnn.runtime.device.vendor_name != "iluvatar":
        pytest.skip("Iluvatar-specific regression test")

    dtype = torch.float32
    a = make_tensor((m, k), dtype)
    b = make_tensor((k, n), dtype)

    ref_a = utils.to_reference(a, ref_kind="compute")
    ref_b = utils.to_reference(b, ref_kind="compute")
    ref = torch.mm(ref_a, ref_b)
    got = flag_dnn.ops.mm(a, b)

    _assert_mm_close(got, ref, dtype, reduce_dim=k)
