# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness test for quantization/triton_scaled_mm."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype, _DEFAULT_TOLERANCE
        )
    )
    tol.update(overrides)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        equal_nan=True,
        **tol,
    )


def _case(
    m, k, n, in_dtype, out_dtype=torch.float16, seed=0, *, scale_range=None
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    if in_dtype == torch.int8:
        a = torch.randint(
            -8,
            8,
            (m, k),
            dtype=in_dtype,
            device=flaggems_sglang.device,
            generator=g,
        )
        b = torch.randint(
            -8,
            8,
            (k, n),
            dtype=in_dtype,
            device=flaggems_sglang.device,
            generator=g,
        )
        rtol, atol = 0.15, 0.1
    else:
        a = torch.clamp(
            0.1
            * torch.randn(
                m,
                k,
                dtype=torch.float16,
                device=flaggems_sglang.device,
                generator=g,
            ),
            -0.3,
            0.3,
        ).to(in_dtype)
        b = torch.clamp(
            0.1
            * torch.randn(
                k,
                n,
                dtype=torch.float16,
                device=flaggems_sglang.device,
                generator=g,
            ),
            -0.3,
            0.3,
        ).to(in_dtype)
        rtol, atol = 0.25, 0.15
    if scale_range is None:
        scale_min, scale_width = 0.1, 0.05
    else:
        scale_min, scale_max = scale_range
        scale_width = scale_max - scale_min
    scale_a = scale_min + scale_width * torch.rand(
        m, 1, dtype=torch.float32, device=flaggems_sglang.device, generator=g
    )
    scale_b = scale_min + scale_width * torch.rand(
        n, 1, dtype=torch.float32, device=flaggems_sglang.device, generator=g
    )
    return dict(
        input=a,
        weight=b,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
        check=_checker(rtol, atol),
    )


def _checker(rtol, atol):
    def _check(actual, expected):
        assert_close(actual, expected, atol=atol, rtol=rtol)

    return _check


_CORRECTNESS_CASES = [
    lambda: _case(32, 32, 32, torch.int8),
    lambda: _case(17, 64, 96, torch.int8),
    lambda: _case(32, 32, 32, torch.float8_e4m3fn),
    lambda: _case(17, 64, 96, torch.float8_e4m3fn),
]
# Exercise FP8 scaling with outputs above the absolute tolerance, including
# a single row and a non-aligned row count. Benchmark inputs stay unchanged.
_NORMAL_SCALE_CASES = [
    lambda: _case(
        1, 4096, 256, torch.float8_e4m3fn, seed=11, scale_range=(0.5, 1.5)
    ),
    lambda: _case(
        17, 1024, 96, torch.float8_e4m3fn, seed=12, scale_range=(0.5, 1.5)
    ),
    lambda: _case(
        64, 4096, 256, torch.float8_e4m3fn, seed=13, scale_range=(0.5, 1.5)
    ),
]


def _build_normal_scale(case):
    """A single row has just one activation scale: keep it away from one."""
    c = case()
    c["scale_a"].fill_(0.5)
    return c


_CORRECTNESS_CASES += [lambda: _build_normal_scale(_NORMAL_SCALE_CASES[0])]
_CORRECTNESS_CASES += [
    (lambda m=m, k=k, n=n: _case(m, k, n, torch.float8_e4m3fn))
    for m in (1, 8, 64, 512, 4096)
    for k, n in ((4096, 4096), (4096, 7168))
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.triton_scaled_mm
def test_triton_scaled_mm(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("triton_scaled_mm")(**kwargs)

    try:
        from flaggems_sglang.ops.triton_scaled_mm import triton_scaled_mm
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/triton_scaled_mm ops module not found")
        return

    try:
        actual = triton_scaled_mm(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/triton_scaled_mm not yet implemented")
        return

    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    if check is not None:
        case["check"] = check


@pytest.mark.parametrize("case_idx", range(len(_NORMAL_SCALE_CASES)))
def test_triton_scaled_mm_rejects_incorrect_scaling(case_idx):
    case = _NORMAL_SCALE_CASES[case_idx]()
    if callable(case):
        case = case()
    # Pin both scales to 0.5 so swapping either for ones (1.0) is a
    # deterministic 2x change. The reference is exact float32 math on shared
    # fp8 inputs, so a strict checker reliably detects the scale swap that
    # the fp8 quantization tolerance would otherwise absorb.
    case["scale_a"].fill_(0.5)
    case["scale_b"].fill_(0.5)
    kwargs = {key: value for key, value in case.items() if key != "check"}
    expected = get_reference("triton_scaled_mm")(**kwargs)
    check = _checker(1e-5, 1e-5)
    for actual in (torch.zeros_like(expected), expected * 0.5):
        with pytest.raises(AssertionError):
            check(actual, expected)
    for missing_scales in [("scale_a",), ("scale_b",), ("scale_a", "scale_b")]:
        broken_kwargs = dict(kwargs)
        for key in missing_scales:
            broken_kwargs[key] = torch.ones_like(kwargs[key])
        actual = get_reference("triton_scaled_mm")(**broken_kwargs)
        with pytest.raises(AssertionError):
            check(actual, expected)
