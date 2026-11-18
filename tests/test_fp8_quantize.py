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

"""Correctness test for quantization/fp8_quantize."""

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


def _x(shape, dtype=torch.bfloat16, seed=0, scale=1.0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    return (
        scale
        * torch.randn(
            *shape,
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.float32,
        )
    ).to(dtype)


def _sliced(rows, keep, drop, dtype=torch.bfloat16, seed=1):
    base = _x((rows, keep + drop), dtype=dtype, seed=seed)
    return base[:, drop:]


def _check(actual, expected):
    assert_close(actual.view(torch.uint8), expected.view(torch.uint8))


def _case(x, scale_inv=1.0, fp8_dtype=torch.float8_e4m3fn):
    return dict(x=x, scale_inv=scale_inv, fp8_dtype=fp8_dtype, check=_check)


_CORRECTNESS_CASES = [
    lambda: _case(_x((7, 128))),
    lambda: _case(_x((83, 512)), scale_inv=0.125),
    lambda: _case(_x((256, 4096), scale=8.0), scale_inv=1.0 / 8.0),
    lambda: _case(_x((4, 8, 512))),
    lambda: _case(_sliced(64, 512, 64)),
    lambda: _case(_x((128, 1024)), fp8_dtype=torch.float8_e5m2),
]
_CORRECTNESS_CASES += [
    (lambda m=m, n=n: _case(_x((m, n))))
    for m in (1, 8, 64, 512, 4096)
    for n in (1024, 4096, 8192)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.fp8_quantize
def test_fp8_quantize(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("fp8_quantize")(**kwargs)

    try:
        from flaggems_sglang.ops.fp8_quantize import fp8_quantize
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/fp8_quantize ops module not found")
        return

    try:
        actual = fp8_quantize(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/fp8_quantize not yet implemented")
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
