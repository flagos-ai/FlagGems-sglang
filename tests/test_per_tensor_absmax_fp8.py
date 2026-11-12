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

"""Correctness test for quantization/per_tensor_absmax_fp8."""

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
        **tol,
    )


def _case(m, k, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    x = torch.randn(m, k, device=flaggems_sglang.device, generator=g).to(
        torch.bfloat16
    )
    return dict(
        input=x,
        output_s=torch.zeros(
            1, dtype=torch.float32, device=flaggems_sglang.device
        ),
        check=assert_close,
    )


_CORRECTNESS_CASES = [
    lambda: _case(1, 1024),
    lambda: _case(37, 4096, seed=1),
    lambda: _case(512, 2048, seed=2),
    lambda: _case(4096, 8192, seed=3),
]
_CORRECTNESS_CASES += [
    (lambda m=m, k=k: _case(m, k, seed=9))
    for m in (1, 8, 64, 512, 4096)
    for k in (2048, 4096, 8192)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.per_tensor_absmax_fp8
def test_per_tensor_absmax_fp8(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("per_tensor_absmax_fp8")(**kwargs)

    try:
        from flaggems_sglang.ops.per_tensor_absmax_fp8 import (
            per_tensor_absmax_fp8,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/per_tensor_absmax_fp8 ops module not found")
        return

    try:
        actual = per_tensor_absmax_fp8(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/per_tensor_absmax_fp8 not yet implemented")
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
