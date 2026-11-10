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

"""Correctness test for quantization/per_token_quant_int8_sum."""

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
        expected.to(torch.float32) if expected.dtype.is_floating_point else expected,
        equal_nan=True,
        **tol,
    )


def _check(actual, expected):
    a = actual[0].to(torch.int32)
    e = expected[0].to(torch.int32)
    diff = (a - e).abs()
    assert int(diff.max()) <= 1, f"int8 code off by {int(diff.max())}"
    off = (diff > 0).float().mean()
    assert off < 1e-3, f"{off:.4%} of codes differ (rounding ties only expected)"
    assert_close(actual[1], expected[1], dtype=torch.float32)
    if len(expected) > 2:
        assert_close(actual[2], expected[2])


def _case(m, n, cal_sum=True, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    x = torch.randn(
        m, n, device=flaggems_sglang.device, generator=g
    ).to(torch.bfloat16).contiguous()
    return dict(x=x, scale_dtype=torch.float32, cal_sum=cal_sum, check=_check)


_CORRECTNESS_CASES = [
    lambda: _case(1, 1024),
    lambda: _case(37, 4096, seed=1),
    lambda: _case(512, 2048, cal_sum=False, seed=2),
    lambda: _case(4096, 1024, seed=3),
]
_CORRECTNESS_CASES += [
    (lambda m=m, n=n: _case(m, n, seed=9))
    for m in (1, 8, 64, 512, 4096)
    for n in (2048, 4096, 8192)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.per_token_quant_int8_sum
def test_per_token_quant_int8_sum(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = case.pop("check", None) if isinstance(case, dict) and "check" in case else None
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("per_token_quant_int8_sum")(**kwargs)

    try:
        from flaggems_sglang.ops.per_token_quant_int8_sum import (
            per_token_quant_int8_sum,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/per_token_quant_int8_sum ops module not found")
        return

    try:
        actual = per_token_quant_int8_sum(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/per_token_quant_int8_sum not yet implemented")
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
