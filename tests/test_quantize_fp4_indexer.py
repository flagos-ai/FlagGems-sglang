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

"""Correctness test for quantization/quantize_fp4_indexer."""

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


def _case(n, dtype=torch.bfloat16, seed=0, scale=1.0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    x = (
        scale
        * torch.randn(
            n, 128, dtype=torch.float32, device=flaggems_sglang.device, generator=g
        )
    ).to(dtype)
    return dict(x=x.contiguous(), check=_check)


def _check(actual, expected):
    a_q, a_sf = actual
    e_q, e_sf = expected
    assert_close(a_sf, e_sf)
    assert_close(a_q.view(torch.uint8), e_q.view(torch.uint8))


_CORRECTNESS_CASES = [
    lambda: _case(7),
    lambda: _case(1024),
    lambda: _case(333, scale=8.0),
    lambda: _case(256, scale=0.01),
    lambda: _case(128, dtype=torch.float16),
]
_CORRECTNESS_CASES += [(lambda n=n: _case(n)) for n in (16, 128, 1024, 8192, 65536)]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.quantize_fp4_indexer
def test_quantize_fp4_indexer(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = case.pop("check", None) if isinstance(case, dict) and "check" in case else None
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("quantize_fp4_indexer")(**kwargs)

    try:
        from flaggems_sglang.ops.quantize_fp4_indexer import quantize_fp4_indexer
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/quantize_fp4_indexer ops module not found")
        return

    try:
        actual = quantize_fp4_indexer(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/quantize_fp4_indexer not yet implemented")
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
