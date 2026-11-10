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

"""Correctness test for quantization/per_tensor_quant_fp8."""

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
    aq, asc = actual
    eq, esc = expected
    torch.testing.assert_close(asc.float(), esc.float(), rtol=2e-3, atol=1e-6)
    a_deq = aq.to(torch.float32) * asc.float()
    e_deq = eq.to(torch.float32) * esc.float()
    mismatch = (a_deq - e_deq).abs() > (2e-2 + 2e-2 * e_deq.abs())
    assert mismatch.float().mean() < 1e-2, "too many fp8 rounding-boundary mismatches"


def _case(m, k, is_static=False, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    x = torch.randn(
        m, k, device=flaggems_sglang.device, generator=g
    ).to(torch.bfloat16)
    return dict(
        input=x,
        output_q=torch.zeros(
            m, k, dtype=torch.float8_e4m3fn, device=flaggems_sglang.device
        ),
        output_s=torch.ones(1, dtype=torch.float32, device=flaggems_sglang.device)
        * (0.01 if is_static else 0.0),
        is_static=is_static,
        check=_check,
    )


_CORRECTNESS_CASES = [
    lambda: _case(1, 1024),
    lambda: _case(37, 4096, seed=1),
    lambda: _case(512, 2048, is_static=True, seed=2),
    lambda: _case(4096, 1024, seed=3),
]
_CORRECTNESS_CASES += [
    (lambda m=m, k=k: _case(m, k, seed=9))
    for m in (1, 8, 64, 512, 4096)
    for k in (2048, 4096, 8192)
]


@pytest.mark.parametrize("case_idx", range(len(_CORRECTNESS_CASES)))
@pytest.mark.per_tensor_quant_fp8
def test_per_tensor_quant_fp8(case_idx):
    case = _CORRECTNESS_CASES[case_idx]()
    check = case.pop("check", None) if isinstance(case, dict) and "check" in case else None
    kwargs = case if isinstance(case, dict) else {}

    expected = get_reference("per_tensor_quant_fp8")(**kwargs)

    try:
        from flaggems_sglang.ops.per_tensor_quant_fp8 import per_tensor_quant_fp8
    except (ImportError, ModuleNotFoundError):
        pytest.skip("quantization/per_tensor_quant_fp8 ops module not found")
        return

    try:
        actual = per_tensor_quant_fp8(**kwargs)
    except NotImplementedError:
        pytest.skip("quantization/per_tensor_quant_fp8 not yet implemented")
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
