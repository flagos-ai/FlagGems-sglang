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

"""Correctness test for activation_norm/gelu_and_mul."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("gelu_and_mul")


# ---------------------------------------------------------------------------
# Tolerance helper (from kernel-comp-baseline/harness/correctness.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/activation_norm/gelu_and_mul/cases.py)
# ---------------------------------------------------------------------------


# _x inlined from silu_and_mul (decoupled from problems repo)
def _x(bs, d, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    return torch.randn(
        bs,
        2 * d,
        dtype=torch.float32,
        device=flaggems_sglang.device,
        generator=g,
    ).to(dtype)


def _check(actual, expected):
    assert_close(actual, expected)


CORRECTNESS_CASES = [
    dict(hidden_states=_x(7, 16), check=_check),
    dict(hidden_states=_x(83, 1024), check=_check),
    dict(hidden_states=_x(48, 3072), check=_check),
    dict(hidden_states=_x(1, 8192), check=_check),
]

BENCH_CASES = [
    dict(hidden_states=_x(bs, d), check=_check)
    for bs in (1, 8, 64, 512, 4096)
    for d in (1024, 4096, 8192)
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.gelu_and_mul
def test_gelu_and_mul(case_idx):
    case = CORRECTNESS_CASES[case_idx]
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    # Reference
    expected = reference(**kwargs)

    # Operator under test
    try:
        from flaggems_sglang.ops.gelu_and_mul import gelu_and_mul
    except (ImportError, ModuleNotFoundError):
        pytest.skip("activation_norm/gelu_and_mul ops module not found")
        return

    try:
        actual = gelu_and_mul(**kwargs)
    except NotImplementedError:
        pytest.skip("activation_norm/gelu_and_mul not yet implemented")
        return

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    # Restore check for reuse
    if check is not None:
        case["check"] = check
