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

"""Correctness test for fla/fused_gdn_gating."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("fused_gdn_gating")


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
# Cases (from kernel-comp-baseline/problems/fla/fused_gdn_gating/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    a_g, a_beta = actual
    e_g, e_beta = expected
    assert_close(a_g, e_g, dtype=torch.float32)
    assert_close(a_beta, e_beta, dtype=torch.float32)


def _case(batch, num_heads, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    a_log = (
        torch.randn(
            num_heads,
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.float32,
        )
        * 0.5
    )
    a = torch.randn(
        batch,
        num_heads,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    )
    b = torch.randn(
        batch,
        num_heads,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    )
    dt_bias = torch.randn(
        num_heads,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    )
    return dict(A_log=a_log, a=a, b=b, dt_bias=dt_bias, check=_check)


CORRECTNESS_CASES = [
    _case(1, 4),
    _case(37, 32),
    _case(256, 8),
]

BENCH_CASES = [_case(m, 64) for m in (1, 8, 64, 512, 4096)]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_gdn_gating
def test_fused_gdn_gating(case_idx):
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
        from flaggems_sglang import fused_gdn_gating
    except (ImportError, ModuleNotFoundError):
        pytest.skip("fla/fused_gdn_gating ops module not found")
        return

    try:
        actual = fused_gdn_gating(**kwargs)
    except NotImplementedError:
        pytest.skip("fla/fused_gdn_gating not yet implemented")
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
