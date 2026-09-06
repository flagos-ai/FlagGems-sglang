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

"""Correctness test for attention/merge_state."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("merge_state")


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
# Cases (from kernel-comp-baseline/problems/attention/merge_state/cases.py)
# ---------------------------------------------------------------------------


def _case(n_tokens, num_heads, head_size, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    prefix_output = torch.randn(
        n_tokens,
        num_heads,
        head_size,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    suffix_output = torch.randn(
        n_tokens,
        num_heads,
        head_size,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    prefix_lse = (
        torch.randn(
            n_tokens, num_heads, generator=g, device=flaggems_sglang.device
        )
        * 3
    )
    suffix_lse = (
        torch.randn(
            n_tokens, num_heads, generator=g, device=flaggems_sglang.device
        )
        * 3
    )
    return dict(
        prefix_output=prefix_output,
        prefix_lse=prefix_lse,
        suffix_output=suffix_output,
        suffix_lse=suffix_lse,
        check=_check,
    )


def _check(actual, expected):
    a_out, a_lse = actual
    e_out, e_lse = expected
    assert_close(a_out, e_out)
    assert_close(a_lse, e_lse, dtype=torch.float32)


CORRECTNESS_CASES = [
    _case(7, 4, 64),
    _case(83, 16, 128),
    _case(3, 32, 512),
]

BENCH_CASES = [_case(n, 32, 128) for n in (1, 8, 64, 512, 4096)]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.merge_state
def test_merge_state(case_idx):
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
        from flaggems_sglang.ops.merge_state import merge_state
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/merge_state ops module not found")
        return

    try:
        actual = merge_state(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/merge_state not yet implemented")
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
