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

"""Correctness test for sampling_grammar/apply_token_bitmask."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("apply_token_bitmask")


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
    equal_nan = tol.pop("equal_nan", True)
    torch.testing.assert_close(
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual,
        (
            expected.to(torch.float32)
            if expected.dtype.is_floating_point
            else expected
        ),
        equal_nan=equal_nan,
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/sampling_grammar/apply_token_bitmask/cases.py)
# ---------------------------------------------------------------------------


def _case(B, V, dtype=torch.float32, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    logits = torch.randn(
        B, V, generator=g, device=flaggems_sglang.device, dtype=dtype
    )
    words = (V + 31) // 32
    bitmask = torch.randint(
        torch.iinfo(torch.int32).min,
        torch.iinfo(torch.int32).max,
        (B, words),
        dtype=torch.int32,
        device=flaggems_sglang.device,
        generator=g,
    )
    return dict(logits=logits, bitmask=bitmask, check=_check)


def _check(actual, expected):
    assert_close(
        actual,
        expected,
        dtype=torch.float32,
        atol=0.0,
        rtol=0.0,
        equal_nan=True,
    )


CORRECTNESS_CASES = [
    _case(3, 100),
    _case(7, 1000),
    _case(1, 32000),
]

BENCH_CASES = [_case(b, 152064) for b in (1, 8, 64, 512, 4096)]

CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.apply_token_bitmask
def test_apply_token_bitmask(case_idx):
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
        from flaggems_sglang.ops.apply_token_bitmask import apply_token_bitmask
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "sampling_grammar/apply_token_bitmask ops module not found"
        )
        return

    try:
        actual = apply_token_bitmask(**kwargs)
    except NotImplementedError:
        pytest.skip("sampling_grammar/apply_token_bitmask not yet implemented")
        return

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if a is None and e is None:
                continue
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    # Restore check for reuse
    if check is not None:
        case["check"] = check
