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

"""Correctness test for speculative/get_target_cache_loc."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

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
# Cases (from
# kernel-comp-baseline/problems/speculative/get_target_cache_loc/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def _case(bs, num_verify_tokens=8, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    correct = torch.randint(
        0,
        num_verify_tokens,
        (bs,),
        dtype=torch.int64,
        device=cfg.device,
        generator=g,
    )
    free = torch.randint(
        0,
        num_verify_tokens // 2 + 1,
        (bs,),
        dtype=torch.int64,
        device=cfg.device,
        generator=g,
    )
    out_cache_loc = torch.randint(
        1,
        1 << 20,
        (bs * num_verify_tokens,),
        dtype=torch.int64,
        device=cfg.device,
        generator=g,
    )
    return dict(
        num_correct_drafts=correct,
        to_free_num_slots=free,
        out_cache_loc=out_cache_loc,
        num_verify_tokens=num_verify_tokens,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(1),
    lambda: _case(9, seed=1),
    lambda: _case(64, num_verify_tokens=16, seed=2),
    lambda: _case(256, num_verify_tokens=4, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.get_target_cache_loc
def test_get_target_cache_loc(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("get_target_cache_loc")
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
        candidate = get_candidate("get_target_cache_loc", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("speculative/get_target_cache_loc ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("speculative/get_target_cache_loc not yet implemented")
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
