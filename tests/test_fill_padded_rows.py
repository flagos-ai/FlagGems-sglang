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
"""Correctness test for moe/fill_padded_rows."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from . import conftest as cfg

reference = get_reference("fill_padded_rows")


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
# Cases (from kernel-comp-baseline/problems/moe/fill_padded_rows/cases.py)
# ---------------------------------------------------------------------------

import torch

from . import conftest as cfg


def _case(rows, cols, valid, dtype=torch.int32, fill_value=-1, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    if dtype.is_floating_point:
        x = torch.randn(rows, cols, device=cfg.device, generator=g).to(dtype)
    else:
        x = torch.randint(
            0, 256, (rows, cols), dtype=dtype, device=cfg.device, generator=g
        )
    return dict(
        x=x,
        num_token_non_padded=torch.tensor(
            [valid], dtype=torch.int32, device=cfg.device
        ),
        fill_value=fill_value,
        check=assert_close,
    )


CORRECTNESS_CASES = [
    _case(1, 8, 1),
    _case(64, 8, 30, seed=1),
    _case(512, 4, 0, seed=2),
    _case(256, 8, 256, dtype=torch.float32, fill_value=0.0, seed=3),
]

BENCH_CASES = [
    _case(rows, 8, rows // 2, seed=9) for rows in (8, 64, 512, 4096, 16384)
]

CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fill_padded_rows
def test_fill_padded_rows(case_idx):
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
        from flaggems_sglang import fill_padded_rows
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/fill_padded_rows ops module not found")
        return

    try:
        actual = fill_padded_rows(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/fill_padded_rows not yet implemented")
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
