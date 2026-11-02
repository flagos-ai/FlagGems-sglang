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

"""Correctness test for attention/dcp_lse_combine."""

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
    # NaN at the same position is a match (both sides overflowed the same
    # way); NaN vs a real value still fails.
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
# Cases (from
# kernel-comp-baseline/problems/attention/dcp_lse_combine/cases.py)
# ---------------------------------------------------------------------------


def _case(
    n, b, h, d, base_e=True, return_lse=True, dtype=torch.bfloat16, seed=0
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    recv_output = torch.randn(
        n, b, h, d, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    recv_lse = (
        torch.randn(
            n, b, h, dtype=torch.float32, device=cfg.device, generator=g
        )
        * 4
    )
    return dict(
        recv_output=recv_output.contiguous(),
        recv_lse=recv_lse.contiguous(),
        is_lse_base_on_e=base_e,
        return_lse=return_lse,
        check=_check,
    )


def _check(actual, expected):
    a_out, a_lse = actual
    e_out, e_lse = expected
    assert_close(a_out, e_out)
    if e_lse is None:
        assert a_lse is None
    else:
        assert_close(a_lse, e_lse, dtype=torch.float32)


_CASE_FACTORIES = [
    lambda: _case(2, 7, 8, 128),
    lambda: _case(4, 256, 16, 128),
    lambda: _case(8, 64, 4, 512, base_e=False),
    lambda: _case(2, 128, 8, 128, return_lse=False),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.dcp_lse_combine
def test_dcp_lse_combine(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("dcp_lse_combine")
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
        candidate = get_candidate("dcp_lse_combine", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/dcp_lse_combine ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/dcp_lse_combine not yet implemented")
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
