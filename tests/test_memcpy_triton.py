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

"""Correctness test for utility/memcpy_triton."""

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
# kernel-comp-baseline/problems/utility/memcpy_triton/cases.py)
# ---------------------------------------------------------------------------


def _case(rows, hidden, offset, sz, offset_src, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    src = torch.randn(rows, hidden, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    dst = torch.zeros(rows, hidden, dtype=torch.bfloat16, device=cfg.device)
    return dict(
        dst=dst,
        src=src,
        dim=0,
        offset=torch.tensor([offset], dtype=torch.int32, device=cfg.device),
        sz=torch.tensor([sz], dtype=torch.int32, device=cfg.device),
        offset_src=offset_src,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(8, 128, 2, 4, True),
    lambda: _case(8, 128, 2, 4, False, seed=1),
    lambda: _case(1024, 4096, 100, 500, True, seed=2),
    lambda: _case(4096, 2048, 0, 4096, False, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.memcpy_triton
def test_memcpy_triton(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("memcpy_triton")
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
        candidate = get_candidate("memcpy_triton", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("utility/memcpy_triton ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("utility/memcpy_triton not yet implemented")
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
