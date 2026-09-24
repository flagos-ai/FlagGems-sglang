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

"""Correctness test for diffusion/group_norm_silu."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from . import conftest as cfg

reference = get_reference("group_norm_silu")


# ---------------------------------------------------------------------------
# Tolerance helper
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
# Cases (from kernel-comp-baseline/problems/diffusion/group_norm_silu/cases.py)
# ---------------------------------------------------------------------------

import torch

from . import conftest as cfg


def _case(n, c, spatial, num_groups=32, eps=1e-5, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    x = torch.randn(n, c, spatial, spatial, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    w = torch.randn(c, device=cfg.device, generator=g).to(torch.bfloat16)
    b = torch.randn(c, device=cfg.device, generator=g).to(torch.bfloat16)
    return dict(
        x=x,
        weight=w,
        bias=b,
        num_groups=num_groups,
        eps=eps,
        check=assert_close,
    )


CORRECTNESS_CASES = [
    _case(1, 128, 16),
    _case(2, 256, 32, seed=1),
    _case(1, 512, 64, seed=2),
    _case(4, 64, 8, num_groups=8, seed=3),
]

BENCH_CASES = [
    _case(n, c, s, seed=9)
    for n in (1, 2)
    for c, s in ((128, 64), (256, 32), (512, 16))
]

CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES

# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.group_norm_silu
def test_group_norm_silu(case_idx):
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
        from flaggems_sglang import group_norm_silu
    except (ImportError, ModuleNotFoundError):
        pytest.skip("diffusion/group_norm_silu ops module not found")
        return

    try:
        actual = group_norm_silu(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/group_norm_silu not yet implemented")
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
