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

"""Correctness test for activation_norm/grouped_gemma_rmsnorm."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

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
# Cases
# ---------------------------------------------------------------------------


def _case(shape, group_size, dtype, eps, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    x = torch.randn(*shape, generator=g).to(device=cfg.device, dtype=dtype)
    weight = (
        torch.randn(shape[-1], generator=g).to(device=cfg.device, dtype=dtype)
        * 0.2
    ).to(dtype)
    return dict(
        x=x, weight=weight, group_size=group_size, eps=eps, check=assert_close
    )


# (shape, group_size, dtype, eps) — group_size is a multiple of 512 that
# divides H.
_BENCH_SPECS = [
    (shape, group, dtype, 1e-6)
    for dtype in (torch.float16, torch.bfloat16)
    for shape, group in (
        ((1, 10240), 2560),
        ((33, 1024), 512),
        ((128, 10240), 2560),
        ((1024, 2048), 1024),
    )
]
_EXTRA_SPECS = [
    ((5, 512), 512, dtype, eps)
    for dtype in (torch.float16, torch.bfloat16)
    for eps in (1e-6, 1e-5)
] + [((2, 3, 2048), 512, torch.bfloat16, 1e-5)]

_CASE_FACTORIES = [(lambda s: lambda: _case(*s))(s) for s in _EXTRA_SPECS] + [
    (lambda s: lambda: _case(*s))(s) for s in _BENCH_SPECS
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.grouped_gemma_rmsnorm
def test_grouped_gemma_rmsnorm(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("grouped_gemma_rmsnorm")
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
        candidate = get_candidate("grouped_gemma_rmsnorm", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "activation_norm/grouped_gemma_rmsnorm ops module not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "activation_norm/grouped_gemma_rmsnorm not yet implemented"
        )
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
