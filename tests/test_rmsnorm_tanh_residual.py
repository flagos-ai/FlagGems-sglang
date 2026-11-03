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

"""Correctness test for diffusion/rmsnorm_tanh_residual."""

import pytest
import torch

from flaggems_sglang.reference import get_reference

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

_EPS = 1e-6


def _case(rows, dim, gate_rows=None, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=cfg.device, generator=g
        ).to(torch.bfloat16)

    gate_rows = gate_rows or rows
    return dict(
        x=_t(rows, dim).contiguous(),
        gate=_t(gate_rows, dim).contiguous(),
        residual=_t(rows, dim).contiguous(),
        weight=_t(dim).contiguous(),
        eps=_EPS,
        check=_check,
    )


def _check(actual, expected):
    # `residual + gated * normed` cancels for a handful of elements, where a
    # one-ulp difference upstream flips the sum to (or off) exact zero and
    # the relative error is unbounded. Allow a small mismatching fraction
    # rather than loosening the tolerance for everything.
    a, e = actual.to(torch.float32), expected.to(torch.float32)
    atol, rtol = 2e-2, 2e-2
    mismatch = (a - e).abs() > (atol + rtol * e.abs())
    assert mismatch.float().mean() < 1e-4, "too many mismatching elements"
    assert (
        a - e
    ).abs().max() < 0.1, "a mismatch is too large to be cancellation"


_CASE_FACTORIES = [
    lambda: _case(32, 2048),
    lambda: _case(1024, 4096),
    lambda: _case(64, 1536, gate_rows=8),
    lambda: _case(7, 256),
] + [
    (lambda r, d: lambda: _case(r, d))(r, d)
    for r in (8, 64, 1024, 8192)
    for d in (2048, 4096)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.rmsnorm_tanh_residual
def test_rmsnorm_tanh_residual(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("rmsnorm_tanh_residual")
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
        from flaggems_sglang import rmsnorm_tanh_residual
    except (ImportError, ModuleNotFoundError):
        pytest.skip("diffusion/rmsnorm_tanh_residual ops module not found")
        return

    try:
        actual = rmsnorm_tanh_residual(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/rmsnorm_tanh_residual not yet implemented")
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
