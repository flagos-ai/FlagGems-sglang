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

"""Correctness test for activation_norm/gemma_rmsnorm_residual_scalar."""

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


def _case(m, n, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    x = torch.randn(
        m, n, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    residual = torch.randn(
        m, n, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    weight = torch.randn(
        n, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    scalar = torch.full((1,), 0.75, dtype=torch.float32, device=cfg.device)
    return dict(
        x=x.contiguous(),
        weight=weight,
        residual=residual.contiguous(),
        scalar=scalar,
        eps=_EPS,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(7, 128),
    lambda: _case(83, 1024),
    lambda: _case(256, 4096),
    lambda: _case(48, 3072),
] + [
    (lambda m, n: lambda: _case(m, n))(m, n)
    for m in (1, 8, 64, 512, 4096)
    for n in (1024, 4096, 8192)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.gemma_rmsnorm_residual_scalar
def test_gemma_rmsnorm_residual_scalar(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("gemma_rmsnorm_residual_scalar")
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
        from flaggems_sglang import gemma_rmsnorm_residual_scalar
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "activation_norm/gemma_rmsnorm_residual_scalar ops module not found"
        )
        return

    try:
        actual = gemma_rmsnorm_residual_scalar(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "activation_norm/gemma_rmsnorm_residual_scalar not yet implemented"
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
