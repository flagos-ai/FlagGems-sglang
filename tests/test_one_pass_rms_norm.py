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

"""Correctness test for diffusion/one_pass_rms_norm."""

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


def _case(s, d, eps=1e-6, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    x = torch.randn(s, d, device=cfg.device, generator=g).to(torch.bfloat16)
    w = torch.randn(d, device=cfg.device, generator=g).to(torch.bfloat16)
    return dict(x=x, w=w, eps=eps, check=assert_close)


_CASE_FACTORIES = [
    lambda: _case(1, 1024),
    lambda: _case(37, 2048, seed=1),
    lambda: _case(4096, 3072, seed=2),
    lambda: _case(32768, 1536, seed=3),
] + [
    (lambda s, d: lambda: _case(s, d, seed=9))(s, d)
    for s in (1, 64, 1024, 16384)
    for d in (1536, 3072)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.one_pass_rms_norm
def test_one_pass_rms_norm(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("one_pass_rms_norm")
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
        from flaggems_sglang import one_pass_rms_norm
    except (ImportError, ModuleNotFoundError):
        pytest.skip("diffusion/one_pass_rms_norm ops module not found")
        return

    try:
        actual = one_pass_rms_norm(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/one_pass_rms_norm not yet implemented")
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
