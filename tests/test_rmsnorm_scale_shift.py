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

"""Correctness test for diffusion/rmsnorm_scale_shift."""

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


def _case(batch, seq, hidden, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=cfg.device, generator=g
        ).to(torch.bfloat16)

    return dict(
        x=_t(batch, seq, hidden).contiguous(),
        weight=_t(hidden).contiguous(),
        scale=_t(batch, 1, hidden).contiguous(),
        shift=_t(batch, 1, hidden).contiguous(),
        eps=_EPS,
        check=assert_close,
    )


# The kernel replicates FlashInfer's RMSNorm thread mapping, which only exists
# for hidden == 64 * threads_per_row -> 2048 (tpr 32) or 4096 (tpr 64).
_CASE_FACTORIES = [
    lambda: _case(1, 16, 2048),
    lambda: _case(2, 256, 4096),
    lambda: _case(4, 77, 2048),
] + [
    (lambda b, s: lambda: _case(b, s, 2048))(b, s)
    for b in (1, 2, 8)
    for s in (77, 1024, 4096)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.rmsnorm_scale_shift
def test_rmsnorm_scale_shift(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("rmsnorm_scale_shift")
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
        from flaggems_sglang import rmsnorm_scale_shift
    except (ImportError, ModuleNotFoundError):
        pytest.skip("diffusion/rmsnorm_scale_shift ops module not found")
        return

    try:
        actual = rmsnorm_scale_shift(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/rmsnorm_scale_shift not yet implemented")
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
