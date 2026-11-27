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

"""Correctness test for diffusion/wan_rmsnorm_silu."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg
from flaggems_sglang.utils.rng_utils import create_generator

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


def _to_channels_last_3d(x):
    """Keep NCDHW values and shape while materializing channels-last-3d."""
    if x.ndim != 5:
        raise ValueError("channels_last_3d requires a five-dimensional tensor")
    return x.contiguous(memory_format=torch.channels_last_3d)


def _case(b, c, t, h, w, with_bias=False, seed=0):
    g = create_generator(cfg.device, seed=seed)
    x = _to_channels_last_3d(
        torch.randn(b, c, t, h, w, device=cfg.device, generator=g).to(
            torch.bfloat16
        )
    )
    gamma = torch.randn(c, device=cfg.device, generator=g).to(torch.bfloat16)
    bias = (
        torch.randn(c, device=cfg.device, generator=g).to(torch.bfloat16)
        if with_bias
        else None
    )
    return dict(
        x=x,
        gamma=gamma,
        bias=bias,
        rms_scale=None,
        eps=1e-12,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(1, 64, 1, 8, 8),
    lambda: _case(1, 128, 4, 16, 16, with_bias=True, seed=1),
    lambda: _case(2, 256, 2, 32, 32, seed=2),
    lambda: _case(1, 512, 1, 16, 16, with_bias=True, seed=3),
] + [
    (lambda c, s: lambda: _case(1, c, 4, s, s, seed=9))(c, s)
    for c in (128, 512)
    for s in (16, 32, 64)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.wan_rmsnorm_silu
def test_wan_rmsnorm_silu(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("wan_rmsnorm_silu")
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
        candidate = get_candidate("wan_rmsnorm_silu", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("diffusion/wan_rmsnorm_silu ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/wan_rmsnorm_silu not yet implemented")
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
