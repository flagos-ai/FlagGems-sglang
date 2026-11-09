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

"""Correctness test for diffusion/causal_conv3d_cat_pad."""

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


def _case(b, c, t, h, w, cache_t, padding, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device=cfg.device).manual_seed(seed)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)

    return dict(
        x=_t(b, c, t, h, w).contiguous(),
        cache_x=_t(b, c, cache_t, h, w).contiguous(),
        padding=padding,
        check=assert_close,
    )


# padding = (w_l, w_r, h_t, h_b, t_front, t_back); t_front must be >= cache_t.
_CASE_FACTORIES = [
    lambda: _case(1, 16, 4, 8, 8, cache_t=2, padding=(1, 1, 1, 1, 2, 0)),
    lambda: _case(2, 32, 8, 16, 16, cache_t=1, padding=(1, 1, 1, 1, 2, 0)),
    lambda: _case(1, 8, 3, 32, 32, cache_t=2, padding=(2, 2, 2, 2, 2, 0)),
] + [
    (
        lambda t: lambda: _case(
            1, 128, t, 32, 32, cache_t=2, padding=(1, 1, 1, 1, 2, 0)
        )
    )(t)
    for t in (4, 8, 16)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.causal_conv3d_cat_pad
def test_causal_conv3d_cat_pad(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("causal_conv3d_cat_pad")
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
        from flaggems_sglang import causal_conv3d_cat_pad
    except (ImportError, ModuleNotFoundError):
        pytest.skip("diffusion/causal_conv3d_cat_pad ops module not found")
        return

    try:
        actual = causal_conv3d_cat_pad(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/causal_conv3d_cat_pad not yet implemented")
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
