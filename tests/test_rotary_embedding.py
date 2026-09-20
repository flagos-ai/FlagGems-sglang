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

"""Correctness test for diffusion/rotary_embedding."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("rotary_embedding")


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
# Cases (from kernel-comp-baseline/problems/diffusion/rotary_embedding)
# ---------------------------------------------------------------------------


def _case(tokens, heads=16, head_size=128, seed=0):
    device = flaggems_sglang.device
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, heads, head_size, device=device, generator=g).to(
        torch.bfloat16
    )
    ang = torch.randn(tokens, head_size // 2, device=device, generator=g)
    return dict(
        x=x,
        cos=torch.cos(ang).to(torch.bfloat16),
        sin=torch.sin(ang).to(torch.bfloat16),
        interleaved=False,
    )


CORRECTNESS_CASES = [
    _case(1),
    _case(37, seed=1),
    _case(1024, heads=8, seed=2),
    _case(4096, head_size=64, seed=3),
]

BENCH_CASES = [_case(t, seed=9) for t in (1, 64, 1024, 8192, 32768)]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.rotary_embedding
def test_rotary_embedding(case_idx):
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
        from flaggems_sglang import rotary_embedding
    except (ImportError, ModuleNotFoundError):
        pytest.skip("diffusion/rotary_embedding ops module not found")
        return

    try:
        actual = rotary_embedding(**kwargs)
    except NotImplementedError:
        pytest.skip("diffusion/rotary_embedding not yet implemented")
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
