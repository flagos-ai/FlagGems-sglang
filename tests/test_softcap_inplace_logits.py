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

"""Correctness test for activation_norm/softcap_inplace_logits."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("softcap_inplace_logits")


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
# Cases (from kernel-comp-baseline/problems/activation_norm/
# softcap_inplace_logits)
# ---------------------------------------------------------------------------


def _logits(m, n, dtype=torch.float32, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    return (
        torch.randn(
            m,
            n,
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.float32,
        )
        * 20
    ).to(dtype)


def _case(m, n, cap=30.0):
    return dict(full_logits=_logits(m, n), final_logit_softcapping=cap)


CORRECTNESS_CASES = [_case(1, 17), _case(37, 1024), _case(4, 32000)]

BENCH_CASES = [
    _case(m, n) for m in (1, 8, 64, 512) for n in (4096, 32000, 128256)
]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.softcap_inplace_logits
def test_softcap_inplace_logits(case_idx):
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
        from flaggems_sglang import softcap_inplace_logits
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "activation_norm/softcap_inplace_logits ops module not found"
        )
        return

    # The op caps in place, so it runs on a copy: the module-level case
    # tensor stays at its pre-cap value and the case remains reusable.
    kwargs = dict(kwargs, full_logits=kwargs["full_logits"].clone())

    try:
        actual = softcap_inplace_logits(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "activation_norm/softcap_inplace_logits not yet implemented"
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
