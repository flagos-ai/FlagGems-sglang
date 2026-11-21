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

"""Correctness test for moe/gemma4_fused_routing."""

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


def _check(actual, expected):
    a_w, a_i = actual
    e_w, e_i = expected
    assert_close(a_i, e_i)
    assert_close(a_w, e_w, dtype=torch.float32)


def _case(t, e, topk, dtype=torch.float32, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    # fp32 logits: the tie-break is part of the contract, but bf16 has few
    # enough distinct values at 128+ experts that exact ties would dominate
    # and the ordering of equal logits is not what this operator is about.
    gating = torch.randn(
        t, e, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    scale = (
        0.5
        + torch.rand(e, dtype=torch.float32, device=cfg.device, generator=g)
    ).to(dtype)
    return dict(
        gating_output=gating,
        per_expert_scale=scale,
        topk=topk,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(7, 8, 2),
    lambda: _case(128, 64, 8),
    lambda: _case(1024, 128, 4),
    lambda: _case(64, 1024, 1),
] + [
    (lambda t, e: lambda: _case(t, e, 8))(t, e)
    for t in (1, 8, 64, 512, 4096)
    for e in (64, 128, 256)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.gemma4_fused_routing
def test_gemma4_fused_routing(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("gemma4_fused_routing")
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
        candidate = get_candidate("gemma4_fused_routing", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/gemma4_fused_routing ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/gemma4_fused_routing not yet implemented")
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
