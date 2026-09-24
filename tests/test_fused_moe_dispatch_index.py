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
"""Correctness test for moe/fused_moe_dispatch_index."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from . import conftest as cfg

reference = get_reference("fused_moe_dispatch_index")


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
# Cases (from kernel-comp-baseline/problems/moe/fused_moe_dispatch_index/cases.py)
# ---------------------------------------------------------------------------

import torch

from . import conftest as cfg


def _check(actual, expected):
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(
        torch.sort(actual[1]).values, torch.sort(expected[1]).values
    )


def _case(num_tokens, topk, num_local_experts, m_max, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    topk_ids = torch.randint(
        0,
        num_local_experts,
        (num_tokens, topk),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    return dict(
        topk_ids=topk_ids,
        num_local_experts=num_local_experts,
        m_max=m_max,
        check=_check,
    )


CORRECTNESS_CASES = [
    _case(1, 8, 32, 128),
    _case(37, 4, 8, 256, seed=1),
    _case(128, 8, 32, 1024, seed=2),
    _case(512, 2, 16, 2048, seed=3),
]

BENCH_CASES = [_case(t, 8, 32, 8192, seed=9) for t in (1, 8, 64, 512, 4096)]

CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_moe_dispatch_index
def test_fused_moe_dispatch_index(case_idx):
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
        from flaggems_sglang import fused_moe_dispatch_index
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/fused_moe_dispatch_index ops module not found")
        return

    try:
        actual = fused_moe_dispatch_index(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/fused_moe_dispatch_index not yet implemented")
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
