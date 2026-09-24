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
"""Correctness test for moe/deepep_permute."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

from . import conftest as cfg

reference = get_reference("deepep_permute")


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
# Cases (from kernel-comp-baseline/problems/moe/deepep_permute/cases.py)
# ---------------------------------------------------------------------------

import torch

from . import conftest as cfg


def _case(num_tokens, topk=8, hidden=4096, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    n = num_tokens * topk
    x = torch.randn(num_tokens, hidden, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    dst = (
        torch.randperm(
            n, generator=torch.Generator(device="cpu").manual_seed(seed)
        )
        .reshape(num_tokens, topk)
        .to(torch.int32)
        .to(cfg.device)
    )
    return dict(
        input=x,
        gateup_input=torch.zeros(
            n, hidden, dtype=torch.bfloat16, device=cfg.device
        ),
        src2dst=dst,
        topk_ids=torch.zeros(
            num_tokens, topk, dtype=torch.int32, device=cfg.device
        ),
        topk=topk,
        hidden_size=hidden,
        check=assert_close,
    )


CORRECTNESS_CASES = [
    _case(1),
    _case(17, topk=4, seed=1),
    _case(256, hidden=2048, seed=2),
    _case(512, topk=2, hidden=7168, seed=3),
]

BENCH_CASES = [_case(t, hidden=7168, seed=9) for t in (1, 8, 64, 512, 2048)]

CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.deepep_permute
def test_deepep_permute(case_idx):
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
        from flaggems_sglang import deepep_permute
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/deepep_permute ops module not found")
        return

    try:
        actual = deepep_permute(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/deepep_permute not yet implemented")
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
