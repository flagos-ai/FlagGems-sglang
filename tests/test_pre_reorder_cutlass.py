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

"""Correctness test for moe/pre_reorder_cutlass."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

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
# Cases (from kernel-comp-baseline/problems/moe/pre_reorder_cutlass/cases.py)
# ---------------------------------------------------------------------------


def _case(num_tokens, topk=8, hidden=4096, num_local_experts=32, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    n = num_tokens * topk
    x = torch.randn(num_tokens, hidden, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    # Expert id == num_local_experts marks a slot routed off this rank.
    ids = torch.randint(
        0,
        num_local_experts + 1,
        (num_tokens, topk),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    dst = (
        torch.randperm(
            n, generator=torch.Generator(device="cpu").manual_seed(seed)
        )
        .to(cfg.device)
        .reshape(num_tokens, topk)
    )
    return dict(
        input=x,
        gateup_input=torch.zeros(
            n, hidden, dtype=torch.bfloat16, device=cfg.device
        ),
        src2dst=dst.to(torch.int32),
        topk_ids=ids,
        a1_scales=torch.tensor([2.0], dtype=torch.float32, device=cfg.device),
        num_local_experts=num_local_experts,
        topk=topk,
        num_tokens=num_tokens,
        hidden_size=hidden,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(1),
    lambda: _case(17, topk=4, seed=1),
    lambda: _case(256, hidden=2048, seed=2),
    lambda: _case(512, topk=2, hidden=7168, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.pre_reorder_cutlass
def test_pre_reorder_cutlass(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("pre_reorder_cutlass")
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
        candidate = get_candidate("pre_reorder_cutlass", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/pre_reorder_cutlass ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/pre_reorder_cutlass not yet implemented")
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
