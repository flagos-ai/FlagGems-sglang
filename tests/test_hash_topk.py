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

"""Correctness test for moe/hash_topk."""

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
# Cases (from kernel-comp-baseline/problems/moe/hash_topk/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    # ids are integer routing output and must match exactly; weights go
    # through the per-dtype tolerance.
    torch.testing.assert_close(actual[1], expected[1])
    assert_close(actual[0], expected[0])


def _case(
    num_tokens,
    num_routed=256,
    topk_routed=8,
    shared=1,
    vocab=32768,
    seed=0,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    logits = torch.randn(
        num_tokens, num_routed, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    tid2eid = torch.randint(
        0,
        num_routed,
        (vocab, topk_routed),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    input_ids = torch.randint(
        0,
        vocab,
        (num_tokens,),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    return dict(
        router_logits=logits,
        input_ids=input_ids,
        tid2eid=tid2eid,
        num_fused_shared_experts=shared,
        routed_scaling_factor=2.5,
        scoring_func="sqrtsoftplus",
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(1),
    lambda: _case(37, seed=1),
    lambda: _case(512, topk_routed=4, seed=2),
    lambda: _case(1024, num_routed=384, shared=0, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.hash_topk
def test_hash_topk(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("hash_topk")
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
        candidate = get_candidate("hash_topk", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/hash_topk ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/hash_topk not yet implemented")
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
