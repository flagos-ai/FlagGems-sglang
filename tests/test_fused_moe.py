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

"""Correctness test for moe/fused_moe."""

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
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=2e-2, rtol=2e-2)


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
# Cases
# ---------------------------------------------------------------------------


def _case(
    num_tokens,
    hidden,
    inter,
    num_experts,
    topk,
    activation="silu",
    routed_scaling_factor=1.0,
    seed=0,
):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    x = torch.randn(num_tokens, hidden, device=cfg.device, generator=g).to(
        torch.bfloat16
    )
    # Keep the weights small so the two-GEMM bf16 path stays well conditioned.
    w1 = (
        torch.randn(
            num_experts, 2 * inter, hidden, device=cfg.device, generator=g
        )
        * 0.05
    ).to(torch.bfloat16)
    w2 = (
        torch.randn(num_experts, hidden, inter, device=cfg.device, generator=g)
        * 0.05
    ).to(torch.bfloat16)
    logits = torch.randn(
        num_tokens, num_experts, device=cfg.device, generator=g
    )
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    topk_output = (
        topk_weights.float().contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        None,
    )
    return dict(
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_output=topk_output,
        activation=activation,
        routed_scaling_factor=routed_scaling_factor,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(1, 128, 256, 8, 2),
    lambda: _case(37, 256, 512, 8, 2, seed=1),
    lambda: _case(64, 512, 256, 16, 4, routed_scaling_factor=2.5, seed=2),
    lambda: _case(128, 256, 512, 8, 2, activation="gelu", seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_moe
def test_fused_moe(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("fused_moe")
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
        candidate = get_candidate("fused_moe", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/fused_moe ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/fused_moe not yet implemented")
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
