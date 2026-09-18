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

"""Correctness test for sampling_grammar/chain_speculative_sampling."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference

reference = get_reference("chain_speculative_sampling")


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
        equal_nan=True,
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/sampling_grammar/chain_speculative_sampling/cases.py)
# ---------------------------------------------------------------------------


def _case(B, S, V, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)

    candidates = torch.randint(
        0,
        V,
        (B, S),
        dtype=torch.int32,
        device=flaggems_sglang.device,
        generator=g,
    )
    retrive_index = torch.arange(
        B * S, dtype=torch.int64, device=flaggems_sglang.device
    ).reshape(B, S)
    uniform_samples = torch.rand(
        B, S - 1, generator=g, device=flaggems_sglang.device
    )
    uniform_samples_for_final_sampling = torch.rand(
        B, generator=g, device=flaggems_sglang.device
    )

    target_logits = torch.randn(
        B, S, V, generator=g, device=flaggems_sglang.device
    )
    target_probs = torch.softmax(target_logits, dim=-1)
    draft_logits = torch.randn(
        B, S - 1, V, generator=g, device=flaggems_sglang.device
    )
    draft_probs = torch.softmax(draft_logits, dim=-1)

    return dict(
        candidates=candidates,
        retrive_index=retrive_index,
        uniform_samples=uniform_samples,
        uniform_samples_for_final_sampling=uniform_samples_for_final_sampling,
        target_probs=target_probs,
        draft_probs=draft_probs,
        # Precomputed here (not derived via .item() inside baseline/solution)
        # since retrive_index = arange(B*S) makes it exact, and a host sync
        # inside the timed function breaks CUDA-graph capture in bench.py.
        num_slots=B * S,
        check=_check,
    )


def _check(actual, expected):
    a_pred, a_idx, a_num = actual
    e_pred, e_idx, e_num = expected
    assert torch.equal(a_pred, e_pred), "predicts mismatch"
    assert torch.equal(a_idx, e_idx), "accept_index mismatch"
    assert torch.equal(a_num, e_num), "accept_token_num mismatch"


CORRECTNESS_CASES = [
    _case(4, 5, 32, seed=0),
    _case(4, 5, 32, seed=1),
    _case(8, 8, 128, seed=2),
    _case(2, 3, 16, seed=3),
]

BENCH_CASES = [_case(b, 4, 32000, seed=0) for b in (1, 8, 64, 512, 4096)]
CORRECTNESS_CASES = CORRECTNESS_CASES + BENCH_CASES


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.chain_speculative_sampling
def test_chain_speculative_sampling(case_idx):
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
        from flaggems_sglang import chain_speculative_sampling
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "sampling_grammar/chain_speculative_sampling ops module not found"
        )
        return

    try:
        actual = chain_speculative_sampling(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "sampling_grammar/chain_speculative_sampling not yet implemented"
        )
        return

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if a is None and e is None:
                continue
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    # Restore check for reuse
    if check is not None:
        case["check"] = check
