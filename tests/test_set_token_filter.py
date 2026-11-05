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

"""Correctness test for utility/set_token_filter."""

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
# Cases (from
# kernel-comp-baseline/problems/utility/set_token_filter/cases.py)
# ---------------------------------------------------------------------------


def _case(batch, vocab, num_tokens, is_allowed=True, reset=True, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    words = (vocab + 31) // 32
    mask = torch.randint(
        -(2**31),
        2**31 - 1,
        (batch, words),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    # token_ids is a Python list, so the permutation stays on the CPU.
    ids = torch.randperm(vocab, generator=torch.Generator().manual_seed(seed))[
        :num_tokens
    ].tolist()
    return dict(
        vocab_mask=mask,
        token_ids=ids,
        batch_idx=batch // 2,
        is_allowed=is_allowed,
        reset_vocab_mask=reset,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(1, 1024, 8),
    lambda: _case(8, 32768, 256, seed=1),
    lambda: _case(16, 152064, 1024, is_allowed=False, seed=2),
    lambda: _case(4, 65536, 0, reset=True, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.set_token_filter
def test_set_token_filter(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("set_token_filter")
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
        candidate = get_candidate("set_token_filter", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("utility/set_token_filter ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("utility/set_token_filter not yet implemented")
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
