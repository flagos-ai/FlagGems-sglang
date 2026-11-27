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

"""Correctness test for sampling_grammar/top_k_renorm."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.utils.rng_utils import create_generator

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


def _case(batch, vocab, k, seed=13):
    g = create_generator("cpu", seed=seed)
    probs = torch.rand((batch, vocab), generator=g)
    probs = (probs / probs.sum(-1, keepdim=True)).to(cfg.device)
    if k == "per_row":
        # One independent k per row.
        k = torch.tensor(
            [1 + (i * 127) % vocab for i in range(batch)],
            dtype=torch.int32,
            device=probs.device,
        )
    elif k == "singleton":
        # A [1] tensor broadcasts to every row.
        k = torch.tensor([3], dtype=torch.int64, device=probs.device)
    return dict(probs=probs, top_k=k, check=assert_close)


_CASE_FACTORIES = [
    lambda: _case(2, 7, 1),
    lambda: _case(2, 7, 7),
    lambda: _case(3, 17, "singleton"),
] + [
    (lambda b, v, k: lambda: _case(b, v, k))(b, v, k)
    for b, v, k in (
        (1, 111, 1),
        (7, 111, 10),
        (8, 32000, 100),
        (32, 128256, 500),
        (16, 4096, "per_row"),
        (4, 4096, 4096),
    )
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.top_k_renorm
def test_top_k_renorm(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("top_k_renorm")
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
        candidate = get_candidate("top_k_renorm", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("sampling_grammar/top_k_renorm ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("sampling_grammar/top_k_renorm not yet implemented")
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
