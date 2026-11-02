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

"""Correctness test for speculative/gather_spec_extras."""

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
# kernel-comp-baseline/problems/speculative/gather_spec_extras/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    for a, e in zip(actual, expected):
        if e is None:
            assert a is None
        else:
            torch.testing.assert_close(a, e)


def _case(m, pool=4096, topk=8, hidden=4096, with_hidden=True, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(seed)
    return dict(
        indices=torch.randperm(pool, device="cpu", generator=perm_g)[:m].to(
            device=cfg.device, dtype=torch.int64
        ),
        topk_p_buf=torch.rand(
            pool, topk, dtype=torch.float32, device=cfg.device, generator=g
        ),
        topk_index_buf=torch.randint(
            0,
            128000,
            (pool, topk),
            dtype=torch.int64,
            device=cfg.device,
            generator=g,
        ),
        output_tokens_buf=torch.randint(
            0,
            128000,
            (pool,),
            dtype=torch.int64,
            device=cfg.device,
            generator=g,
        ),
        hidden_states_buf=(
            torch.randn(pool, hidden, device=cfg.device, generator=g).to(
                torch.bfloat16
            )
            if with_hidden
            else None
        ),
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(1),
    lambda: _case(37, seed=1),
    lambda: _case(512, with_hidden=False, seed=2),
    lambda: _case(2048, hidden=7168, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.gather_spec_extras
def test_gather_spec_extras(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("gather_spec_extras")
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
        candidate = get_candidate("gather_spec_extras", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("speculative/gather_spec_extras ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("speculative/gather_spec_extras not yet implemented")
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
