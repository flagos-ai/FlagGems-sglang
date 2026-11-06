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

"""Correctness test for moe/fused_append_shared_experts."""

import pytest
import torch

from flaggems_sglang.reference import get_reference

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


def _case(m, k, s, n_base=256, scale=0.25, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    topk_ids = torch.randint(
        0,
        n_base,
        (m, k),
        device=cfg.device,
        generator=g,
        dtype=torch.int32,
    )
    topk_weights = torch.rand(
        m, k, dtype=torch.float32, device=cfg.device, generator=g
    )
    return dict(
        topk_ids=topk_ids.contiguous(),
        topk_weights=topk_weights.contiguous(),
        num_fused_shared_experts=s,
        scale_factor=scale,
        N=n_base,
        check=_check,
    )


def _check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e)


_CASE_FACTORIES = [
    lambda: _case(7, 8, 1),
    lambda: _case(1024, 8, 2),
    lambda: _case(333, 4, 4),
] + [
    (lambda m: lambda: _case(m, 8, 1))(m) for m in (1, 8, 64, 512, 4096, 16384)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_append_shared_experts
def test_fused_append_shared_experts(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("fused_append_shared_experts")
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
        from flaggems_sglang import fused_append_shared_experts
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/fused_append_shared_experts ops module not found")
        return

    try:
        actual = fused_append_shared_experts(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/fused_append_shared_experts not yet implemented")
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
