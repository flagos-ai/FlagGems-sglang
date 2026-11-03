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

"""Correctness test for sglang/sglang_fl_rotary_embedding."""

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
# Cases
# ---------------------------------------------------------------------------


def _check(actual, expected):
    # actual/expected are (query, key) rotated tensors.
    assert_close(actual[0], expected[0])
    assert_close(actual[1], expected[1])


def _case(
    nnz,
    num_q_heads,
    num_kv_heads,
    head_size=128,
    rotary_dim=None,
    is_neox=True,
    max_position=8192,
    seed=0,
):
    rotary_dim = head_size if rotary_dim is None else rotary_dim
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    positions = torch.randint(
        0, max_position, (nnz,), device=cfg.device, generator=g
    )
    query = torch.randn(
        nnz, num_q_heads * head_size, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    key = torch.randn(
        nnz, num_kv_heads * head_size, device=cfg.device, generator=g
    ).to(torch.bfloat16)
    cos_sin_cache = torch.randn(
        max_position, rotary_dim, device=cfg.device, generator=g
    )
    return dict(
        positions=positions,
        query=query,
        key=key,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(1, 32, 8),
    lambda: _case(37, 16, 2, seed=1),
    lambda: _case(512, 8, 1, rotary_dim=64, seed=2),
    lambda: _case(128, 8, 2, is_neox=False, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.sglang_fl_rotary_embedding
def test_sglang_fl_rotary_embedding(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("sglang_fl_rotary_embedding")
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
        candidate = get_candidate(
            "sglang_fl_rotary_embedding", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("sglang/sglang_fl_rotary_embedding ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("sglang/sglang_fl_rotary_embedding not yet implemented")
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
