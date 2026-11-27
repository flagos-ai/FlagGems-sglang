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

"""Correctness test for kvcache/reshape_and_cache_flash."""

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


def _check(actual, expected):
    for a, e in zip(actual, expected):
        assert_close(a, e)


def _case(
    tokens,
    heads,
    head_size,
    blocks,
    block_size=16,
    seed=0,
    dtype=torch.bfloat16,
    swa=False,
    scale=False,
    drop=0.0,
):
    g = create_generator(cfg.device, seed=seed)

    def _t(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=cfg.device, generator=g
        ).to(dtype)

    total_slots = blocks * block_size
    slots = torch.randperm(total_slots, generator=g, device=cfg.device)[
        :tokens
    ].to(torch.int64)
    swa_table = None
    if swa:
        # The first table indexes a second remap table.
        swa_table = torch.randperm(
            total_slots, generator=g, device=cfg.device
        ).to(torch.int64)
    if drop:
        # slot < 0 means drop the token: the kernel touches no cache row.
        mask = torch.rand(slots.shape, device=cfg.device, generator=g) < drop
        slots = torch.where(mask, torch.full_like(slots, -1), slots)

    return dict(
        key=_t(tokens, heads, head_size).contiguous(),
        value=_t(tokens, heads, head_size).contiguous(),
        key_cache=_t(blocks, block_size, heads, head_size).contiguous(),
        value_cache=_t(blocks, block_size, heads, head_size).contiguous(),
        slot_mapping=slots,
        swa_slot_mapping=swa_table,
        k_scale=(
            torch.full((1,), 0.5, dtype=torch.float32, device=cfg.device)
            if scale
            else None
        ),
        v_scale=(
            torch.full((1,), 2.0, dtype=torch.float32, device=cfg.device)
            if scale
            else None
        ),
        check=_check,
    )


_CASE_FACTORIES = [
    lambda: _case(16, 4, 64, blocks=8),
    lambda: _case(1024, 8, 128, blocks=128, drop=0.2),
    lambda: _case(333, 2, 96, blocks=64, scale=True),
    lambda: _case(128, 4, 64, blocks=32, swa=True),
] + [
    (lambda t: lambda: _case(t, 8, 128, blocks=max(4, t // 16 + 1)))(t)
    for t in (16, 128, 1024, 4096)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.reshape_and_cache_flash
def test_reshape_and_cache_flash(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("reshape_and_cache_flash")
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
            "reshape_and_cache_flash", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("kvcache/reshape_and_cache_flash ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("kvcache/reshape_and_cache_flash not yet implemented")
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
