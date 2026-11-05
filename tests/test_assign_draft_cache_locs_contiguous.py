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

"""Correctness test for speculative/assign_draft_cache_locs_contiguous."""

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
# kernel-comp-baseline/problems/speculative/assign_draft_cache_locs_contiguous/cases.py)
# ---------------------------------------------------------------------------


def _case(bs, topk=4, steps=3, max_batch=256, pool_len=8192, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    # randperm is generated on CPU and moved: some vendor backends have
    # no on-device kernel for it.
    perm_g = torch.Generator(device="cpu").manual_seed(seed)
    req_to_token = torch.randint(
        0,
        1 << 20,
        (max_batch, pool_len),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    pool_idx = torch.randperm(max_batch, device="cpu", generator=perm_g)[
        :bs
    ].to(device=cfg.device, dtype=torch.int32)
    seq_lens = torch.randint(
        0,
        pool_len - topk * steps,
        (bs,),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    return dict(
        req_pool_indices=pool_idx,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        out_cache_loc=torch.zeros(
            bs * topk * steps, dtype=torch.int32, device=cfg.device
        ),
        topk=topk,
        speculative_num_steps=steps,
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(1),
    lambda: _case(17, seed=1),
    lambda: _case(128, topk=8, steps=5, seed=2),
    lambda: _case(512, topk=1, steps=1, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.assign_draft_cache_locs_contiguous
def test_assign_draft_cache_locs_contiguous(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("assign_draft_cache_locs_contiguous")
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
            "assign_draft_cache_locs_contiguous", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "speculative/assign_draft_cache_locs_contiguous ops module not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "speculative/assign_draft_cache_locs_contiguous not yet implemented"
        )
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
