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

"""Correctness test for speculative/assign_extend_cache_locs_uniform."""

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


def _case(bs, draft_token_num=8, max_batch=4096, pool_len=8192, seed=0):
    g = create_generator(cfg.device, seed=seed)
    req_to_token = torch.randint(
        0,
        1 << 20,
        (max_batch, pool_len),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    pool_idx = torch.randperm(max_batch, device=cfg.device, generator=g)[
        :bs
    ].to(torch.int32)
    start_offset = torch.randint(
        0,
        pool_len - draft_token_num,
        (bs,),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    return dict(
        req_pool_indices=pool_idx,
        req_to_token=req_to_token,
        start_offset=start_offset,
        batch_size=bs,
        draft_token_num=draft_token_num,
        device=torch.device(cfg.device),
        check=assert_close,
    )


_CASE_FACTORIES = [
    lambda: _case(1),
    lambda: _case(17, seed=1),
    lambda: _case(128, draft_token_num=4, seed=2),
    lambda: _case(512, draft_token_num=64, seed=3),
] + [
    (lambda b: lambda: _case(b, draft_token_num=8, seed=9))(b)
    for b in (1, 8, 64, 512, 2048)
]


CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.assign_extend_cache_locs_uniform
def test_assign_extend_cache_locs_uniform(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("assign_extend_cache_locs_uniform")
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
            "assign_extend_cache_locs_uniform", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "speculative/assign_extend_cache_locs_uniform ops module not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "speculative/assign_extend_cache_locs_uniform not yet implemented"
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
