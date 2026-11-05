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

"""Correctness test for moe/moe_align_block_size."""

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
# kernel-comp-baseline/problems/moe/moe_align_block_size/cases.py)
# ---------------------------------------------------------------------------


def _expert_runs(expert_ids, num_blocks):
    """Consecutive block ranges that belong to the same expert."""
    runs = []
    beg = 0
    ids = expert_ids[:num_blocks].tolist()
    for i in range(1, num_blocks + 1):
        if i == num_blocks or ids[i] != ids[beg]:
            runs.append((beg, i))
            beg = i
    return runs


def _make_check(block_size, pad_value):
    def _check(actual, expected):
        a_sorted, a_eids, a_npost = actual
        e_sorted, e_eids, e_npost = expected
        torch.testing.assert_close(a_npost, e_npost)

        n = int(e_npost[0])
        nb = n // block_size
        torch.testing.assert_close(a_eids[:nb], e_eids[:nb])

        # Tokens are placed inside their expert's range by an atomic cursor,
        # so the order within one expert is not deterministic. Compare each
        # expert's range as a multiset instead.
        for beg, end in _expert_runs(e_eids, nb):
            lo, hi = beg * block_size, end * block_size
            torch.testing.assert_close(
                torch.sort(a_sorted[lo:hi]).values,
                torch.sort(e_sorted[lo:hi]).values,
            )
        # Everything past num_tokens_post_pad stays at the padding sentinel.
        assert bool(
            (a_sorted[n:] == pad_value).all()
        ), "tail not left as padding"

    return _check


def _case(num_tokens, top_k, num_experts, block_size, seed=0):
    """``num_experts`` is the model's routed-expert count; ids run in
    ``[0, num_experts)``. The kernel is called with ``num_experts + 1`` -- the
    extra slot is the EP "filtered expert" bucket and gets no blocks, which is
    exactly how ``moe_runner.triton_utils`` calls it."""
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int32,
        device=cfg.device,
        generator=g,
    )
    numel = topk_ids.numel()
    arg_experts = num_experts + 1
    if numel < arg_experts + 1:
        max_padded = numel * block_size
    else:
        max_padded = numel + (arg_experts + 1) * (block_size - 1)
    # Baseline uses triton.cdiv; spelled out to keep triton off the test's
    # import path.
    max_blocks = (max_padded + block_size - 1) // block_size

    return dict(
        topk_ids=topk_ids,
        num_experts=arg_experts,
        block_size=block_size,
        sorted_token_ids=torch.full(
            (max_padded,), numel, dtype=torch.int32, device=cfg.device
        ),
        expert_ids=torch.zeros(
            max_blocks, dtype=torch.int32, device=cfg.device
        ),
        num_tokens_post_pad=torch.zeros(
            1, dtype=torch.int32, device=cfg.device
        ),
        cumsum_buffer=torch.zeros(
            arg_experts + 1, dtype=torch.int32, device=cfg.device
        ),
        pad_sorted_token_ids=True,
        check=_make_check(block_size, numel),
    )


_CASE_FACTORIES = [
    lambda: _case(1, 8, 64, 64),
    lambda: _case(33, 6, 128, 64, seed=1),
    lambda: _case(512, 8, 256, 128, seed=2),
    lambda: _case(4096, 4, 32, 16, seed=3),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.moe_align_block_size
def test_moe_align_block_size(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("moe_align_block_size")
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
        candidate = get_candidate("moe_align_block_size", reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/moe_align_block_size ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/moe_align_block_size not yet implemented")
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
