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

"""Correctness test for gemm/grouped_gemm."""

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

_BLOCK_SIZE_M = 128


def _grouped_gemm_metadata(sorted_topk_ids, num_experts, block_size_m=128):
    """Rebuild SGLang's grouped-gemm metadata tensors with torch ops.

    sorted_topk_ids rows are expert-sorted (each row belongs to exactly one
    expert). Returns, on the same device, all int32: the per-expert token
    count, the token and block ranges, and the block schedule, which holds
    (local_block << 16) | expert per block and -1 outside any expert's blocks.
    """
    num_routed_tokens = sorted_topk_ids.numel()
    device = sorted_topk_ids.device

    # expert_token_offs[e + 1] = number of tokens with expert id <= e. SGLang
    # does a per-expert binary search over the sorted ids; searchsorted with
    # side="right" computes the same count.
    expert_token_offs = torch.empty(
        num_experts + 1, device=device, dtype=torch.int32
    )
    expert_token_offs[0] = 0
    expert_ids = torch.arange(num_experts, device=device, dtype=torch.int32)
    expert_token_offs[1:] = torch.searchsorted(
        sorted_topk_ids, expert_ids, side="right"
    ).to(torch.int32)

    num_tokens_per_expert = (
        expert_token_offs[1:] - expert_token_offs[:-1]
    ).contiguous()

    # Block ranges: ceil-div each expert's token count by block_size_m, then
    # exclusive-prefix-sum, so expert e's blocks start where e-1's end.
    num_blocks_per_expert = (
        num_tokens_per_expert + block_size_m - 1
    ) // block_size_m
    expert_block_offs = torch.empty(
        num_experts + 1, device=device, dtype=torch.int32
    )
    expert_block_offs[0] = 0
    expert_block_offs[1:num_experts] = torch.cumsum(
        num_blocks_per_expert, dim=0
    )[: num_experts - 1]
    expert_block_offs[num_experts] = int(num_blocks_per_expert.sum())

    max_num_blocks = (
        (num_routed_tokens + block_size_m - 1) // block_size_m
        + num_experts
        - 1
    )
    expert_block_schedule = torch.full(
        (max_num_blocks,), -1, device=device, dtype=torch.int32
    )
    for e in range(num_experts):
        nb = int(num_blocks_per_expert[e])
        off = int(expert_block_offs[e])
        if nb > 0:
            local = torch.arange(nb, device=device, dtype=torch.int32)
            expert_block_schedule[off : off + nb] = (local << 16) + e

    return (
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
    )


def _case(m, n, k, num_experts, dtype=torch.bfloat16, seed=0):
    """Rows are already expert-sorted, so every row of the output belongs to
    exactly one expert and is written by exactly one block."""
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    a = torch.randn(
        m, k, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    b = (
        0.1
        * torch.randn(
            num_experts,
            n,
            k,
            dtype=torch.float32,
            device=cfg.device,
            generator=g,
        )
    ).to(dtype)
    ids = torch.randint(
        0,
        num_experts,
        (m,),
        device=cfg.device,
        generator=g,
        dtype=torch.int32,
    )
    sorted_ids = torch.sort(ids)[0].contiguous()
    (
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
    ) = _grouped_gemm_metadata(sorted_ids, num_experts, _BLOCK_SIZE_M)
    return dict(
        a=a.contiguous(),
        b=b.contiguous(),
        num_experts=num_experts,
        num_tokens_per_expert=num_tokens_per_expert,
        expert_token_offs=expert_token_offs,
        expert_block_offs=expert_block_offs,
        expert_block_schedule=expert_block_schedule,
        check=_check,
    )


def _check(actual, expected):
    assert_close(actual, expected, atol=2e-2, rtol=2e-2)


_CASE_FACTORIES = [
    lambda: _case(128, 256, 512, num_experts=4),
    lambda: _case(1024, 512, 1024, num_experts=8),
    lambda: _case(333, 256, 256, num_experts=16),
] + [
    (lambda m, e: lambda: _case(m, 512, 2048, num_experts=e))(m, e)
    for m in (128, 1024, 4096)
    for e in (8, 32)
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.grouped_gemm
def test_grouped_gemm(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("grouped_gemm")
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
        from flaggems_sglang import grouped_gemm
    except (ImportError, ModuleNotFoundError):
        pytest.skip("gemm/grouped_gemm ops module not found")
        return

    try:
        actual = grouped_gemm(**kwargs)
    except NotImplementedError:
        pytest.skip("gemm/grouped_gemm not yet implemented")
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
