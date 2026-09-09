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

"""Benchmark for moe/moe_fused_mul_sum."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("moe_fused_mul_sum")

device = flaggems_sglang.device


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
# Cases (from kernel-comp-baseline/problems/moe/moe_fused_mul_sum/cases.py)
# ---------------------------------------------------------------------------


def _case(
    num_tokens,
    top_k,
    size,
    is_ep=False,
    use_expert_map=False,
    num_experts=8,
    routed_scaling_factor=None,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    inputs = torch.randn(
        num_tokens,
        top_k,
        size,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)

    topk_ids = None
    expert_map = None
    if use_expert_map or is_ep:
        topk_ids = torch.randint(
            0,
            num_experts,
            (num_tokens, top_k),
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.int32,
        )
        if is_ep and not use_expert_map:
            # is_ep (no expert_map): the kernel checks `id_val >= 0` directly,
            # so -1 sentinels in topk_ids are a valid "already dropped" marker.
            drop = (
                torch.rand(
                    num_tokens,
                    top_k,
                    generator=g,
                    device=flaggems_sglang.device,
                )
                < 0.3
            )
            topk_ids = torch.where(
                drop, torch.full_like(topk_ids, -1), topk_ids
            )
    if use_expert_map:
        # has_expert_map path indexes `expert_map[topk_ids]` with no id_val>=0
        # guard, so topk_ids must stay valid (dropping is expressed entirely
        # via expert_map's own -1 entries, never via a -1 topk_id).
        expert_map = torch.arange(
            num_experts, device=flaggems_sglang.device, dtype=torch.int32
        )
        expert_map[num_experts // 2 :] = -1

    return dict(
        inputs=inputs,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
        routed_scaling_factor=routed_scaling_factor,
        is_ep=is_ep,
        check=assert_close,
    )


CORRECTNESS_CASES = [
    _case(1, 2, 128),
    _case(37, 4, 256, routed_scaling_factor=2.5),
    _case(83, 6, 512, is_ep=True, num_experts=16),
    _case(64, 4, 256, use_expert_map=True, num_experts=16),
]

BENCH_CASES = [_case(m, 8, 4096) for m in (1, 8, 64, 512, 4096)]

BENCH_IDS = [f"m{m}" for m in (1, 8, 64, 512, 4096)]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.moe_fused_mul_sum
def test_moe_fused_mul_sum_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.moe_fused_mul_sum import moe_fused_mul_sum
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/moe_fused_mul_sum ops module not found")
        return

    try:
        moe_fused_mul_sum(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/moe_fused_mul_sum not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: moe_fused_mul_sum(**kwargs))

    record_case(
        "moe/moe_fused_mul_sum", BENCH_IDS[case_idx], ref_us, triton_us
    )
