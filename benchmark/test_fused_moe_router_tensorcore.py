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

"""Benchmark for moe/fused_moe_router_tensorcore."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("fused_moe_router_tensorcore")

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
# Cases (from kernel-comp-baseline/problems/moe/fused_moe_router_tensorcore/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    a_w, a_ids = actual
    e_w, e_ids = expected
    assert torch.equal(a_ids.long(), e_ids.long()), "topk expert ids mismatch"
    assert_close(a_w, e_w, dtype=torch.float32)


def _case(
    bs,
    hidden_dim,
    num_experts,
    topk,
    softcap=0.0,
    has_bias=False,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    x = torch.randn(
        bs,
        hidden_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    router_weight = torch.randn(
        num_experts,
        hidden_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    bias = None
    if has_bias:
        bias = (
            torch.randn(
                num_experts,
                generator=g,
                device=flaggems_sglang.device,
                dtype=torch.float32,
            )
            * 0.1
        )
    return dict(
        x=x,
        router_weight=router_weight,
        topk=topk,
        moe_softcapping=softcap,
        correction_bias=bias,
        check=_check,
    )


# hidden_dim is a multiple of 64 (this problem's BLOCK_SIZE_K choice) and
# topk <= 2 (the tensorcore kernel's supported range).
CORRECTNESS_CASES = [
    _case(1, 64, 8, 1),
    _case(37, 256, 16, 2, softcap=30.0),
    _case(83, 512, 32, 2, has_bias=True),
]

BENCH_CASES = [_case(m, 4096, 256, 2) for m in (1, 8, 64, 512, 4096)]

BENCH_IDS = [f"m{m}" for m in (1, 8, 64, 512, 4096)]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.fused_moe_router_tensorcore
def test_fused_moe_router_tensorcore_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.fused_moe_router_tensorcore import (
            fused_moe_router_tensorcore,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/fused_moe_router_tensorcore ops module not found")
        return

    try:
        fused_moe_router_tensorcore(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/fused_moe_router_tensorcore not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: fused_moe_router_tensorcore(**kwargs))

    record_case(
        "moe/fused_moe_router_tensorcore",
        BENCH_IDS[case_idx],
        ref_us,
        triton_us,
    )
