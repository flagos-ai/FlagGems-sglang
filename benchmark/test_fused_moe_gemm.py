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

"""Benchmark for moe/fused_moe_gemm."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("fused_moe_gemm")


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
# Cases (from kernel-comp-baseline/problems/moe/fused_moe_gemm/cases.py)
# ---------------------------------------------------------------------------


def _case(T, E, N, K, top_k, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    A = torch.randn(
        T, K, generator=g, device=flaggems_sglang.device, dtype=torch.float32
    ).to(dtype)
    B = torch.randn(
        E,
        N,
        K,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    topk_ids = torch.randint(
        0,
        E,
        (T, top_k),
        dtype=torch.int32,
        device=flaggems_sglang.device,
        generator=g,
    )
    topk_weights = torch.rand(
        T,
        top_k,
        device=flaggems_sglang.device,
        generator=g,
        dtype=torch.float32,
    )

    return dict(
        A=A,
        B=B,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        top_k=top_k,
        check=_check,
    )


def _check(actual, expected):
    assert_close(actual, expected, atol=0.5, rtol=1e-2)


CORRECTNESS_CASES = [
    _case(8, 4, 64, 128, 2),
    _case(17, 8, 128, 256, 2),
    _case(5, 4, 64, 128, 1),
]

BENCH_CASES = [_case(t, 8, 4096, 4096, 2) for t in (1, 8, 64, 512, 4096)]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.fused_moe_gemm
def test_fused_moe_gemm_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.fused_moe_gemm import fused_moe_gemm
    except (ImportError, ModuleNotFoundError):
        pytest.skip("moe/fused_moe_gemm ops module not found")
        return

    try:
        fused_moe_gemm(**kwargs)
    except NotImplementedError:
        pytest.skip("moe/fused_moe_gemm not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: fused_moe_gemm(**kwargs))
    record_case("moe/fused_moe_gemm", f"case{case_idx}", ref_us, triton_us)
