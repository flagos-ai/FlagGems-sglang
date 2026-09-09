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

"""Benchmark for mamba/layernorm_gated."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("mamba_layernorm_gated")


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
# Cases (from kernel-comp-baseline/problems/mamba/layernorm_gated/cases.py)
# ---------------------------------------------------------------------------


def _case(
    m,
    n,
    group_size=None,
    has_z=True,
    norm_before_gate=True,
    is_rms_norm=True,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    x = torch.randn(
        m, n, generator=g, device=flaggems_sglang.device, dtype=torch.float32
    ).to(dtype)
    weight = torch.randn(
        n, generator=g, device=flaggems_sglang.device, dtype=torch.float32
    ).to(dtype)
    bias = torch.randn(
        n, generator=g, device=flaggems_sglang.device, dtype=torch.float32
    ).to(dtype)
    z = None
    if has_z:
        z = torch.randn(
            m,
            n,
            generator=g,
            device=flaggems_sglang.device,
            dtype=torch.float32,
        ).to(dtype)
    return dict(
        x=x,
        weight=weight,
        bias=bias,
        eps=1e-5,
        z=z,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=is_rms_norm,
    )


CORRECTNESS_CASES = [
    _case(1, 64),
    _case(37, 256, group_size=64),
    _case(83, 512, group_size=128, norm_before_gate=False),
    _case(4, 1024, is_rms_norm=False, has_z=False),
]

BENCH_CASES = [_case(m, 4096, group_size=128) for m in (1, 8, 64, 512, 4096)]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.mamba_layernorm_gated
def test_mamba_layernorm_gated_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.mamba_layernorm_gated import (
            mamba_layernorm_gated,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("mamba/layernorm_gated ops module not found")
        return

    try:
        mamba_layernorm_gated(**kwargs)
    except NotImplementedError:
        pytest.skip("mamba/layernorm_gated not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: mamba_layernorm_gated(**kwargs))
    record_case("mamba/layernorm_gated", f"case{case_idx}", ref_us, triton_us)
