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

"""Benchmark for speculative/fused_norm_rope_stacked."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("fused_norm_rope_stacked")

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
        equal_nan=True,
        **tol,
    )


# ---------------------------------------------------------------------------
# Cases (from kernel-comp-baseline/problems/speculative/fused_norm_rope_stacked/cases.py)
# ---------------------------------------------------------------------------


def _check(actual, expected):
    a_k, a_v = actual
    e_k, e_v = expected
    assert_close(a_k, e_k)
    assert_close(a_v, e_v)


def _case(
    t,
    n_layers,
    num_kv_heads,
    head_dim,
    rotary_dim,
    max_pos=4096,
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    kv_size = num_kv_heads * head_dim
    kv = torch.randn(
        t,
        n_layers,
        2 * kv_size,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    k_norm_weight = torch.randn(
        n_layers,
        head_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    )
    eps = torch.full(
        (n_layers,), 1e-6, device=flaggems_sglang.device, dtype=torch.float32
    )
    cos_sin_cache = torch.randn(
        max_pos,
        rotary_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    positions = torch.randint(
        0,
        max_pos,
        (t,),
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.int64,
    )
    return dict(
        kv=kv,
        k_norm_weight=k_norm_weight,
        eps=eps,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        check=_check,
    )


CORRECTNESS_CASES = [
    _case(1, 2, 2, 64, 64),
    _case(37, 3, 4, 128, 128),
    _case(129, 2, 2, 128, 64),
]

BENCH_CASES = [_case(t, 8, 8, 128, 128) for t in (1, 128, 2048, 8192)]

BENCH_IDS = [f"t{t}" for t in (1, 128, 2048, 8192)]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.fused_norm_rope_stacked
def test_fused_norm_rope_stacked_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    case = case() if callable(case) else case
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.fused_norm_rope_stacked import (
            fused_norm_rope_stacked,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("rope/fused_norm_rope_stacked ops module not found")
        return

    try:
        fused_norm_rope_stacked(**kwargs)
    except NotImplementedError:
        pytest.skip("rope/fused_norm_rope_stacked not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: fused_norm_rope_stacked(**kwargs))

    record_case(
        "rope/fused_norm_rope_stacked", BENCH_IDS[case_idx], ref_us, triton_us
    )
