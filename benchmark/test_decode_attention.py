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

"""Benchmark for attention/decode_attention."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("decode_attention")


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
# Cases (from kernel-comp-baseline/problems/attention/decode_attention/cases.py)
# ---------------------------------------------------------------------------


def _case(B, H_Q, H_KV, D, seq_len, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    total_tokens = B * seq_len
    sm_scale = 1.0 / (D**0.5)

    q = torch.randn(
        B,
        H_Q,
        D,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    k_buffer = torch.randn(
        total_tokens,
        H_KV,
        D,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    v_buffer = torch.randn(
        total_tokens,
        H_KV,
        D,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)

    b_seq_len = torch.full((B,), seq_len, device=flaggems_sglang.device)
    kv_indptr = torch.zeros(
        (B + 1,), dtype=torch.int32, device=flaggems_sglang.device
    )
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
    kv_indices = torch.arange(
        total_tokens, device=flaggems_sglang.device, dtype=torch.int32
    )

    return dict(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        sm_scale=sm_scale,
        check=_check,
    )


def _check(actual, expected):
    # bf16 split-KV accumulation reorders reductions vs. the single-pass
    # fp32 reference; a handful of elements near a softmax tie can exceed a
    # tight tolerance without indicating an actual bug (matches SGLang's own
    # decode-attention test looseness for larger configs).
    assert_close(actual.to(torch.float32), expected, atol=3e-2, rtol=1e-2)


CORRECTNESS_CASES = [
    _case(2, 4, 4, 64, 10),
    _case(2, 4, 2, 64, 10),
    _case(2, 4, 4, 80, 10),
    _case(2, 16, 1, 512, 128),
]

# Lazy (zero-arg callables): total KV-buffer size scales with B * seq_len,
# so cases are generated one at a time by harness.bench.run_bench_cases
# rather than all held in memory at once.
BENCH_CASES = [
    (lambda B=B, seq_len=seq_len: _case(B, 32, 8, 128, seq_len))
    for B, seq_len in (
        (1, 2048),
        (8, 2048),
        (64, 512),
        (512, 128),
        (4096, 128),
    )
]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.decode_attention
def test_decode_attention_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    case = case() if callable(case) else case
    kwargs = {k: v for k, v in case.items() if k != "check"}

    try:
        from flaggems_sglang.ops.decode_attention import decode_attention
    except (ImportError, ModuleNotFoundError):
        pytest.skip("attention/decode_attention ops module not found")
        return

    try:
        decode_attention(**kwargs)
    except NotImplementedError:
        pytest.skip("attention/decode_attention not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: decode_attention(**kwargs))
    record_case(
        "attention/decode_attention", f"case{case_idx}", ref_us, triton_us
    )
