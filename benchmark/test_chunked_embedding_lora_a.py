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

"""Benchmark for lora/chunked_embedding_lora_a."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("chunked_embedding_lora_a")

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
# Cases (from kernel-comp-baseline/problems/lora/chunked_embedding_lora_a/cases.py)
# ---------------------------------------------------------------------------


from flaggems_sglang.reference._lora_batch_utils import make_batch_info


def _case(
    seg_lens,
    num_lora,
    r,
    vocab_size,
    permutation="identity",
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    s = sum(seg_lens)
    input_ids = torch.randint(
        0,
        vocab_size,
        (s,),
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.int64,
    )
    weights = torch.randn(
        num_lora,
        r,
        vocab_size,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    batch_info = make_batch_info(
        seg_lens,
        weight_indices,
        lora_ranks=[r] * num_lora,
        permutation=permutation,
    )
    return dict(
        input_ids=input_ids,
        weights=weights,
        batch_info=batch_info,
        vocab_size=vocab_size,
    )


CORRECTNESS_CASES = [
    _case([5], 1, 16, 128),
    _case([3, 7, 0, 12], 2, 32, 512),
    _case([9, 4], 2, 16, 256, permutation="shuffled"),
]

BENCH_CASES = [
    _case([512] * 8, 4, 32, 32000),
    _case([2048] * 4, 2, 64, 128256),
]

BENCH_IDS = ["seg512x8_r32_v32000", "seg2048x4_r64_v128256"]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.chunked_embedding_lora_a
def test_chunked_embedding_lora_a_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    case = case() if callable(case) else case
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.chunked_embedding_lora_a import (
            chunked_embedding_lora_a,
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("lora/chunked_embedding_lora_a ops module not found")
        return

    try:
        chunked_embedding_lora_a(**kwargs)
    except NotImplementedError:
        pytest.skip("lora/chunked_embedding_lora_a not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: chunked_embedding_lora_a(**kwargs))

    record_case(
        "lora/chunked_embedding_lora_a", BENCH_IDS[case_idx], ref_us, triton_us
    )
