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

"""Benchmark for lora/chunked_sgmv_expand."""

import pytest
import torch

import flaggems_sglang
from benchmark.bench_report import do_bench_us, record_case
from flaggems_sglang.reference import get_reference

reference = get_reference("chunked_sgmv_expand")

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
# Cases (from kernel-comp-baseline/problems/lora/chunked_sgmv_expand/cases.py)
# ---------------------------------------------------------------------------


from flaggems_sglang.reference._lora_batch_utils import make_batch_info


def _case(
    seg_lens,
    num_lora,
    r,
    slice_sizes,
    permutation="identity",
    dtype=torch.bfloat16,
    seed=0,
):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    s = sum(seg_lens)
    n_slices = len(slice_sizes)
    output_dim = sum(slice_sizes)
    max_slice_size = max(slice_sizes)

    offsets = [0]
    for sz in slice_sizes:
        offsets.append(offsets[-1] + sz)
    slice_offsets = torch.tensor(
        offsets, dtype=torch.int32, device=flaggems_sglang.device
    )

    x = torch.randn(
        s,
        n_slices * r,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    weights = torch.randn(
        num_lora,
        output_dim,
        r,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    base_output = torch.randn(
        s,
        output_dim,
        generator=g,
        device=flaggems_sglang.device,
        dtype=torch.float32,
    ).to(dtype)
    weight_indices = [i % num_lora for i in range(len(seg_lens))]
    scalings = [0.5 + 0.25 * i for i in range(num_lora)]
    batch_info = make_batch_info(
        seg_lens,
        weight_indices,
        lora_ranks=[r] * num_lora,
        scalings=scalings,
        permutation=permutation,
    )
    return dict(
        x=x,
        weights=weights,
        batch_info=batch_info,
        slice_offsets=slice_offsets,
        max_slice_size=max_slice_size,
        base_output=base_output,
    )


# BLOCK_M == max(seg_lens) is used directly as a Triton arange size, so the
# largest segment in each case must be a power of 2.
CORRECTNESS_CASES = [
    _case([8], 1, 16, [64]),
    _case([3, 7, 0, 16], 2, 16, [128, 64]),
    _case([16, 4], 2, 32, [256, 128, 128], permutation="shuffled"),
]

BENCH_CASES = [
    _case([64] * 8, 4, 32, [4096, 4096]),
    _case([256] * 4, 2, 64, [4096, 1024, 1024]),
]

BENCH_IDS = ["seg64x8_r32", "seg256x4_r64"]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(BENCH_CASES)))
@pytest.mark.chunked_sgmv_expand
def test_chunked_sgmv_expand_perf(case_idx):
    """Benchmark triton kernel vs torch reference; record per-case speedup."""
    case = BENCH_CASES[case_idx]
    case = case() if callable(case) else case
    kwargs = (
        {k: v for k, v in case.items() if k != "check"}
        if isinstance(case, dict)
        else case
    )

    try:
        from flaggems_sglang.ops.chunked_sgmv_expand import chunked_sgmv_expand
    except (ImportError, ModuleNotFoundError):
        pytest.skip("lora/chunked_sgmv_expand ops module not found")
        return

    try:
        chunked_sgmv_expand(**kwargs)
    except NotImplementedError:
        pytest.skip("lora/chunked_sgmv_expand not yet implemented")
        return

    ref_us = do_bench_us(lambda: reference(**kwargs))
    triton_us = do_bench_us(lambda: chunked_sgmv_expand(**kwargs))

    record_case(
        "lora/chunked_sgmv_expand", BENCH_IDS[case_idx], ref_us, triton_us
    )
