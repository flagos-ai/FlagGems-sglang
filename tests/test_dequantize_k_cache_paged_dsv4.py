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

"""Correctness test for kv_cache/dequantize_k_cache_paged_dsv4."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

_TOLERANCES = {
    torch.float32: dict(atol=1e-4, rtol=1e-4),
    torch.bfloat16: dict(atol=1.5e-2, rtol=1.5e-2),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
}
_DEFAULT_TOLERANCE = dict(atol=1e-2, rtol=1e-2)

DIM_NOPE = 448
DIM_ROPE = 64
NUM_SCALE_TILES = 7
NOPE_ROPE_BYTES = DIM_NOPE + DIM_ROPE * 2
PADDED_SCALE_PER_TOKEN = NUM_SCALE_TILES + 1


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype,
            _DEFAULT_TOLERANCE,
        )
    )
    tol.update(overrides)
    actual_f = (
        actual.to(torch.float32) if actual.dtype.is_floating_point else actual
    )
    expected_f = (
        expected.to(torch.float32)
        if expected.dtype.is_floating_point
        else expected
    )
    # torch.testing.assert_close defaults equal_nan to False here, so a cache
    # holding an fp8 NaN payload (byte 0x7F) fails even when the kernel and the
    # reference produce NaN at exactly the same positions. Upstream's test
    # passes equal_nan=True for the same reason.
    if expected_f.dtype.is_floating_point:
        tol.setdefault("equal_nan", True)
    torch.testing.assert_close(actual_f, expected_f, **tol)


def _pack(num_pages, page_size, seed=0):
    """Build a valid DeepSeek-V4 paged cache: data block then scale block."""
    dev = flaggems_sglang.device
    g = torch.Generator(device=dev).manual_seed(seed)
    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = ((raw + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES) * (
        NOPE_ROPE_BYTES
    )
    buf = torch.zeros(
        (num_pages, bytes_per_page), dtype=torch.uint8, device=dev
    )

    q = (
        torch.randn(
            (num_pages, page_size, DIM_NOPE),
            generator=g,
            device=dev,
            dtype=torch.float32,
        )
        * 5.0
    ).to(torch.float8_e4m3fn)
    r = torch.randn(
        (num_pages, page_size, DIM_ROPE),
        generator=g,
        device=dev,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    data = torch.cat(
        [
            q.view(torch.uint8).reshape(num_pages, page_size, DIM_NOPE),
            r.view(torch.uint8).reshape(num_pages, page_size, DIM_ROPE * 2),
        ],
        dim=2,
    ).reshape(num_pages, page_size * NOPE_ROPE_BYTES)
    buf[:, : page_size * NOPE_ROPE_BYTES] = data

    # ue8m0 bytes stay in 100..160 (2^-27 .. 2^33), the range real MX scales
    # live in. Extrema are covered by _random_cache instead: byte 0 exercises
    # the subnormal flush and byte 255 the NaN path, and both are exactly the
    # checks that caught the scale decode bug, so they belong in the suite
    # rather than being avoided.
    scales = torch.randint(
        100,
        161,
        (num_pages, page_size, NUM_SCALE_TILES),
        generator=g,
        device=dev,
        dtype=torch.uint8,
    )
    padded = torch.zeros(
        (num_pages, page_size, PADDED_SCALE_PER_TOKEN),
        dtype=torch.uint8,
        device=dev,
    )
    padded[:, :, :NUM_SCALE_TILES] = scales
    buf[:, page_size * NOPE_ROPE_BYTES :][
        :, : page_size * PADDED_SCALE_PER_TOKEN
    ] = padded.reshape(num_pages, -1)

    return buf


def _random_cache(num_pages, page_size, seed=0):
    """Cache filled with random bytes, mirroring the upstream registered test.

    Every fp8 payload byte and every ue8m0 scale byte is drawn from 0..255, so
    the scale range spans ``2 ** -127 .. 2 ** 128`` and byte 255 lands on NaN.
    The upstream test in
    ``sglang/test/registered/kernels/ops/attention/test_dsv4_dequant_k_cache.py``
    builds its cache exactly this way. Restricting scales to the "realistic"
    100..160 band (as ``_pack`` does) leaves the flush-to-zero path and the NaN
    path unexercised, so a reference that forgets to flush subnormal scales
    still passes.
    """
    dev = flaggems_sglang.device
    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = ((raw + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES) * (
        NOPE_ROPE_BYTES
    )
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.randint(
        0,
        256,
        (num_pages, bytes_per_page),
        generator=g,
        device=dev,
        dtype=torch.uint8,
    )


def _random_case(num_tokens, page_size, seed=0):
    dev = flaggems_sglang.device
    num_pages = max(2, (num_tokens + page_size - 1) // page_size + 1)
    g = torch.Generator(device=dev).manual_seed(seed)
    return dict(
        quant_k_cache=_random_cache(num_pages, page_size, seed),
        page_table_1_flattened=torch.randint(
            0,
            num_pages * page_size,
            (num_tokens,),
            generator=g,
            device=dev,
            dtype=torch.int32,
        ),
        page_size=page_size,
    )


def _page_table(num_pages, page_size, num_tokens, mode, seed=0):
    dev = flaggems_sglang.device
    g = torch.Generator(device=dev).manual_seed(seed)
    total = num_pages * page_size
    if mode == "sequential":
        index = torch.arange(num_tokens, device=dev)
    elif mode == "shuffled":
        index = torch.randperm(total, generator=g, device=dev)[:num_tokens]
    elif mode == "shared":
        index = torch.randint(total, (num_tokens,), generator=g, device=dev)
    else:
        raise ValueError(f"Unknown page table mode: {mode}")
    return index.to(torch.int32)


def _case(num_pages, page_size, num_tokens, mode, seed=0):
    return dict(
        quant_k_cache=_pack(num_pages, page_size, seed),
        page_table_1_flattened=_page_table(
            num_pages, page_size, num_tokens, mode, seed
        ),
        page_size=page_size,
    )


_CASE_FACTORIES = [
    lambda: _case(16, 1, 16, "sequential"),
    lambda: _case(4, 4, 16, "sequential"),
    lambda: _case(4, 4, 5, "sequential"),
    lambda: _case(4, 4, 16, "shuffled"),
    lambda: _case(2, 2, 16, "shared"),
    lambda: _case(8, 8, 64, "sequential"),
    lambda: _case(8, 16, 128, "shuffled"),
    lambda: _case(1, 1, 1, "sequential"),
    # Upstream's registered CASES: tails of 0/1/3 modulo the 4 tokens per
    # program, and both page sizes the cache pool actually uses.
    lambda: _random_case(1, 64),
    lambda: _random_case(3, 64),
    lambda: _random_case(333, 64),
    lambda: _random_case(256, 256),
    lambda: _random_case(4097, 256),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.dequantize_k_cache_paged_dsv4
def test_dequantize_k_cache_paged_dsv4(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("dequantize_k_cache_paged_dsv4")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        candidate = get_candidate(
            "dequantize_k_cache_paged_dsv4", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "kv_cache/dequantize_k_cache_paged_dsv4 ops module not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "kv_cache/dequantize_k_cache_paged_dsv4 not yet implemented"
        )
        return

    if check is not None:
        check(actual, expected)
    elif isinstance(expected, torch.Tensor):
        assert_close(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            if isinstance(e, torch.Tensor):
                assert_close(a, e)
    if check is not None:
        case["check"] = check
