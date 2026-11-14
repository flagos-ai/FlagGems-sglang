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

"""Correctness test for kv_cache/dequantize_k_cache_paged."""

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


def assert_close(actual, expected, *, dtype=None, **overrides):
    tol = dict(
        _TOLERANCES.get(
            dtype if dtype is not None else expected.dtype,
            _DEFAULT_TOLERANCE,
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


def _pack(num_rows, dim_nope, dim_rope, group_size=128, seed=0):
    """Build a packed row buffer with a valid fp8 / scale / rope layout."""
    dev = flaggems_sglang.device
    g = torch.Generator(device=dev).manual_seed(seed)
    num_tiles = dim_nope // group_size
    dim_quant = dim_nope + num_tiles * 4 + dim_rope * 2
    buf = torch.empty((num_rows, dim_quant), dtype=torch.uint8, device=dev)

    q = (
        torch.randn(
            (num_rows, dim_nope),
            generator=g,
            device=dev,
            dtype=torch.float32,
        )
        * 50.0
    ).to(torch.float8_e4m3fn)
    buf[:, :dim_nope] = q.view(torch.uint8)

    scale = (
        torch.rand(
            (num_rows, num_tiles),
            generator=g,
            device=dev,
            dtype=torch.float32,
        )
        * 0.02
        + 0.005
    )
    buf[:, dim_nope : dim_nope + num_tiles * 4] = scale.contiguous().view(
        torch.uint8
    )

    rope = torch.randn(
        (num_rows, dim_rope),
        generator=g,
        device=dev,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    buf[:, dim_nope + num_tiles * 4 :] = rope.view(torch.uint8)

    return buf.view(torch.float8_e4m3fn)


def _page_table(num_tokens, num_rows, mode, seed=0):
    g = torch.Generator(device=flaggems_sglang.device).manual_seed(seed)
    if mode == "sequential":
        index = torch.arange(num_tokens, device=flaggems_sglang.device)
    elif mode == "shuffled":
        index = torch.randperm(
            num_rows, generator=g, device=flaggems_sglang.device
        )[:num_tokens]
    elif mode == "shared":
        index = torch.randint(
            num_rows,
            (num_tokens,),
            generator=g,
            device=flaggems_sglang.device,
        )
    else:
        raise ValueError(f"Unknown page table mode: {mode}")
    return index.to(torch.int32)


_DV = 512
_TILE = 128
_D = 576


def _quantize_upstream(x, dv=_DV, tile_size=_TILE):
    """Pack rows exactly the way upstream's quantizer does.

    Mirrors ``python/sglang/test/kernels/dsa_k_cache_reference.py``
    (2165 B), the only quantizer upstream ships for this layout. sglang is
    not a dependency of this repo, so the statements are reproduced here
    rather than imported.

    Args:
        x: ``bfloat16`` tensor of shape ``(blocks, block_size, 1, d)``.

    Returns:
        ``float8_e4m3fn`` tensor of shape ``(blocks, block_size, 1, 656)``
        laid out as ``[dv fp8 nope | dv/tile_size fp32 scales | bf16 rope]``.
    """
    assert dv % tile_size == 0
    num_tiles = dv // tile_size
    blocks, block_size, _, d = x.shape
    flat = x.squeeze(2)
    dim_quant = dv + num_tiles * 4 + flat.dtype.itemsize * (d - dv)
    result = torch.empty(
        (blocks, block_size, dim_quant),
        dtype=torch.float8_e4m3fn,
        device=flat.device,
    )
    nope = result[..., :dv]
    scales = result[..., dv : dv + num_tiles * 4].view(torch.float32)
    rope = result[..., dv + num_tiles * 4 :].view(flat.dtype)
    rope[:] = flat[..., dv:]

    for tile in range(num_tiles):
        sl = slice(tile * tile_size, (tile + 1) * tile_size)
        inv = flat[..., sl].abs().max(dim=-1).values / 448.0
        scales[..., tile] = inv
        nope[..., sl] = (flat[..., sl].float() / inv.float().unsqueeze(-1)).to(
            torch.float8_e4m3fn
        )

    return result.view(blocks, block_size, 1, dim_quant)


def _case(
    num_rows, num_tokens, dim_nope, dim_rope, mode, seed=0, block_size=None
):
    packed = _pack(num_rows, dim_nope, dim_rope, seed=seed)
    if block_size is not None:
        assert num_rows % block_size == 0
        packed = packed.view(num_rows // block_size, block_size, 1, -1)
    else:
        packed = packed.view(num_rows, 1, -1)
    return dict(
        quant_k_cache=packed,
        page_table_1_flattened=_page_table(num_tokens, num_rows, mode, seed),
    )


_CASE_FACTORIES = [
    lambda: _case(16, 16, 512, 64, "sequential"),
    lambda: _case(16, 5, 512, 64, "sequential"),
    lambda: _case(16, 16, 512, 64, "shuffled"),
    lambda: _case(8, 16, 512, 64, "shared"),
    lambda: _case(2048, 2048, 512, 64, "sequential"),
    lambda: _case(64, 64, 256, 0, "sequential"),
    # dim_rope is inferred, and the packed layout only resolves 0 or 64,
    # so this case keeps 64 instead of 128.
    lambda: _case(64, 64, 1024, 64, "shuffled"),
    lambda: _case(32, 16, 512, 64, "sequential", block_size=4),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.dequantize_k_cache_paged
def test_dequantize_k_cache_paged(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("dequantize_k_cache_paged")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        candidate = get_candidate(
            "dequantize_k_cache_paged", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("kv_cache/dequantize_k_cache_paged ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("kv_cache/dequantize_k_cache_paged not yet implemented")
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


@pytest.mark.parametrize("num_rows", [8, 64, 512])
@pytest.mark.dequantize_k_cache_paged
def test_paged_matches_direct_rows(num_rows):
    """Paged addressing must not change the decode.

    The same upstream-packed rows go through the paged kernel (with an
    identity page table) and the direct-row kernel. Comparing them to each
    other rather than to this repo's reference keeps the oracle out of the
    pair: any disagreement is a real defect in the addressing rewrite, and
    bit equality is the right bar because neither side performs arithmetic
    the other does not.
    """
    dev = flaggems_sglang.device
    g = torch.Generator(device=dev).manual_seed(num_rows)
    x = torch.randn((num_rows, 1, 1, _D), generator=g, device=dev).to(
        torch.bfloat16
    )
    packed = _quantize_upstream(x)
    rows = packed.reshape(num_rows, -1)

    paged = get_candidate("dequantize_k_cache_paged")
    direct = get_candidate("dequantize_k_cache_fast")
    table = torch.arange(num_rows, dtype=torch.int32, device=dev)

    got_paged = paged(rows, table, _TILE).reshape(num_rows, -1)
    got_direct = direct(rows, _DV, _D - _DV, _TILE).reshape(num_rows, -1)

    assert got_paged.dtype == torch.bfloat16
    assert torch.equal(got_paged, got_direct), (
        "paged and direct-row decoders disagree on the same packed rows; "
        "the page table only selects a row, so the decode must be identical"
    )
