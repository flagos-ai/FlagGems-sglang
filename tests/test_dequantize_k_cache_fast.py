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

"""Correctness test for kv_cache/dequantize_k_cache_fast."""

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


_DV = 512
_TILE = 128
_D = 576


def _quantize_upstream(x, dv=_DV, tile_size=_TILE):
    """Pack rows exactly the way upstream's quantizer does.

    Mirrors ``python/sglang/test/kernels/dsa_k_cache_reference.py``
    (2165 B), the only quantizer upstream ships for this layout. sglang is
    not a dependency of this repo, so the statements are reproduced here
    rather than imported; the row it builds is what the kernels under test
    are expected to read.

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


def _case(num_tokens, dim_nope, dim_rope, seed=0, block_size=None):
    packed = _pack(num_tokens, dim_nope, dim_rope, seed=seed)
    if block_size is not None:
        assert num_tokens % block_size == 0
        packed = packed.view(num_tokens // block_size, block_size, -1)
    return dict(
        quant_k_cache=packed,
        dim_nope=dim_nope,
        dim_rope=dim_rope,
    )


_CASE_FACTORIES = [
    lambda: _case(1, 512, 64),
    lambda: _case(37, 512, 64),
    lambda: _case(512, 512, 64),
    lambda: _case(2048, 512, 64),
    lambda: _case(129, 512, 64),
    lambda: _case(8, 256, 0),
    lambda: _case(8, 512, 0),
    lambda: _case(8, 128, 64),
    lambda: _case(8, 1024, 64),
    lambda: _case(8, 1024, 128),
    lambda: _case(64, 512, 64, block_size=8),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.dequantize_k_cache_fast
def test_dequantize_k_cache_fast(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("dequantize_k_cache_fast")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    expected = reference(**kwargs)

    try:
        candidate = get_candidate(
            "dequantize_k_cache_fast", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip("kv_cache/dequantize_k_cache_fast ops module not found")
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip("kv_cache/dequantize_k_cache_fast not yet implemented")
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
@pytest.mark.dequantize_k_cache_fast
def test_against_upstream_quantizer(num_rows):
    """The original bf16 is the oracle, not this repo's reference.

    Upstream ships only a quantizer for this layout, so this case has it
    write the packed rows and compares the kernel's output against the
    input it quantized -- never against ``reference/dequantize_k_cache_fast``.
    The RoPE tail is stored as bf16 and copied verbatim, so it must return
    bit-exact; misreading where that tail starts shifts it and fails
    immediately. The NoPE half goes through fp8 e4m3 (3 mantissa bits,
    half-ulp = 1/16 of the tile magnitude), so it is held to that format
    precision rather than to bit equality.
    """
    dev = flaggems_sglang.device
    g = torch.Generator(device=dev).manual_seed(num_rows)
    x = torch.randn((num_rows, 1, 1, _D), generator=g, device=dev).to(
        torch.bfloat16
    )

    packed = _quantize_upstream(x)
    candidate = get_candidate("dequantize_k_cache_fast")
    actual = candidate(
        packed.reshape(num_rows, -1),
        _DV,
        _D - _DV,
        _TILE,
    )
    # Tolerate both a row-major and a leading singleton dimension.
    actual = actual.reshape(num_rows, -1)

    want_rope = x.reshape(num_rows, _D)[:, _DV:].to(torch.float32)
    got_rope = actual[:, _DV:].to(torch.float32)
    assert torch.equal(got_rope, want_rope), (
        "RoPE tail must be bit-exact: it is copied verbatim as bf16, so a "
        "non-zero diff means the dequantizer reads the wrong tail offset"
    )

    want_nope = x.reshape(num_rows, _D)[:, :_DV].to(torch.float32)
    got_nope = actual[:, :_DV].to(torch.float32)
    scale_ref = want_nope.abs().max().item()
    rel = (got_nope - want_nope).abs().max().item() / scale_ref
    # fp8 e4m3 half-ulp is 1/16 of the tile magnitude, and the output also
    # rounds to bf16; 5% of the tensor max is the format's own precision.
    assert rel < 0.05, f"NoPE relative error {rel:.3%} exceeds fp8 precision"
