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

"""Correctness test for kv_cache/concat_cast_kv_fp8_pad."""

import pytest
import torch

from flaggems_sglang.reference import get_reference

from . import conftest as cfg

FP8 = torch.float8_e4m3fn

# Poison byte: 0x7F is a valid (positive) e4m3 payload, so anything the kernel
# fails to write stays visible rather than aliasing to a zero.
POISON = 0x7F

NOPE = 512


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def _case(tokens, rope, pad_rows, head_nope=NOPE, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    head = head_nope + rope

    # k / k_rope arrive as [tokens, 1, width] views upstream (one KV head).
    k = (
        (torch.randn(tokens, head_nope, generator=g, device=cfg.device) * 0.5)
        .to(torch.bfloat16)
        .unsqueeze(1)
    )
    if rope > 0:
        # Build the rope segment by slicing a wider bf16 row, as the backend does
        # (it holds the fused [nope | rope] row and passes the tail view).
        wide = (
            torch.randn(tokens, 1, head, generator=g, device=cfg.device) * 0.5
        ).to(torch.bfloat16)
        k_rope = wide[..., head_nope:]
    else:
        k_rope = k.new_empty((tokens, 1, 0))

    # out is a slice of a larger buffer, so under- and over-writes both show.
    big = torch.empty(
        tokens + pad_rows + 50, head, dtype=FP8, device=cfg.device
    )
    _bytes(big).fill_(POISON)
    out = big[: tokens + pad_rows]

    return dict(
        out=out,
        k=k,
        k_rope=k_rope,
        num_tokens=tokens,
        _big=big,
        _head=head,
        _pad_rows=pad_rows,
    )


# (tokens, rope, pad_rows) — the first two mirror the upstream registered test's
# GLM-5.3-Flash geometry (rope 0 with a 2176-row band, rope 64 with 2048).
CASES = [
    (437, 0, 2176),
    (437, 64, 2048),
    (1, 0, 2176),
    (1, 64, 2048),
    (0, 0, 32),
    (437, 0, 0),
    (437, 64, 0),
    (7, 64, 3),
    (256, 64, 2048),
]


@pytest.mark.parametrize("case_idx", range(len(CASES)))
@pytest.mark.concat_cast_kv_fp8_pad
def test_concat_cast_kv_fp8_pad(case_idx):
    tokens, rope, pad_rows = CASES[case_idx]
    case = _case(tokens, rope, pad_rows)

    from flaggems_sglang import concat_cast_kv_fp8_pad

    out = case["out"]
    head = case["_head"]
    big = case["_big"]

    concat_cast_kv_fp8_pad(out, case["k"], case["k_rope"], tokens)

    # Real rows: byte-identical to a plain cat + cast.
    if tokens:
        expected = torch.cat(
            [
                case["k"].reshape(tokens, NOPE),
                case["k_rope"].reshape(tokens, rope),
            ],
            dim=-1,
        ).to(FP8)
        assert torch.equal(_bytes(out[:tokens]), _bytes(expected))

    # Pad band: must be zeros, and nothing beyond `out` may be touched.
    if pad_rows:
        assert (_bytes(out[tokens:]) == 0).all()
    if big.shape[0] > out.shape[0]:
        assert (_bytes(big[out.shape[0] :]) == POISON).all()

    # And the module-level reference must agree with the op.
    ref = get_reference("concat_cast_kv_fp8_pad")
    got_ref = ref(case["out"], case["k"], case["k_rope"], tokens)
    assert torch.equal(_bytes(out), _bytes(got_ref))

    assert out.shape == (tokens + pad_rows, head)


@pytest.mark.parametrize("case_idx", range(len(CASES)))
@pytest.mark.concat_cast_kv_fp8_pad
def test_reference_matches_torch_cat(case_idx):
    """The reference must itself agree with the obvious torch formulation."""
    tokens, rope, pad_rows = CASES[case_idx]
    case = _case(tokens, rope, pad_rows)
    ref = get_reference("concat_cast_kv_fp8_pad")
    got = ref(case["out"], case["k"], case["k_rope"], tokens)
    assert got.shape == (tokens + pad_rows, case["_head"])
    assert got.dtype == FP8
    if tokens:
        expected = torch.cat(
            [
                case["k"].reshape(tokens, NOPE),
                case["k_rope"].reshape(tokens, rope),
            ],
            dim=-1,
        ).to(FP8)
        assert torch.equal(_bytes(got[:tokens]), _bytes(expected))
    if pad_rows:
        assert (_bytes(got[tokens:]) == 0).all()


@pytest.mark.concat_cast_kv_fp8_pad
def test_rejects_wrong_dtype():
    """Only float8_e4m3fn destinations are accepted."""
    from flaggems_sglang import concat_cast_kv_fp8_pad

    k = torch.randn(4, 1, NOPE, dtype=torch.bfloat16, device=cfg.device)
    k_rope = k.new_empty((4, 1, 0))
    out = torch.zeros(4, NOPE, dtype=torch.bfloat16, device=cfg.device)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        concat_cast_kv_fp8_pad(out, k, k_rope, 4)


@pytest.mark.concat_cast_kv_fp8_pad
def test_rejects_width_mismatch():
    """out width must equal nope + rope."""
    from flaggems_sglang import concat_cast_kv_fp8_pad

    k = torch.randn(4, 1, NOPE, dtype=torch.bfloat16, device=cfg.device)
    k_rope = torch.randn(4, 1, 64, dtype=torch.bfloat16, device=cfg.device)
    out = torch.zeros(4, NOPE, dtype=FP8, device=cfg.device)  # missing rope
    with pytest.raises(ValueError, match="nope \\+ rope"):
        concat_cast_kv_fp8_pad(out, k, k_rope, 4)


@pytest.mark.concat_cast_kv_fp8_pad
def test_rejects_too_many_tokens():
    from flaggems_sglang import concat_cast_kv_fp8_pad

    k = torch.randn(8, 1, NOPE, dtype=torch.bfloat16, device=cfg.device)
    k_rope = k.new_empty((8, 1, 0))
    out = torch.zeros(4, NOPE, dtype=FP8, device=cfg.device)
    with pytest.raises(ValueError, match="exceeds total rows"):
        concat_cast_kv_fp8_pad(out, k, k_rope, 8)
