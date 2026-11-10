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

"""Correctness test for attention/sglang_fl_mrotary_embedding."""

import copy

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference, get_tolerance
from flaggems_sglang.reference_candidate import get_candidate

OP = "sglang_fl_mrotary_embedding"

_KEYS = (
    "q",
    "k",
    "cos_sin_cache",
    "positions",
    "mrope_section",
    "head_size",
    "rotary_dim",
    "mrope_interleaved",
    "mrope_interleaved_glm",
    "is_neox_style",
    "axis_map",
)


def _build_axis_map(mrope_section, rotary_dim, interleaved, glm):
    """Prebuild the per-lane axis map the way the upstream layer does."""
    num_pairs = rotary_dim // 2
    section = list(mrope_section)
    if glm:
        lanes = []
        spent = [0, 0, 0]
        for lane in range(num_pairs):
            axis = lane % 3
            while spent[axis] >= section[axis]:
                axis = (axis + 1) % 3
            spent[axis] += 1
            lanes.append(axis)
        return torch.tensor(lanes, dtype=torch.long)
    return None


def _cos_sin_cache(max_pos, rotary_dim, dtype, device, base=100000.0):
    """Same construction the module uses: cos block then sin block."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, device=device).float()
            / float(rotary_dim)
        )
    )
    positions = torch.arange(max_pos, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def _case(
    num_tokens,
    head_size,
    rotary_dim,
    mrope_section,
    num_q_heads,
    num_k_heads,
    dtype=torch.bfloat16,
    interleaved=False,
    glm=False,
    neox=True,
    seed=0,
    max_pos=4096,
):
    device = flaggems_sglang.device
    g = torch.Generator(device=device).manual_seed(seed)

    q = torch.randn(
        num_tokens,
        num_q_heads * head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    k = torch.randn(
        num_tokens,
        num_k_heads * head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(dtype)

    cos_sin_cache = _cos_sin_cache(max_pos, rotary_dim, dtype, device)
    positions = torch.randint(
        0,
        max_pos,
        (3, num_tokens),
        generator=g,
        device=device,
        dtype=torch.int64,
    )

    axis_map = _build_axis_map(mrope_section, rotary_dim, interleaved, glm)
    if axis_map is not None:
        axis_map = axis_map.to(device)

    case = dict(
        q=q,
        k=k,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
        mrope_section=list(mrope_section),
        head_size=head_size,
        rotary_dim=rotary_dim,
        mrope_interleaved=interleaved,
        mrope_interleaved_glm=glm,
        is_neox_style=neox,
        axis_map=axis_map,
    )
    case["check"] = _make_check(case, dtype)
    return case


def _make_check(case, dtype):
    head_size = case["head_size"]
    rotary_dim = case["rotary_dim"]
    tol = get_tolerance(OP)[
        dtype if dtype in get_tolerance(OP) else torch.float32
    ]
    orig_q = case["q"].clone()
    orig_k = case["k"].clone()

    def _check(actual, expected):
        for name, orig in (("q", orig_q), ("k", orig_k)):
            if not torch.allclose(actual[name], expected[name], **tol):
                worst = (
                    (actual[name].float() - expected[name].float()).abs().max()
                )
                raise AssertionError(
                    f"{name} mismatch, max abs diff {worst.item():.3e} "
                    f"(atol {tol['atol']}, rtol {tol['rtol']})"
                )
            # Channels past rotary_dim take no part in the rotation, so the
            # in-place contract must leave them byte for byte unchanged. The
            # band has to be sliced per head: on the flat (tokens, heads *
            # head_size) layout, ``[..., rotary_dim:]`` would cut across
            # heads instead of inside them.
            if rotary_dim < head_size:

                def tail(tensor):
                    return tensor.view(tensor.shape[0], -1, head_size)[
                        ..., rotary_dim:
                    ]

                assert torch.equal(
                    tail(actual[name]), tail(orig)
                ), f"{name} tail beyond rotary_dim was modified"

    return _check


# The contract's own geometry: head_size 256, rotary_dim 64, sections
# 11/11/10, Neox pairing -- three mask strategies times two pairings.
_CASES = [
    # 0 segmented axes, Neox -- the contract geometry.
    lambda: _case(128, 256, 64, [11, 11, 10], 32, 8, seed=0),
    # 1 segmented axes, GPT-J pairing.
    lambda: _case(128, 256, 64, [11, 11, 10], 32, 8, neox=False, seed=1),
    # 2 interleaved axes (every third lane), Neox.
    lambda: _case(128, 256, 64, [11, 11, 10], 32, 8, interleaved=True, seed=2),
    # 3 interleaved axes, GPT-J pairing.
    lambda: _case(
        128,
        256,
        64,
        [11, 11, 10],
        32,
        8,
        interleaved=True,
        neox=False,
        seed=3,
    ),
    # 4 GLM interleaved, driven by the prebuilt axis map.
    lambda: _case(
        128, 256, 64, [11, 11, 10], 32, 8, interleaved=True, glm=True, seed=4
    ),
    # 5 rotary_dim == head_size: no pass-through band at all.
    lambda: _case(64, 64, 64, [11, 11, 10], 8, 8, seed=5),
    # 6 head count not a power of two: 12 heads pads to 16.
    lambda: _case(96, 128, 32, [6, 6, 4], 12, 3, seed=6),
    # 7 a single token, smallest grid the kernel can launch.
    lambda: _case(1, 256, 64, [11, 11, 10], 32, 8, seed=7),
]


_DTYPE_FACTORIES = [
    lambda: _case(
        64, 256, 64, [11, 11, 10], 32, 8, dtype=torch.float16, seed=8
    ),
    lambda: _case(
        64, 256, 64, [11, 11, 10], 32, 8, dtype=torch.float32, seed=9
    ),
    lambda: _case(
        64,
        256,
        64,
        [11, 11, 10],
        32,
        8,
        dtype=torch.bfloat16,
        interleaved=True,
        glm=True,
        neox=False,
        seed=10,
    ),
]

CORRECTNESS_CASES = list(range(len(_CASES) + len(_DTYPE_FACTORIES)))


def _build(idx):
    if idx < len(_CASES):
        return _CASES[idx]()
    return _DTYPE_FACTORIES[idx - len(_CASES)]()


def _clone(case):
    out = {}
    for key in _KEYS:
        value = case[key]
        if torch.is_tensor(value):
            out[key] = value.clone()
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@pytest.mark.parametrize("case_idx", CORRECTNESS_CASES)
@pytest.mark.sglang_fl_mrotary_embedding
def test_sglang_fl_mrotary_embedding(case_idx):
    case = _build(case_idx)
    check = case["check"]
    reference = get_reference(OP)

    # Both sides mutate q and k in place, so each needs its own copy of
    # byte-identical inputs.
    expected = _clone(case)
    actual = _clone(case)

    reference(**expected)

    try:
        candidate = get_candidate(OP, reference=reference)
    except (ImportError, ModuleNotFoundError):
        pytest.skip(f"attention/{OP} ops module not found")
        return

    try:
        candidate(**actual)
    except NotImplementedError:
        pytest.skip(f"attention/{OP} not yet implemented")
        return

    check(actual, expected)


@pytest.mark.sglang_fl_mrotary_embedding
def test_sglang_fl_mrotary_embedding_strided_positions():
    """positions may be a strided view, not a fresh contiguous tensor."""
    device = flaggems_sglang.device
    op = flaggems_sglang.sglang_fl_mrotary_embedding
    reference = get_reference(OP)

    num_tokens = 40
    wide = torch.randint(
        0,
        4096,
        (3, num_tokens * 2),
        generator=torch.Generator(device=device).manual_seed(21),
        device=device,
        dtype=torch.int64,
    )
    positions = wide[:, ::2]
    assert positions.stride(0) > num_tokens

    def build():
        g = torch.Generator(device=device).manual_seed(22)
        q = torch.randn(
            num_tokens,
            32 * 256,
            generator=g,
            device=device,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        k = torch.randn(
            num_tokens,
            8 * 256,
            generator=g,
            device=device,
            dtype=torch.float32,
        ).to(torch.bfloat16)
        return q, k

    cache = _cos_sin_cache(4096, 64, torch.bfloat16, device)
    section = [11, 11, 10]

    q_ref, k_ref = build()
    q_act, k_act = build()
    reference(
        q_ref,
        k_ref,
        cache,
        positions,
        section,
        256,
        64,
        False,
        False,
        True,
        None,
    )
    op(q_act, k_act, cache, positions, section, 256, 64, False, False, True)
    tol = get_tolerance(OP)[torch.bfloat16]
    assert torch.allclose(q_act, q_ref, **tol)
    assert torch.allclose(k_act, k_ref, **tol)


@pytest.mark.sglang_fl_mrotary_embedding
def test_sglang_fl_mrotary_embedding_empty_tokens():
    """Zero tokens launches nothing and is not an error."""
    device = flaggems_sglang.device
    op = flaggems_sglang.sglang_fl_mrotary_embedding

    q = torch.zeros(0, 32 * 256, dtype=torch.bfloat16, device=device)
    k = torch.zeros(0, 8 * 256, dtype=torch.bfloat16, device=device)
    cache = _cos_sin_cache(64, 64, torch.bfloat16, device)
    positions = torch.zeros(3, 0, dtype=torch.int64, device=device)

    assert (
        op(q, k, cache, positions, [11, 11, 10], 256, 64, False, False, True)
        is None
    )
    assert q.shape == (0, 32 * 256)


@pytest.mark.sglang_fl_mrotary_embedding
def test_sglang_fl_mrotary_embedding_axis_map_agrees():
    """The reference refuses a supplied map that contradicts the sections."""
    device = flaggems_sglang.device
    reference = get_reference(OP)

    cache = _cos_sin_cache(64, 64, torch.bfloat16, device)
    q = torch.zeros(4, 32 * 256, dtype=torch.bfloat16, device=device)
    k = torch.zeros(4, 8 * 256, dtype=torch.bfloat16, device=device)
    positions = torch.zeros(3, 4, dtype=torch.int64, device=device)

    bad = torch.zeros(32, dtype=torch.long, device=device)
    with pytest.raises(AssertionError):
        reference(
            q,
            k,
            cache,
            positions,
            [11, 11, 10],
            256,
            64,
            True,
            True,
            True,
            bad,
        )


@pytest.mark.sglang_fl_mrotary_embedding
@pytest.mark.parametrize(
    "mutation",
    [
        "bad_ndim",
        "token_mismatch",
        "positions_shape",
        "head_not_divisible",
        "rotary_odd",
        "rotary_too_wide",
        "cache_width",
        "section_length",
        "section_sum",
        "axis_map_missing",
        "axis_map_size",
        "positions_dtype",
    ],
)
def test_sglang_fl_mrotary_embedding_rejects(mutation):
    device = flaggems_sglang.device
    op = flaggems_sglang.sglang_fl_mrotary_embedding

    q = torch.zeros(4, 32 * 256, dtype=torch.bfloat16, device=device)
    k = torch.zeros(4, 8 * 256, dtype=torch.bfloat16, device=device)
    cache = _cos_sin_cache(64, 64, torch.bfloat16, device)
    positions = torch.zeros(3, 4, dtype=torch.int64, device=device)
    section = [11, 11, 10]

    args = dict(
        q=q,
        k=k,
        cos_sin_cache=cache,
        positions=positions,
        mrope_section=list(section),
        head_size=256,
        rotary_dim=64,
        mrope_interleaved=False,
        mrope_interleaved_glm=False,
        is_neox_style=True,
        axis_map=None,
    )

    if mutation == "bad_ndim":
        args["q"] = q[None]
    elif mutation == "token_mismatch":
        args["k"] = k[:2]
    elif mutation == "positions_shape":
        args["positions"] = positions[:2]
    elif mutation == "head_not_divisible":
        args["head_size"] = 100
    elif mutation == "rotary_odd":
        args["rotary_dim"] = 63
    elif mutation == "rotary_too_wide":
        args["rotary_dim"] = 512
    elif mutation == "cache_width":
        args["cos_sin_cache"] = cache[:, :32].contiguous()
    elif mutation == "section_length":
        args["mrope_section"] = [11, 11]
    elif mutation == "section_sum":
        args["mrope_section"] = [11, 11, 11]
    elif mutation == "axis_map_missing":
        args["mrope_interleaved"] = True
        args["mrope_interleaved_glm"] = True
        args["axis_map"] = None
    elif mutation == "axis_map_size":
        args["mrope_interleaved"] = True
        args["mrope_interleaved_glm"] = True
        args["axis_map"] = torch.zeros(16, dtype=torch.long, device=device)
    else:
        args["positions"] = positions.to(torch.float32)

    with pytest.raises(ValueError):
        op(**args)
