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

"""Native SGLang collection workload for ernie45_rope_fused."""

import torch

OP_NAME = "ernie45_rope_fused"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (0, 1)

SHAPES = [(t, 8, 2, 128, 128) for t in (1, 128, 2048, 8192)]
MORE_SHAPES = [(1, 4, 1, 64, 64), (37, 8, 2, 128, 128), (129, 16, 2, 128, 64)]
_MAX_POS = 4096


def native():
    """解析 triton_ernie45_rope_fused_inplace。"""
    try:
        from sglang.kernels.ops.attention.rotary_triton import (
            triton_ernie45_rope_fused_inplace as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "n_qh": n_qh,
            "n_kh": n_kh,
            "head_size": head_size,
            "rotary_dim": rotary_dim,
            **{
                "mrope_section": (
                    rotary_dim // 8,
                    rotary_dim // 8,
                    rotary_dim // 2 - 2 * (rotary_dim // 8),
                )
            },
        }
        for (num_tokens, n_qh, n_kh, head_size, rotary_dim) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, n_qh, n_kh, head_size, rotary_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    # section_h == section_w == rotary_dim // 8, section_t takes the rest of
    # rotary_dim // 2.
    section_h = rotary_dim // 8
    mrope_section = [
        section_h,
        section_h,
        rotary_dim // 2 - 2 * section_h,
    ]
    q = torch.randn(
        num_tokens,
        n_qh * head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    k = torch.randn(
        num_tokens,
        n_kh * head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    cos_sin_cache = torch.randn(
        _MAX_POS,
        rotary_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    positions = torch.randint(
        0,
        _MAX_POS,
        (3, num_tokens),
        generator=g,
        device=device,
        dtype=torch.int64,
    )
    return q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["n_qh"],
        binding["n_kh"],
        binding["head_size"],
        binding["rotary_dim"],
    )
    return build_case(shape, dtype, device), {"is_neox_style": True}


def key_shape(binding):
    return (
        "nt{num_tokens}_qh{n_qh}_kh{n_kh}_hs{head_size}_rd{rotary_dim}".format(
            **binding
        )
    )


def config(binding, dtype):
    nt = binding["num_tokens"]
    n_qh, n_kh = binding["n_qh"], binding["n_kh"]
    head_size, rotary_dim = binding["head_size"], binding["rotary_dim"]
    mrope_section = binding["mrope_section"]
    dt = str(dtype)
    return {
        "inputs": {
            "q": {
                "shape": [nt, n_qh * head_size],
                "dtype": dt,
                "note": "原地写回",
            },
            "k": {
                "shape": [nt, n_kh * head_size],
                "dtype": dt,
                "note": "原地写回",
            },
            "cos_sin_cache": {"shape": [_MAX_POS, rotary_dim], "dtype": dt},
            "positions": {"shape": [3, nt], "dtype": "torch.int64"},
            "mrope_section": {"scalar": list(mrope_section)},
            "head_size": {"scalar": head_size},
            "rotary_dim": {"scalar": rotary_dim},
            "is_neox_style": {"scalar": True},
        },
        "outputs": {
            "q": {"shape": [nt, n_qh * head_size], "dtype": dt},
            "k": {"shape": [nt, n_kh * head_size], "dtype": dt},
        },
        "dims": {
            "num_tokens": nt,
            "n_qh": n_qh,
            "n_kh": n_kh,
            "head_size": head_size,
            "rotary_dim": rotary_dim,
            "max_pos": _MAX_POS,
        },
    }
