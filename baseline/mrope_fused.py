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

"""Native SGLang collection workload for mrope_fused."""

import torch

OP_NAME = "mrope_fused"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (0, 1)

MROPE_SECTION = [16, 24, 24]
MAX_POSITION = 4096
SHAPES = [(n, 8, 2, 128, 128) for n in (1, 128, 2048, 8192)]


def native():
    """解析 triton_mrope_fused。"""
    try:
        from sglang.kernels.ops.attention.rotary_triton import (
            triton_mrope_fused as op,
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
            **{"mrope_section": MROPE_SECTION},
        }
        for (num_tokens, n_qh, n_kh, head_size, rotary_dim) in SHAPES
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, n_qh, n_kh, head_size, rotary_dim = shape
    q = torch.randn(
        num_tokens, n_qh * head_size, device=device, dtype=cur_dtype
    )
    k = torch.randn(
        num_tokens, n_kh * head_size, device=device, dtype=cur_dtype
    )
    cos_sin_cache = torch.randn(
        MAX_POSITION, rotary_dim, device=device, dtype=cur_dtype
    )
    positions = torch.randint(
        0, MAX_POSITION, (3, num_tokens), device=device, dtype=torch.int64
    )
    return q, k, cos_sin_cache, positions, MROPE_SECTION, head_size, rotary_dim


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["n_qh"],
        binding["n_kh"],
        binding["head_size"],
        binding["rotary_dim"],
    )
    axis_map = torch.zeros(1, dtype=torch.int32, device=device)
    return build_case(shape, dtype, device), {
        "mrope_interleaved": False,
        "mrope_interleaved_glm": False,
        "is_neox_style": True,
        "axis_map": axis_map,
    }


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
            "cos_sin_cache": {
                "shape": [MAX_POSITION, rotary_dim],
                "dtype": dt,
            },
            "positions": {"shape": [3, nt], "dtype": "torch.int64"},
            "mrope_section": {"scalar": list(mrope_section)},
            "head_size": {"scalar": head_size},
            "rotary_dim": {"scalar": rotary_dim},
            "mrope_interleaved": {"scalar": False},
            "mrope_interleaved_glm": {"scalar": False},
            "is_neox_style": {"scalar": True},
            "axis_map": {"shape": [1], "dtype": "torch.int32"},
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
            "max_pos": MAX_POSITION,
        },
    }
