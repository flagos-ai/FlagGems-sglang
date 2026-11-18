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

"""Native SGLang v0.5.21 collection workload for sglang_fl_rotary_embedding."""

import torch

OP_NAME = "sglang_fl_rotary_embedding"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0, 1)

SHAPES = [(t, 32, 8, 128) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(4096, 32, 8, 128)]
_MAX_POSITION = 8192


def native():
    try:
        from sglang.kernels.ops.attention.rope import apply_rope_inplace as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(
        zip(("num_tokens", "num_q_heads", "num_kv_heads", "head_size"), shape)
    )


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    num_tokens, num_q_heads, num_kv_heads, head_size = shape
    rotary_dim = head_size
    g = torch.Generator(device=device).manual_seed(9)
    positions = torch.randint(
        0, _MAX_POSITION, (num_tokens,), device=device, generator=g
    )
    query = torch.randn(
        num_tokens, num_q_heads * head_size, device=device, generator=g
    ).to(cur_dtype)
    key = torch.randn(
        num_tokens, num_kv_heads * head_size, device=device, generator=g
    ).to(cur_dtype)
    cos_sin_cache = torch.randn(
        _MAX_POSITION, rotary_dim, device=device, generator=g
    )
    return positions, query, key, head_size, cos_sin_cache, {"is_neox": True}


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["num_q_heads"],
        binding["num_kv_heads"],
        binding["head_size"],
    )
    positions, query, key, head_size, cache, kwargs = build_case(
        shape, dtype, device
    )
    tokens = positions.numel()
    query = query.view(tokens, -1, head_size)
    key = key.view(tokens, -1, head_size)
    return (query, key, cache, positions), kwargs


def key_shape(binding):
    return [
        binding["num_tokens"],
        binding["num_q_heads"],
        binding["num_kv_heads"],
        binding["head_size"],
    ]


def config(binding, dtype):
    num_tokens, num_q_heads, num_kv_heads, head_size = (
        binding["num_tokens"],
        binding["num_q_heads"],
        binding["num_kv_heads"],
        binding["head_size"],
    )

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "query": tensor([num_tokens, num_q_heads, head_size]),
            "key": tensor([num_tokens, num_kv_heads, head_size]),
            "cos_sin_cache": tensor(
                [_MAX_POSITION, head_size], "torch.float32"
            ),
            "positions": tensor([num_tokens], "torch.int64"),
            "is_neox": {"scalar": True},
        },
        "outputs": {
            "query": tensor([num_tokens, num_q_heads, head_size]),
            "key": tensor([num_tokens, num_kv_heads, head_size]),
        },
        "dims": dict(binding),
    }
