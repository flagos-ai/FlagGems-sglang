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

"""Native SGLang collection workload for concat_and_cast_mha_k."""

import torch

OP_NAME = "concat_and_cast_mha_k"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SHAPES = [(t, 32, 128, 64) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 32, 128, 64) for t in (4096,)]


def native():
    """解析 concat_and_cast_mha_k_triton。"""
    try:
        from sglang.kernels.ops.kvcache.cache_ops import (
            concat_and_cast_mha_k_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "num_heads": num_heads,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
        }
        for (num_tokens, num_heads, nope_dim, rope_dim) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, num_heads, nope_dim, rope_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    k = torch.zeros(
        num_tokens,
        num_heads,
        nope_dim + rope_dim,
        dtype=cur_dtype,
        device=device,
    )
    k_nope = torch.randn(
        num_tokens, num_heads, nope_dim, device=device, generator=g
    ).to(cur_dtype)
    k_rope = torch.randn(
        num_tokens, 1, rope_dim, device=device, generator=g
    ).to(cur_dtype)
    return k, k_nope, k_rope


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["num_heads"],
        binding["nope_dim"],
        binding["rope_dim"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [
        binding["num_tokens"],
        binding["num_heads"],
        binding["nope_dim"] + binding["rope_dim"],
    ]


def config(binding, dtype):
    t, num_heads = binding["num_tokens"], binding["num_heads"]
    nope_dim, rope_dim = binding["nope_dim"], binding["rope_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "k": {
                "shape": [t, num_heads, nope_dim + rope_dim],
                "dtype": dt,
                "note": "原地写回",
            },
            "k_nope": {"shape": [t, num_heads, nope_dim], "dtype": dt},
            "k_rope": {"shape": [t, 1, rope_dim], "dtype": dt},
        },
        "outputs": {
            "k": {"shape": [t, num_heads, nope_dim + rope_dim], "dtype": dt},
        },
        "dims": {
            "num_tokens": t,
            "num_heads": num_heads,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
        },
    }
