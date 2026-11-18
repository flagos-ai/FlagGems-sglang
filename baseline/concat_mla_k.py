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

"""Native SGLang collection workload for concat_mla_k."""

import torch

OP_NAME = "concat_mla_k"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0,)

SHAPES = [(t, 128, 128, 64) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 128, 128, 64) for t in (4096,)]


def native():
    """解析 sglang.kernels.ops.attention.concat_mla.concat_mla_k。"""
    try:
        from sglang.kernels.ops.attention.concat_mla import concat_mla_k as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "num_tokens": total_tokens,
            "num_heads": num_heads,
            "nope_dim": nope_dim,
            "rope_dim": rope_dim,
        }
        for (total_tokens, num_heads, nope_dim, rope_dim) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    total_tokens, num_heads, nope_dim, rope_dim = shape
    g = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(
        total_tokens,
        num_heads,
        nope_dim + rope_dim,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(cur_dtype)
    k_nope = torch.randn(
        total_tokens,
        num_heads,
        nope_dim,
        dtype=torch.float32,
        device=device,
        generator=g,
    ).to(cur_dtype)
    k_rope = torch.randn(
        total_tokens,
        1,
        rope_dim,
        dtype=torch.float32,
        device=device,
        generator=g,
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
        binding["nope_dim"],
        binding["rope_dim"],
    ]


def config(binding, dtype):
    t = binding["num_tokens"]
    nh, nope, rope = (
        binding["num_heads"],
        binding["nope_dim"],
        binding["rope_dim"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "out": {
                "shape": [t, nh, nope + rope],
                "dtype": dt,
                "note": "原地写回",
            },
            "k_nope": {"shape": [t, nh, nope], "dtype": dt},
            "k_rope": {"shape": [t, 1, rope], "dtype": dt},
        },
        "outputs": {
            "out": {"shape": [t, nh, nope + rope], "dtype": dt},
        },
        "dims": {
            "num_tokens": t,
            "num_heads": nh,
            "nope_dim": nope,
            "rope_dim": rope,
        },
    }
