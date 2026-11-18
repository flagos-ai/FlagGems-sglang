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

"""Native SGLang collection workload for merge_state."""

import torch

OP_NAME = "merge_state"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # output / output_lse 省略，kernel 内另分配

_NUM_HEADS = 32
_HEAD_SIZE = 128
_NS = [1, 8, 64, 512, 4096]


def native():
    """解析 sglang.kernels.ops.attention.merge_state.merge_state_triton。"""
    try:
        from sglang.kernels.ops.attention.merge_state import (
            merge_state_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


SHAPES = [(n, 32, 128) for n in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(7, 4, 64), (83, 16, 128), (3, 32, 512)]


def grid():
    return [
        {"n_tokens": n_tokens, "num_heads": num_heads, "head_size": head_size}
        for (n_tokens, num_heads, head_size) in dict.fromkeys(
            (*SHAPES, *MORE_SHAPES)
        )
    ]


def build_inputs(binding, dtype, device):
    n, num_heads, head_size = (
        binding["n_tokens"],
        binding["num_heads"],
        binding["head_size"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    prefix_output = torch.randn(
        n,
        num_heads,
        head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    suffix_output = torch.randn(
        n,
        num_heads,
        head_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    prefix_lse = torch.randn(n, num_heads, generator=g, device=device) * 3
    suffix_lse = torch.randn(n, num_heads, generator=g, device=device) * 3
    return (prefix_output, prefix_lse, suffix_output, suffix_lse), {}


def key_shape(binding):
    return [binding["n_tokens"], binding["num_heads"], binding["head_size"]]


def config(binding, dtype):
    n, num_heads, head_size = (
        binding["n_tokens"],
        binding["num_heads"],
        binding["head_size"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "prefix_output": {"shape": [n, num_heads, head_size], "dtype": dt},
            "prefix_lse": {"shape": [n, num_heads], "dtype": "torch.float32"},
            "suffix_output": {"shape": [n, num_heads, head_size], "dtype": dt},
            "suffix_lse": {"shape": [n, num_heads], "dtype": "torch.float32"},
        },
        "outputs": {
            "output": {"shape": [n, num_heads, head_size], "dtype": dt},
            "output_lse": {"shape": [n, num_heads], "dtype": "torch.float32"},
        },
        "dims": {
            "n_tokens": n,
            "num_heads": num_heads,
            "head_size": head_size,
        },
    }
