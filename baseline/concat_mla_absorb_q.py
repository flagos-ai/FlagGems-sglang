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

"""Native SGLang collection workload for concat_mla_absorb_q."""

import torch

OP_NAME = "concat_mla_absorb_q"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(t, 128, 512, 64) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 128, 512, 64) for t in (4096,)]


def native():
    """解析 sglang.kernels.ops.attention.concat_mla.concat_mla_absorb_q。"""
    try:
        from sglang.kernels.ops.attention.concat_mla import (
            concat_mla_absorb_q as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "num_heads": num_heads,
            "a_last": a_last,
            "b_last": b_last,
        }
        for (num_tokens, num_heads, a_last, b_last) in (*SHAPES, *MORE_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    num_tokens, num_heads, a_last, b_last = (
        binding["num_tokens"],
        binding["num_heads"],
        binding["a_last"],
        binding["b_last"],
    )
    g = torch.Generator(device=device).manual_seed(9)
    a = torch.randn(
        num_tokens, num_heads, a_last, device=device, generator=g
    ).to(dtype)
    b = torch.randn(
        num_tokens, num_heads, b_last, device=device, generator=g
    ).to(dtype)
    return (a, b), {}


def key_shape(binding):
    return [
        binding["num_tokens"],
        binding["num_heads"],
        binding["a_last"],
        binding["b_last"],
    ]


def config(binding, dtype):
    t = binding["num_tokens"]
    nh, a_last, b_last = (
        binding["num_heads"],
        binding["a_last"],
        binding["b_last"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "a": {"shape": [t, nh, a_last], "dtype": dt},
            "b": {"shape": [t, nh, b_last], "dtype": dt},
        },
        "outputs": {
            "out": {"shape": [t, nh, a_last + b_last], "dtype": dt},
        },
        "dims": {
            "num_tokens": t,
            "num_heads": nh,
            "a_last": a_last,
            "b_last": b_last,
        },
    }
