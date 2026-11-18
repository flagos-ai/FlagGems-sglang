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

"""Native SGLang collection workload for interleaved_rope."""

import torch

OP_NAME = "interleaved_rope"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(s, 128, (16, 24, 24)) for s in (1, 128, 2048, 8192)]
MORE_SHAPES = [
    (1, 48, (8, 8, 8)),
    (37, 96, (16, 16, 16)),
    (257, 128, (16, 24, 24)),
]


def native():
    """解析 apply_interleaved_rope_triton。"""
    try:
        from sglang.kernels.ops.attention.mrope import (
            apply_interleaved_rope_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"s": s, "d": d, "mrope_section": mrope_section}
        for (s, d, mrope_section) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    s, d, mrope_section = shape
    # The op reads the three rope variants from a single [3, s, d] tensor.
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        3, s, d, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    return x, list(mrope_section)


def build_inputs(binding, dtype, device):
    shape = (binding["s"], binding["d"], binding["mrope_section"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [3, binding["s"], binding["d"]]


def config(binding, dtype):
    s, d = binding["s"], binding["d"]
    mrope_section = binding["mrope_section"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [3, s, d], "dtype": dt},
            "mrope_section": {"scalar": list(mrope_section)},
        },
        "outputs": {
            "out": {"shape": [3, s, d], "dtype": dt},
        },
        "dims": {"s": s, "d": d, "mrope_section": list(mrope_section)},
    }
