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

"""Native SGLang collection workload for per_token_group_quant_int8."""

import torch

OP_NAME = "per_token_group_quant_int8"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (m, k, 128) for m in (1, 8, 64, 512, 4096) for k in (2048, 4096, 8192)
]
MORE_SHAPES = [(7, 128, 128), (83, 512, 128), (256, 4096, 128), (3, 256, 64)]


def native():
    """解析 per_token_group_quant_int8。"""
    try:
        from sglang.kernels.ops.quantization.int8_kernel import (
            per_token_group_quant_int8 as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "k": k, "group_size": group_size}
        for (m, k, group_size) in (*SHAPES, *MORE_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    m, k, group_size = binding["m"], binding["k"], binding["group_size"]
    g = torch.Generator(device=device).manual_seed(0)
    x = (
        (
            torch.rand(m, k, generator=g, device=device, dtype=torch.float32)
            * 2
            - 1
        )
        .to(dtype)
        .contiguous()
    )
    return (x, group_size), {}


def key_shape(binding):
    return [binding["m"], binding["k"]]


def config(binding, dtype):
    m, k, group_size = binding["m"], binding["k"], binding["group_size"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, k], "dtype": dt},
            "group_size": {"scalar": group_size},
        },
        "outputs": {
            "q": {"shape": [m, k], "dtype": "torch.int8"},
            "scale": {"shape": [m, k // group_size], "dtype": "torch.float32"},
        },
        "dims": {"m": m, "k": k, "group_size": group_size},
    }
