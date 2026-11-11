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

"""Native SGLang collection workload for fused_gdn_gating."""

import torch

OP_NAME = "fused_gdn_gating"
DTYPES = [torch.float32]
IS_INPLACE = False

SHAPES = [(m, 64) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(1, 4), (37, 32), (256, 8)]


def native():
    """解析 sglang.kernels.ops.attention.fla.fused_gdn_gating.fused_gdn_gating。"""
    try:
        from sglang.kernels.ops.attention.fla.fused_gdn_gating import (
            fused_gdn_gating as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": batch, "num_heads": num_heads}
        for (batch, num_heads) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    batch, num_heads = shape
    g = torch.Generator(device=device).manual_seed(0)
    # This op is float32 throughout; cur_dtype is not applied.
    a_log = (
        torch.randn(num_heads, generator=g, device=device, dtype=torch.float32)
        * 0.5
    )
    a = torch.randn(
        batch, num_heads, generator=g, device=device, dtype=torch.float32
    )
    b = torch.randn(
        batch, num_heads, generator=g, device=device, dtype=torch.float32
    )
    dt_bias = torch.randn(
        num_heads, generator=g, device=device, dtype=torch.float32
    )
    return a_log, a, b, dt_bias


def build_inputs(binding, dtype, device):
    shape = (binding["batch"], binding["num_heads"])
    return build_case(shape, dtype, device), {"beta": 1.0, "threshold": 20.0}


def key_shape(binding):
    return [binding["batch"], binding["num_heads"]]


def config(binding, dtype):
    batch, num_heads = binding["batch"], binding["num_heads"]
    return {
        "inputs": {
            "A_log": {"shape": [num_heads], "dtype": "torch.float32"},
            "a": {"shape": [batch, num_heads], "dtype": "torch.float32"},
            "b": {"shape": [batch, num_heads], "dtype": "torch.float32"},
            "dt_bias": {"shape": [num_heads], "dtype": "torch.float32"},
            "beta": {"scalar": 1.0},
            "threshold": {"scalar": 20.0},
        },
        "outputs": {
            "g": {"shape": [batch, num_heads], "dtype": "torch.float32"},
            "beta_output": {
                "shape": [batch, num_heads],
                "dtype": "torch.float32",
            },
        },
        "dims": {"batch": batch, "num_heads": num_heads},
    }
