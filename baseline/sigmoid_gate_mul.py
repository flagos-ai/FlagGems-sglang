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

"""Native SGLang collection workload for sigmoid_gate_mul."""

import torch

OP_NAME = "sigmoid_gate_mul"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (bs, hidden) for bs in (1, 8, 64, 512) for hidden in (1024, 4096, 8192)
]
MORE_SHAPES = [(bs, hidden) for bs in (4096,) for hidden in (1024, 4096, 8192)]


def native():
    """解析 moe.triton_sigmoid_gate_mul.sigmoid_gate_mul。"""
    try:
        from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (
            sigmoid_gate_mul as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "hidden": hidden}
        for (bs, hidden) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, hidden = shape
    g = torch.Generator(device=device).manual_seed(0)
    gate = torch.randn(
        bs, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    up = torch.randn(
        bs, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    return gate, up


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["hidden"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [bs, hidden], "dtype": dt},
            "gate": {"shape": [bs, hidden], "dtype": dt},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
