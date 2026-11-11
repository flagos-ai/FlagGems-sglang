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

"""Native SGLang collection workload for fused_gate_sigmoid_mul_add."""

import torch

OP_NAME = "fused_gate_sigmoid_mul_add"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (3,)

SHAPES = [
    (bs, hidden) for bs in (1, 8, 64, 512) for hidden in (2048, 4096, 7168)
]
MORE_SHAPES = [(bs, hidden) for bs in (4096,) for hidden in (2048, 4096, 7168)]


def native():
    """解析 sglang.kernels.ops.elementwise.elementwise.fused_gate_sigmoid_mul_add。"""
    try:
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_gate_sigmoid_mul_add as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": num_tokens, "hidden": hidden}
        for (num_tokens, hidden) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, hidden = shape

    def _rand(*size, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(
            *size, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    return (
        _rand(num_tokens, hidden, seed=0),
        _rand(hidden, seed=1),
        _rand(num_tokens, hidden, seed=2),
        _rand(num_tokens, hidden, seed=3),
    )


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
            "hidden_states": {"shape": [bs, hidden], "dtype": dt},
            "gate_weight": {"shape": [hidden], "dtype": dt},
            "shared_output": {"shape": [bs, hidden], "dtype": dt},
            "out": {"shape": [bs, hidden], "dtype": dt, "note": "原地写回"},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
