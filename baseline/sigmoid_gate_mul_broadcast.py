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

"""Native SGLang collection workload for sigmoid_gate_mul_broadcast."""

import torch

OP_NAME = "sigmoid_gate_mul_broadcast"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (bs, hidden) for bs in (1, 8, 64, 512) for hidden in (1024, 4096, 8192)
]
MORE_SHAPES = [(bs, hidden) for bs in (4096,) for hidden in (1024, 4096, 8192)]


def native():
    """解析 moe.triton_sigmoid_gate_mul.sigmoid_gate_mul_broadcast。"""
    try:
        from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (
            sigmoid_gate_mul_broadcast as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "hidden": hidden}
        for (bs, hidden) in dict.fromkeys((*SHAPES, *MORE_SHAPES))
    ]


def build_inputs(binding, dtype, device):
    bs, hidden = binding["bs"], binding["hidden"]
    g0 = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        bs, hidden, dtype=torch.float32, device=device, generator=g0
    ).to(dtype)
    g1 = torch.Generator(device=device).manual_seed(1)
    gate = torch.randn(
        bs, 1, dtype=torch.float32, device=device, generator=g1
    ).to(dtype)
    return (x, gate), {}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [bs, hidden], "dtype": dt},
            "gate": {"shape": [bs, 1], "dtype": dt, "note": "沿末维广播"},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
