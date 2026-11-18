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

"""Native SGLang v0.5.21 collection workload for residual_gate_add."""

import torch

OP_NAME = "residual_gate_add"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (rows, 3072, broadcast) for rows in (1, 64, 1024) for broadcast in (0, 1)
]
MORE_SHAPES = [
    (rows, 3072, broadcast) for rows in (16384,) for broadcast in (0, 1)
]


def native():
    try:
        from sglang.kernels.kda_kernels.residual_gate_add_jit import (
            residual_gate_add_cuda as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("rows", "hidden", "broadcast"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    rows, hidden, broadcast = shape
    broadcast = bool(broadcast)
    g = torch.Generator(device=device).manual_seed(0)

    def r(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=device, generator=g
        ).to(cur_dtype)

    gate = r(1, hidden) if broadcast else r(rows, hidden)
    residual = r(rows, hidden)
    update = r(rows, hidden)
    return residual, update, gate


def build_inputs(binding, dtype, device):
    shape = (binding["rows"], binding["hidden"], binding["broadcast"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["rows"], binding["hidden"], binding["broadcast"]]


def config(binding, dtype):
    rows, hidden, broadcast = (
        binding["rows"],
        binding["hidden"],
        binding["broadcast"],
    )

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "residual": tensor([rows, hidden]),
            "update": tensor([rows, hidden]),
            "gate": tensor([1 if broadcast else rows, hidden]),
        },
        "outputs": {"out": tensor([rows, hidden])},
        "dims": dict(binding),
    }
