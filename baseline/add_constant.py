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

"""Native SGLang collection workload for add_constant."""

import torch

OP_NAME = "add_constant"
DTYPES = [torch.int32]
IS_INPLACE = False
_CONSTANT = 42

SHAPES = [(1 << 12,), (1 << 16,), (1 << 20,)]
MORE_SHAPES = [(1 << 24,)]


def native():
    """解析 sglang.kernels.ops.elementwise.add_constant.add_constant。"""
    try:
        from sglang.kernels.ops.elementwise.add_constant import (
            add_constant as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [{"n": n} for (n,) in (*SHAPES, *MORE_SHAPES)]


def build_inputs(binding, dtype, device):
    n = binding["n"]
    g = torch.Generator(device=device).manual_seed(9)
    src = torch.randint(
        -(2**20),
        2**20,
        (n,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    return (src, _CONSTANT), {}


def key_shape(binding):
    return [binding["n"]]


def config(binding, dtype):
    n = binding["n"]
    return {
        "inputs": {
            "src": {"shape": [n], "dtype": "torch.int32"},
            "constant": {"scalar": _CONSTANT},
        },
        "outputs": {
            "out": {"shape": [n], "dtype": "torch.int32"},
        },
        "dims": {"n": n},
    }
