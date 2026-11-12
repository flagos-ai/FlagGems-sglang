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

"""Native SGLang collection workload for fused_dual_residual_rmsnorm."""

import torch

OP_NAME = "fused_dual_residual_rmsnorm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(bs, h) for bs in (1, 8, 64, 512, 4096) for h in (1024, 4096, 8192)]
MORE_SHAPES = [(7, 16), (83, 1024), (48, 3072), (1, 8192)]
_EPS = 1e-6


def native():
    """解析 sglang.kernels.ops.elementwise.elementwise.fused_dual_residual_rmsnorm。"""
    try:
        from sglang.kernels.ops.elementwise.elementwise import (
            fused_dual_residual_rmsnorm as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "hidden": hidden}
        for (bs, hidden) in dict.fromkeys((*SHAPES, *MORE_SHAPES))
    ]


def build_case(shape, cur_dtype, device):
    bs, hidden = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        bs, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    residual = torch.randn(
        bs, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    weight1 = torch.randn(
        hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    weight2 = torch.randn(
        hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    return x, residual, weight1, weight2, _EPS


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["hidden"])
    return build_case(shape, dtype, device), {"autotune": False}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [bs, hidden], "dtype": dt},
            "residual": {"shape": [bs, hidden], "dtype": dt},
            "weight1": {"shape": [hidden], "dtype": dt},
            "weight2": {"shape": [hidden], "dtype": dt},
            "eps": {"scalar": _EPS},
            "autotune": {"scalar": False},
        },
        "outputs": {
            "output": {"shape": [bs, hidden], "dtype": dt},
            "mid": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
