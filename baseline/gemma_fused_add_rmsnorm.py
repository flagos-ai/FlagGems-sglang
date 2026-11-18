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

"""Native SGLang collection workload for gemma_fused_add_rmsnorm."""

import torch

OP_NAME = "gemma_fused_add_rmsnorm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (1024, 4096, 8192)]
MORE_SHAPES = [(4096, n) for n in (1024, 4096, 8192)]
_EPS = 1e-6


def native():
    """解析 minimax_m3_rmsnorm.gemma_fused_add_rmsnorm。"""
    try:
        from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
            gemma_fused_add_rmsnorm as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n} for (m, n) in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    m, n = shape
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(m, n, dtype=torch.float32, device=device, generator=g).to(
        cur_dtype
    )
    residual = torch.randn(
        m, n, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    weight = (
        0.5 * torch.randn(n, dtype=torch.float32, device=device, generator=g)
    ).to(cur_dtype)
    return x, residual, weight, _EPS


def build_inputs(binding, dtype, device):
    shape = (binding["m"], binding["n"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "residual": {"shape": [m, n], "dtype": dt},
            "weight": {"shape": [n], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": dt},
            "residual_out": {"shape": [m, n], "dtype": dt},
        },
        "dims": {"m": m, "n": n},
    }
