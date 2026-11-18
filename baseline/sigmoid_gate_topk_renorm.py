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

"""Native SGLang collection workload for sigmoid_gate_topk_renorm."""

import torch

OP_NAME = "sigmoid_gate_topk_renorm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

ROUTE_SCALE = 1.5
GLOBAL_SCALE = 0.8
SHAPES = [(m, 128, 3, 5) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(1, 32, 1, 3), (83, 64, 1, 6), (9, 256, 4, 12)]


def native():
    """解析 sigmoid_gate_topk_renorm。"""
    try:
        from sglang.kernels.ops.moe.sigmoid_gate_topk_renorm import (
            sigmoid_gate_topk_renorm as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "m": m,
            "n_routed": n_routed,
            "n_shared": n_shared,
            "k": k,
            **{"route_scale": ROUTE_SCALE, "global_scale": GLOBAL_SCALE},
        }
        for (m, n_routed, n_shared, k) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    m, n_routed, n_shared, k = shape
    g = torch.Generator(device=device).manual_seed(0)
    n = n_routed + n_shared
    logits = torch.randn(
        m, n, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    bias = 0.1 * torch.randn(
        n_routed, generator=g, device=device, dtype=torch.float32
    )
    global_scale = torch.tensor(
        GLOBAL_SCALE, device=device, dtype=torch.float32
    )
    return logits, k, n_shared, ROUTE_SCALE, global_scale, bias


def build_inputs(binding, dtype, device):
    shape = (
        binding["m"],
        binding["n_routed"],
        binding["n_shared"],
        binding["k"],
    )
    return build_case(shape, dtype, device), {"return_packed_topk": False}


def key_shape(binding):
    n = binding["n_routed"] + binding["n_shared"]
    return [binding["m"], n, binding["k"]]


def config(binding, dtype):
    m, n_routed, n_shared, k, route_scale, global_scale = (
        binding["m"],
        binding["n_routed"],
        binding["n_shared"],
        binding["k"],
        binding["route_scale"],
        binding["global_scale"],
    )
    n = n_routed + n_shared
    dt = str(dtype)
    return {
        "inputs": {
            "logits": {"shape": [m, n], "dtype": dt},
            "k": {"scalar": k},
            "n_shared_experts": {"scalar": n_shared},
            "route_scale": {"scalar": route_scale},
            "global_scale": {
                "shape": [],
                "dtype": "torch.float32",
                "note": "0-d 标量张量",
            },
            "bias": {"shape": [n_routed], "dtype": "torch.float32"},
            "return_packed_topk": {"scalar": False},
        },
        "outputs": {
            "routed_w": {"shape": [m, k], "dtype": "torch.float32"},
            "indices": {"shape": [m, k], "dtype": "torch.int32"},
            "shared_w": {"shape": [m, n_shared], "dtype": "torch.float32"},
        },
        "dims": {
            "m": m,
            "n": n,
            "n_routed": n_routed,
            "n_shared": n_shared,
            "k": k,
            "route_scale": route_scale,
            "global_scale": global_scale,
        },
    }
