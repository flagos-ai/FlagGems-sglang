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

"""sigmoid_gate_topk_renorm baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.moe.sigmoid_gate_topk_renorm.sigmoid_gate_topk_renorm
    调用形态复刻 problems/moe/sigmoid_gate_topk_renorm/baseline.py：
        sigmoid_gate_topk_renorm(logits, k, n_shared_experts, route_scale,
                                 global_scale, bias, return_packed_topk=False)
    另分配输出（routed_w, indices, shared_w, _packed），不原地写回 ->
    IS_INPLACE=False。

输入构造复刻 problems/moe/sigmoid_gate_topk_renorm/cases.py 的 BENCH_CASES：
    n = n_routed + n_shared；logits=randn(m, n).to(dtype);
    bias=0.1*randn(n_routed); global_scale=tensor(global_scale) fp32
    k=5, n_routed=128, n_shared=3, route_scale=1.5, global_scale=0.8。
dtype 取 benchmark/test_sigmoid_gate_topk_renorm.py 的 [bfloat16]。
"""

import torch

OP_NAME = "sigmoid_gate_topk_renorm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# (m, n_routed, n_shared, k, route_scale, global_scale)
_SHAPES = [(m, 128, 3, 5, 1.5, 0.8) for m in (1, 8, 64, 512, 4096)]


def native():
    """解析 sigmoid_gate_topk_renorm。"""
    try:
        from sglang.kernels.ops.moe.sigmoid_gate_topk_renorm import (
            sigmoid_gate_topk_renorm as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "m": m, "n_routed": nr, "n_shared": ns, "k": k,
            "route_scale": rs, "global_scale": gs,
        }
        for (m, nr, ns, k, rs, gs) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    m, n_routed, n_shared, k, route_scale, global_scale = (
        binding["m"], binding["n_routed"], binding["n_shared"], binding["k"],
        binding["route_scale"], binding["global_scale"],
    )
    n = n_routed + n_shared
    g = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(dtype)
    bias = 0.1 * torch.randn(n_routed, generator=g, device=device, dtype=torch.float32)
    global_scale_t = torch.tensor(global_scale, device=device, dtype=torch.float32)
    args = (logits, k, n_shared, route_scale, global_scale_t, bias)
    return args, {"return_packed_topk": False}


def key_shape(binding):
    n = binding["n_routed"] + binding["n_shared"]
    return [binding["m"], n, binding["k"]]


def config(binding, dtype):
    m, n_routed, n_shared, k, route_scale, global_scale = (
        binding["m"], binding["n_routed"], binding["n_shared"], binding["k"],
        binding["route_scale"], binding["global_scale"],
    )
    n = n_routed + n_shared
    dt = str(dtype)
    return {
        "inputs": {
            "logits": {"shape": [m, n], "dtype": dt},
            "k": {"scalar": k},
            "n_shared_experts": {"scalar": n_shared},
            "route_scale": {"scalar": route_scale},
            "global_scale": {"shape": [], "dtype": "torch.float32",
                             "note": "0-d 标量张量"},
            "bias": {"shape": [n_routed], "dtype": "torch.float32"},
            "return_packed_topk": {"scalar": False},
        },
        "outputs": {
            "routed_w": {"shape": [m, k], "dtype": "torch.float32"},
            "indices": {"shape": [m, k], "dtype": "torch.int32"},
            "shared_w": {"shape": [m, n_shared], "dtype": "torch.float32"},
        },
        "dims": {
            "m": m, "n": n, "n_routed": n_routed, "n_shared": n_shared, "k": k,
            "route_scale": route_scale, "global_scale": global_scale,
        },
    }
