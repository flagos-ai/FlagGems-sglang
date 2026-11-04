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

"""post_reorder_cutlass baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.post_reorder_for_cutlass_moe
（destination-passing）。复刻 problems/moe/post_reorder_cutlass/baseline.py：
out=output.clone()，调 post_reorder_for_cutlass_moe(down_output, out, src2dst,
topk_ids, topk_weights, num_local_experts, topk, num_tokens, hidden_size,
routed_scaling_factor)，原地写 out 并返回。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,2048}, topk=8, hidden=7168,
num_local_experts=32, scaling=1.0）。IS_INPLACE=True（写回 out）。
"""

import torch

OP_NAME = "post_reorder_cutlass"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

_SCALE = 1.0
_SHAPES = [(t, 8, 7168, 32) for t in (1, 8, 64, 512, 2048)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            post_reorder_for_cutlass_moe as _f,
        )
    except Exception:
        return None

    def _op(down_output, out, src2dst, topk_ids, topk_weights, num_local_experts,
            topk, num_tokens, hidden_size, routed_scaling_factor):
        _f(down_output, out, src2dst, topk_ids, topk_weights, num_local_experts,
           topk, num_tokens, hidden_size, routed_scaling_factor)
        return out

    return _op


def grid():
    return [
        {"num_tokens": t, "topk": tk, "hidden": h, "num_local_experts": e}
        for (t, tk, h, e) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    num_tokens, topk, hidden, num_local_experts = (
        binding["num_tokens"], binding["topk"], binding["hidden"], binding["num_local_experts"],
    )
    n = num_tokens * topk
    g = torch.Generator(device=device).manual_seed(0)
    down = torch.randn(n, hidden, device=device, generator=g).to(dtype)
    out = torch.zeros(num_tokens, hidden, dtype=dtype, device=device)
    ids = torch.randint(
        0, num_local_experts + 1, (num_tokens, topk), dtype=torch.int32, device=device, generator=g
    )
    dst = torch.randperm(n, device=device, generator=g).reshape(num_tokens, topk).to(torch.int32)
    weights = torch.rand(num_tokens, topk, dtype=torch.float32, device=device, generator=g)
    return (down, out, dst, ids, weights, num_local_experts, topk, num_tokens, hidden, _SCALE), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_topk{binding['topk']}_h{binding['hidden']}_E{binding['num_local_experts']}"


def config(binding, dtype):
    num_tokens, topk, hidden, num_local_experts = (
        binding["num_tokens"], binding["topk"], binding["hidden"], binding["num_local_experts"],
    )
    n = num_tokens * topk
    dt = str(dtype)
    return {
        "inputs": {
            "down_output": {"shape": [n, hidden], "dtype": dt},
            "out": {"shape": [num_tokens, hidden], "dtype": dt, "note": "原地写回（output.clone()）"},
            "src2dst": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "topk_ids": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "topk_weights": {"shape": [num_tokens, topk], "dtype": "torch.float32"},
            "num_local_experts": {"scalar": num_local_experts},
            "topk": {"scalar": topk},
            "num_tokens": {"scalar": num_tokens},
            "hidden_size": {"scalar": hidden},
            "routed_scaling_factor": {"scalar": _SCALE},
        },
        "outputs": {
            "out": {"shape": [num_tokens, hidden], "dtype": dt},
        },
        "dims": {"num_tokens": num_tokens, "topk": topk, "hidden": hidden,
                 "num_local_experts": num_local_experts, "n": n},
    }
