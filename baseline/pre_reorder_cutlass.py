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

"""pre_reorder_cutlass baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.pre_reorder_for_cutlass_moe
（destination-passing）。复刻 problems/moe/pre_reorder_cutlass/baseline.py：
out=gateup_input.clone()，调 pre_reorder_for_cutlass_moe(input, out, src2dst,
topk_ids, a1_scales, num_local_experts, topk, num_tokens, hidden_size)，原地写 out。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,2048}, topk=8, hidden=7168,
num_local_experts=32）：input=randn(t,hidden).bf16; gateup_input=zeros(t*topk,hidden);
ids=randint(0,num_local_experts+1).int32; a1_scales=tensor([2.0]).fp32。
IS_INPLACE=True（写回 out）。
"""

import torch

OP_NAME = "pre_reorder_cutlass"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

_SHAPES = [(t, 8, 7168, 32) for t in (1, 8, 64, 512, 2048)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            pre_reorder_for_cutlass_moe as _f,
        )
    except Exception:
        return None

    def _op(input, out, src2dst, topk_ids, a1_scales, num_local_experts,
            topk, num_tokens, hidden_size):
        _f(input, out, src2dst, topk_ids, a1_scales, num_local_experts,
           topk, num_tokens, hidden_size)
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
    x = torch.randn(num_tokens, hidden, device=device, generator=g).to(dtype)
    out = torch.zeros(n, hidden, dtype=dtype, device=device)
    ids = torch.randint(
        0, num_local_experts + 1, (num_tokens, topk), dtype=torch.int32, device=device, generator=g
    )
    dst = torch.randperm(n, device=device, generator=g).reshape(num_tokens, topk).to(torch.int32)
    a1_scales = torch.tensor([2.0], dtype=torch.float32, device=device)
    return (x, out, dst, ids, a1_scales, num_local_experts, topk, num_tokens, hidden), {}


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
            "input": {"shape": [num_tokens, hidden], "dtype": dt},
            "out": {"shape": [n, hidden], "dtype": dt, "note": "原地写回（gateup_input.clone()）"},
            "src2dst": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "topk_ids": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "a1_scales": {"shape": [1], "dtype": "torch.float32"},
            "num_local_experts": {"scalar": num_local_experts},
            "topk": {"scalar": topk},
            "num_tokens": {"scalar": num_tokens},
            "hidden_size": {"scalar": hidden},
        },
        "outputs": {
            "out": {"shape": [n, hidden], "dtype": dt},
        },
        "dims": {"num_tokens": num_tokens, "topk": topk, "hidden": hidden,
                 "num_local_experts": num_local_experts, "n": n},
    }
