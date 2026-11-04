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

"""fused_moe_dispatch_index baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.fused_moe_dispatch_index。
复刻 problems/moe/fused_moe_dispatch_index/baseline.py：
    masked_m, src2dst = fused_moe_dispatch_index(topk_ids, num_local_experts, m_max)
返回新张量，不原地写回 -> IS_INPLACE=False。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,4096}, topk=8, num_local_experts=32,
m_max=8192）：topk_ids=randint(0,num_local_experts,(t,topk)).int32。
"""

import torch

OP_NAME = "fused_moe_dispatch_index"
DTYPES = [torch.int32]
IS_INPLACE = False

_SHAPES = [(t, 8, 32, 8192) for t in (1, 8, 64, 512, 4096)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            fused_moe_dispatch_index as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": t, "topk": tk, "num_local_experts": e, "m_max": mm}
        for (t, tk, e, mm) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    t, topk, e, m_max = (
        binding["num_tokens"], binding["topk"], binding["num_local_experts"], binding["m_max"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    topk_ids = torch.randint(
        0, e, (t, topk), dtype=torch.int32, device=device, generator=g
    )
    return (topk_ids, e, m_max), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_topk{binding['topk']}_E{binding['num_local_experts']}_mmax{binding['m_max']}"


def config(binding, dtype):
    t, topk, e, m_max = (
        binding["num_tokens"], binding["topk"], binding["num_local_experts"], binding["m_max"],
    )
    return {
        "inputs": {
            "topk_ids": {"shape": [t, topk], "dtype": "torch.int32"},
            "num_local_experts": {"scalar": e},
            "m_max": {"scalar": m_max},
        },
        "outputs": {
            "masked_m": {"shape": [e], "dtype": "torch.int32"},
            "src2dst": {"shape": [t, topk], "dtype": "torch.int32"},
        },
        "dims": {"num_tokens": t, "topk": topk, "num_local_experts": e, "m_max": m_max},
    }
