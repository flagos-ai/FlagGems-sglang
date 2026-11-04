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

"""ep_scatter baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.ep_scatter（5 个原地目标）。
复刻 problems/moe/ep_scatter/baseline.py：clone expert_start_loc/output_tensor/
output_tensor_scale/m_indices/output_index 后调 ep_scatter(recv_x, recv_x_scale,
recv_topk, num_recv_tokens_per_expert, num_valid_tokens_per_expert,
expert_start_loc, output_tensor, output_tensor_scale, m_indices, output_index)。

输入复刻 cases.py BENCH_CASES（t∈{64,512,4096}, topk=8, hidden=2048,
num_experts=8, dtype=fp8_e4m3）：按 recv_topk 的 per-expert 计数 pad 到 BLOCK_E=128
得 total 行；recv_x 为 fp8，scale 为 fp32（K//128 组）。total 数据相关（随机
ids 决定），config 中以符号描述。IS_INPLACE=True（5 个缓冲均被写回）。
"""

import torch

OP_NAME = "ep_scatter"
DTYPES = [torch.float8_e4m3fn]
IS_INPLACE = True

_BLOCK_E = 128
_QUANT_BLOCK = 128
_SHAPES = [(t, 8, 2048, 8) for t in (64, 512, 4096)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import ep_scatter as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"t": t, "topk": tk, "hidden": h, "num_experts": e}
        for (t, tk, h, e) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    t, topk, hidden, num_experts = (
        binding["t"], binding["topk"], binding["hidden"], binding["num_experts"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    ids = torch.randint(
        0, num_experts, (t, topk), device=device, generator=g, dtype=torch.int32
    )
    valid = torch.zeros(num_experts, dtype=torch.int32, device=device)
    for e in range(num_experts):
        valid[e] = int((ids == e).sum())
    padded = ((valid + _BLOCK_E - 1) // _BLOCK_E * _BLOCK_E).to(torch.int32)
    total = int(padded.sum())

    recv_x = torch.randn(
        t, hidden, dtype=torch.float32, device=device, generator=g
    ).to(dtype).contiguous()
    recv_x_scale = torch.rand(
        t, hidden // _QUANT_BLOCK, dtype=torch.float32, device=device, generator=g
    ).contiguous()

    expert_start_loc = torch.zeros(num_experts, dtype=torch.int32, device=device)
    output_tensor = torch.zeros((total, hidden), dtype=dtype, device=device)
    output_tensor_scale = torch.zeros(
        (total, hidden // _QUANT_BLOCK), dtype=torch.float32, device=device
    )
    m_indices = torch.zeros(total, dtype=torch.int32, device=device)
    output_index = torch.full((t, topk), -1, dtype=torch.int32, device=device)
    return (
        recv_x, recv_x_scale, ids, padded, valid,
        expert_start_loc, output_tensor, output_tensor_scale,
        m_indices, output_index,
    ), {}


def key_shape(binding):
    return f"t{binding['t']}_topk{binding['topk']}_h{binding['hidden']}_E{binding['num_experts']}"


def config(binding, dtype):
    t, topk, hidden, num_experts = (
        binding["t"], binding["topk"], binding["hidden"], binding["num_experts"],
    )
    dt = str(dtype)
    g = hidden // _QUANT_BLOCK
    return {
        "inputs": {
            "recv_x": {"shape": [t, hidden], "dtype": dt},
            "recv_x_scale": {"shape": [t, g], "dtype": "torch.float32"},
            "recv_topk": {"shape": [t, topk], "dtype": "torch.int32"},
            "num_recv_tokens_per_expert": {"shape": [num_experts], "dtype": "torch.int32"},
            "num_valid_tokens_per_expert": {"shape": [num_experts], "dtype": "torch.int32"},
            "expert_start_loc": {"shape": [num_experts], "dtype": "torch.int32", "note": "原地写回"},
            "output_tensor": {"shape": ["total", hidden], "dtype": dt, "note": "原地写回"},
            "output_tensor_scale": {"shape": ["total", g], "dtype": "torch.float32", "note": "原地写回"},
            "m_indices": {"shape": ["total"], "dtype": "torch.int32", "note": "原地写回"},
            "output_index": {"shape": [t, topk], "dtype": "torch.int32", "note": "原地写回"},
        },
        "outputs": {
            "expert_start_loc": {"shape": [num_experts], "dtype": "torch.int32"},
            "output_tensor": {"shape": ["total", hidden], "dtype": dt},
            "output_tensor_scale": {"shape": ["total", g], "dtype": "torch.float32"},
            "m_indices": {"shape": ["total"], "dtype": "torch.int32"},
            "output_index": {"shape": [t, topk], "dtype": "torch.int32"},
        },
        "dims": {
            "t": t, "topk": topk, "hidden": hidden, "num_experts": num_experts,
            "BLOCK_E": _BLOCK_E, "quant_block": _QUANT_BLOCK,
            "note": "total = sum(pad(per-expert counts, BLOCK_E))，随机 ids 决定，数据相关",
        },
    }
