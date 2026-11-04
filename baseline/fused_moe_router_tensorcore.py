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

"""fused_moe_router_tensorcore baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.moe.router.fused_moe_router_tensorcore
    调用形态复刻 problems/moe/fused_moe_router_tensorcore/baseline.py：
        fused_moe_router_tensorcore(
            x, router_weight, topk, moe_softcapping,
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=max(16, 1 << (num_experts - 1).bit_length()),
            BLOCK_SIZE_K=min(64, hidden_dim),
            correction_bias=None)
    另分配输出（topk_weights, topk_ids），不原地写回 -> IS_INPLACE=False。

输入构造复刻 problems/moe/fused_moe_router_tensorcore/cases.py 的 BENCH_CASES：
    x=randn(bs, hidden).to(dtype); router_weight=randn(num_experts, hidden).to(dtype)
    topk=2; moe_softcapping=0.0; correction_bias=None
    其中 hidden=4096, num_experts=256（tensorcore kernel 要求 topk<=2）。
dtype 取 benchmark/test_fused_moe_router_tensorcore.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_moe_router_tensorcore"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [(m, 4096, 256, 2) for m in (1, 8, 64, 512, 4096)]


def native():
    """解析 sglang.kernels.ops.moe.router.fused_moe_router_tensorcore。"""
    try:
        from sglang.kernels.ops.moe.router import fused_moe_router_tensorcore as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": bs, "hidden": h, "num_experts": e, "topk": tk}
        for (bs, h, e, tk) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    bs, hidden, num_experts, topk = (
        binding["bs"], binding["hidden"], binding["num_experts"], binding["topk"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(bs, hidden, generator=g, device=device, dtype=torch.float32).to(dtype)
    router_weight = torch.randn(
        num_experts, hidden, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    kwargs = dict(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=max(16, 1 << (num_experts - 1).bit_length()),
        BLOCK_SIZE_K=min(64, hidden),
        correction_bias=None,
    )
    return (x, router_weight, topk, 0.0), kwargs


def key_shape(binding):
    return [binding["bs"], binding["hidden"], binding["num_experts"], binding["topk"]]


def config(binding, dtype):
    bs, hidden, num_experts, topk = (
        binding["bs"], binding["hidden"], binding["num_experts"], binding["topk"],
    )
    dt = str(dtype)
    block_n = max(16, 1 << (num_experts - 1).bit_length())
    block_k = min(64, hidden)
    return {
        "inputs": {
            "x": {"shape": [bs, hidden], "dtype": dt},
            "router_weight": {"shape": [num_experts, hidden], "dtype": dt},
            "topk": {"scalar": topk},
            "moe_softcapping": {"scalar": 0.0},
            "BLOCK_SIZE_M": {"scalar": 16},
            "BLOCK_SIZE_N": {"scalar": block_n},
            "BLOCK_SIZE_K": {"scalar": block_k},
            "correction_bias": {"scalar": None},
        },
        "outputs": {
            "topk_weights": {"shape": [bs, topk], "dtype": "torch.float32"},
            "topk_ids": {"shape": [bs, topk], "dtype": "torch.int32"},
        },
        "dims": {
            "bs": bs, "hidden": hidden,
            "num_experts": num_experts, "topk": topk,
            "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k,
        },
    }
