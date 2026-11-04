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

"""moe_fused_gate baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.moe.moe_fused_gate.moe_fused_gate
    调用形态复刻 problems/moe/moe_fused_gate/baseline.py：
        moe_fused_gate(scores, bias, topk, scoring_func=..., num_fused_shared_experts=...,
                       renormalize=..., routed_scaling_factor=...,
                       apply_routed_scaling_factor_on_output=..., moe_softcapping=...,
                       num_expert_group=..., topk_group=...)
    另分配输出（topk_weights, topk_ids），不原地写回 -> IS_INPLACE=False。

输入构造复刻 problems/moe/moe_fused_gate/cases.py 的 BENCH_CASES：
    scores=randn(m, n).to(dtype); bias=0.1*randn(n)
    topk=8, n=256, num_fused_shared_experts=1, num_expert_group=8, topk_group=4，
    其余参数取 _case 缺省（scoring_func="sigmoid", renormalize=True,
    routed_scaling_factor=1.0, apply_routed_scaling_factor_on_output=False,
    moe_softcapping=0.0）。
dtype 取 benchmark/test_moe_fused_gate.py 的 [bfloat16]。
"""

import torch

OP_NAME = "moe_fused_gate"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# (m, n, topk, num_fused_shared_experts, num_expert_group, topk_group)
_SHAPES = [(m, 256, 8, 1, 8, 4) for m in (1, 8, 64, 512, 4096)]


def native():
    """解析 sglang.kernels.ops.moe.moe_fused_gate.moe_fused_gate。"""
    try:
        from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "n": n, "topk": tk, "nfse": nfse, "neg": neg, "tkg": tkg}
        for (m, n, tk, nfse, neg, tkg) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    m, n, topk, nfse, neg, tkg = (
        binding["m"], binding["n"], binding["topk"],
        binding["nfse"], binding["neg"], binding["tkg"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    scores = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(dtype)
    bias = 0.1 * torch.randn(n, generator=g, device=device, dtype=torch.float32)
    kwargs = dict(
        scoring_func="sigmoid",
        num_fused_shared_experts=nfse,
        renormalize=True,
        routed_scaling_factor=1.0,
        apply_routed_scaling_factor_on_output=False,
        moe_softcapping=0.0,
        num_expert_group=neg,
        topk_group=tkg,
    )
    return (scores, bias, topk), kwargs


def key_shape(binding):
    return [binding["m"], binding["n"], binding["topk"]]


def config(binding, dtype):
    m, n, topk, nfse, neg, tkg = (
        binding["m"], binding["n"], binding["topk"],
        binding["nfse"], binding["neg"], binding["tkg"],
    )
    dt = str(dtype)
    # kernel 输出每 token top-(topk+num_fused_shared_experts) 的权重与索引。
    out_k = topk + nfse
    return {
        "inputs": {
            "scores": {"shape": [m, n], "dtype": dt},
            "bias": {"shape": [n], "dtype": "torch.float32"},
            "topk": {"scalar": topk},
            "scoring_func": {"scalar": "sigmoid"},
            "num_fused_shared_experts": {"scalar": nfse},
            "renormalize": {"scalar": True},
            "routed_scaling_factor": {"scalar": 1.0},
            "apply_routed_scaling_factor_on_output": {"scalar": False},
            "moe_softcapping": {"scalar": 0.0},
            "num_expert_group": {"scalar": neg},
            "topk_group": {"scalar": tkg},
        },
        "outputs": {
            "topk_weights": {"shape": [m, out_k], "dtype": "torch.float32"},
            "topk_ids": {"shape": [m, out_k], "dtype": "torch.int32"},
        },
        "dims": {
            "m": m, "n": n, "topk": topk,
            "num_fused_shared_experts": nfse,
            "num_expert_group": neg, "topk_group": tkg,
        },
    }
