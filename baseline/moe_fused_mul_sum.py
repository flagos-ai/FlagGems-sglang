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

"""moe_fused_mul_sum baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.moe.moe_fused_mul_sum.moe_fused_mul_sum
    调用形态复刻 problems/moe/moe_fused_mul_sum/baseline.py：
        moe_fused_mul_sum(inputs, topk_weights, topk_ids=None, expert_map=None,
                          routed_scaling_factor=None, is_ep=False)
    另分配输出，不原地写回 -> IS_INPLACE=False。

输入构造复刻 problems/moe/moe_fused_mul_sum/cases.py 的 BENCH_CASES：
    inputs=randn(num_tokens, top_k, size).to(dtype);
    topk_weights=rand(num_tokens, top_k).to(dtype)
    top_k=8, size=4096；topk_ids/expert_map=None, routed_scaling_factor=None,
    is_ep=False（BENCH 走非 EP、无 expert_map 路径）。
dtype 取 benchmark/test_moe_fused_mul_sum.py 的 [bfloat16]。
"""

import torch

OP_NAME = "moe_fused_mul_sum"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [(m, 8, 4096) for m in (1, 8, 64, 512, 4096)]


def native():
    """解析 sglang.kernels.ops.moe.moe_fused_mul_sum.moe_fused_mul_sum。"""
    try:
        from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"num_tokens": m, "top_k": tk, "size": s}
        for (m, tk, s) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    num_tokens, top_k, size = (
        binding["num_tokens"], binding["top_k"], binding["size"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    inputs = torch.randn(
        num_tokens, top_k, size, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    topk_weights = torch.rand(
        num_tokens, top_k, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    kwargs = dict(
        topk_ids=None,
        expert_map=None,
        routed_scaling_factor=None,
        is_ep=False,
    )
    return (inputs, topk_weights), kwargs


def key_shape(binding):
    return [binding["num_tokens"], binding["top_k"], binding["size"]]


def config(binding, dtype):
    num_tokens, top_k, size = (
        binding["num_tokens"], binding["top_k"], binding["size"],
    )
    dt = str(dtype)
    # 沿 top_k 维加权求和（weighted mul-sum），输出 [num_tokens, size]。
    return {
        "inputs": {
            "inputs": {"shape": [num_tokens, top_k, size], "dtype": dt},
            "topk_weights": {"shape": [num_tokens, top_k], "dtype": dt},
            "topk_ids": {"scalar": None},
            "expert_map": {"scalar": None},
            "routed_scaling_factor": {"scalar": None},
            "is_ep": {"scalar": False},
        },
        "outputs": {
            "output": {"shape": [num_tokens, size], "dtype": dt},
        },
        "dims": {"num_tokens": num_tokens, "top_k": top_k, "size": size},
    }
