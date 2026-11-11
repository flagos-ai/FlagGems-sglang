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

"""Native SGLang collection workload for fused_moe_router_cudacore."""

import torch

OP_NAME = "fused_moe_router_cudacore"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(m, 4096, 256, 8, 0.0, False) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [
    (1, 64, 8, 1, 0.0, False),
    (37, 256, 16, 2, 30.0, False),
    (83, 512, 32, 3, 0.0, True),
]


def native():
    """解析 sglang.kernels.ops.moe.router.fused_moe_router_cudacore。"""
    try:
        from sglang.kernels.ops.moe.router import (
            fused_moe_router_cudacore as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "bs": bs,
            "hidden": hidden_dim,
            "num_experts": num_experts,
            "topk": topk,
            "softcap": softcap,
            "has_bias": has_bias,
        }
        for (bs, hidden_dim, num_experts, topk, softcap, has_bias) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    bs, hidden_dim, num_experts, topk, softcap, has_bias = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        bs, hidden_dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    router_weight = torch.randn(
        num_experts,
        hidden_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    bias = None
    if has_bias:
        bias = (
            torch.randn(
                num_experts, generator=g, device=device, dtype=torch.float32
            )
            * 0.1
        )
    return x, router_weight, topk, softcap, bias


def build_inputs(binding, dtype, device):
    shape = (
        binding["bs"],
        binding["hidden"],
        binding["num_experts"],
        binding["topk"],
        binding["softcap"],
        binding["has_bias"],
    )
    *args, bias = build_case(shape, dtype, device)
    return tuple(args), {"correction_bias": bias}


def key_shape(binding):
    return [
        binding["bs"],
        binding["hidden"],
        binding["num_experts"],
        binding["topk"],
        binding["softcap"],
        binding["has_bias"],
    ]


def config(binding, dtype):
    bs, hidden, num_experts, topk = (
        binding["bs"],
        binding["hidden"],
        binding["num_experts"],
        binding["topk"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [bs, hidden], "dtype": dt},
            "router_weight": {"shape": [num_experts, hidden], "dtype": dt},
            "topk": {"scalar": topk},
            "moe_softcapping": {"scalar": binding["softcap"]},
            "correction_bias": (
                {"shape": [num_experts], "dtype": "torch.float32"}
                if binding["has_bias"]
                else {"scalar": None}
            ),
        },
        "outputs": {
            "topk_weights": {"shape": [bs, topk], "dtype": "torch.float32"},
            "topk_ids": {"shape": [bs, topk], "dtype": "torch.int32"},
        },
        "dims": {
            "bs": bs,
            "hidden": hidden,
            "num_experts": num_experts,
            "topk": topk,
        },
    }
