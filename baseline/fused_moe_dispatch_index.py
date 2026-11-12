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

"""Native SGLang collection workload for fused_moe_dispatch_index."""

import torch

OP_NAME = "fused_moe_dispatch_index"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(t, 8, 32, 8192) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 8, 32, 8192) for t in (4096,)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            fused_moe_dispatch_index as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "num_tokens": total_tokens,
            "topk": topk,
            "num_local_experts": num_local_experts,
            "m_max": m_max,
        }
        for (total_tokens, topk, num_local_experts, m_max) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    total_tokens, topk, num_local_experts, m_max = shape
    g = torch.Generator(device=device).manual_seed(0)
    topk_ids = torch.randint(
        0,
        num_local_experts,
        (total_tokens, topk),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    return topk_ids, num_local_experts, m_max


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["topk"],
        binding["num_local_experts"],
        binding["m_max"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_topk{binding['topk']}_E{binding['num_local_experts']}_mmax{binding['m_max']}"


def config(binding, dtype):
    t, topk, e, m_max = (
        binding["num_tokens"],
        binding["topk"],
        binding["num_local_experts"],
        binding["m_max"],
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
        "dims": {
            "num_tokens": t,
            "topk": topk,
            "num_local_experts": e,
            "m_max": m_max,
        },
    }
