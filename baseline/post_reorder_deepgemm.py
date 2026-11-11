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

"""Native SGLang collection workload for post_reorder_deepgemm."""

import torch

OP_NAME = "post_reorder_deepgemm"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (1,)

SHAPES = [(t, 8, 7168, 32) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 8, 7168, 32) for t in (2048,)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            post_reorder_deepgemm as _f,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _op(
        down_output,
        out,
        src2dst,
        topk_ids,
        topk_weights,
        topk,
        num_tokens,
        hidden_size,
        routed_scaling_factor,
    ):
        _f(
            down_output,
            out,
            src2dst,
            topk_ids,
            topk_weights,
            topk,
            num_tokens,
            hidden_size,
            routed_scaling_factor,
        )
        return out

    return _op


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "topk": topk,
            "hidden": hidden,
            "num_experts": num_experts,
        }
        for (num_tokens, topk, hidden, num_experts) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, topk, hidden, num_experts = shape
    g = torch.Generator(device=device).manual_seed(9)
    n = num_tokens * topk
    down = torch.randn(n, hidden, device=device, generator=g).to(cur_dtype)
    ids = torch.randint(
        -1,
        num_experts + 1,
        (num_tokens, topk),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    dst = (
        torch.randperm(
            n, generator=torch.Generator(device="cpu").manual_seed(9)
        )
        .to(device)
        .reshape(num_tokens, topk)
    )
    dst = torch.where(ids >= 0, dst.to(torch.int32), torch.zeros_like(ids))
    output = torch.zeros(num_tokens, hidden, dtype=cur_dtype, device=device)
    weights = torch.rand(
        num_tokens, topk, dtype=torch.float32, device=device, generator=g
    )
    return (
        down,
        output,
        dst.to(torch.int32),
        ids,
        weights,
        topk,
        num_tokens,
        hidden,
        1.0,
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["topk"],
        binding["hidden"],
        binding["num_experts"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_topk{binding['topk']}_h{binding['hidden']}_E{binding['num_experts']}"


def config(binding, dtype):
    num_tokens, topk, hidden, num_experts = (
        binding["num_tokens"],
        binding["topk"],
        binding["hidden"],
        binding["num_experts"],
    )
    n = num_tokens * topk
    dt = str(dtype)
    return {
        "inputs": {
            "down_output": {"shape": [n, hidden], "dtype": dt},
            "out": {
                "shape": [num_tokens, hidden],
                "dtype": dt,
                "note": "原地写回（output.clone()）",
            },
            "src2dst": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "topk_ids": {"shape": [num_tokens, topk], "dtype": "torch.int32"},
            "topk_weights": {
                "shape": [num_tokens, topk],
                "dtype": "torch.float32",
            },
            "topk": {"scalar": topk},
            "num_tokens": {"scalar": num_tokens},
            "hidden_size": {"scalar": hidden},
            "routed_scaling_factor": {"scalar": 1.0},
        },
        "outputs": {
            "out": {"shape": [num_tokens, hidden], "dtype": dt},
        },
        "dims": {
            "num_tokens": num_tokens,
            "topk": topk,
            "hidden": hidden,
            "num_experts": num_experts,
            "n": n,
        },
    }
