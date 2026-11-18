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

"""Native SGLang v0.5.21 collection workload for fused_moe."""

import torch

OP_NAME = "fused_moe"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (1, 2048, 1408, 64, 8),
    (8, 2048, 1408, 64, 8),
    (64, 2048, 1408, 64, 8),
    (512, 2048, 1408, 64, 8),
]
MORE_SHAPES = [(4096, 2048, 1408, 64, 8)]


def native():
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_moe as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    def run(
        hidden_states, w1, w2, topk_output, activation, routed_scaling_factor
    ):
        runner = MoeRunnerConfig(
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
            inplace=False,
            gate_up_interleaved=False,
        )
        return op(
            hidden_states, w1, w2, StandardTopKOutput(*topk_output), runner
        )

    return run


def binding_for_case(shape):
    return dict(
        zip(("num_tokens", "hidden", "inter", "num_experts", "topk"), shape)
    )


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    num_tokens, hidden, inter, num_experts, topk = shape
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(num_tokens, hidden, device=device, generator=g).to(
        cur_dtype
    )
    w1 = (
        torch.randn(num_experts, 2 * inter, hidden, device=device, generator=g)
        * 0.05
    ).to(cur_dtype)
    w2 = (
        torch.randn(num_experts, hidden, inter, device=device, generator=g)
        * 0.05
    ).to(cur_dtype)
    logits = torch.randn(num_tokens, num_experts, device=device, generator=g)
    probs = torch.softmax(logits, dim=-1)
    topk_weights, topk_ids = torch.topk(probs, topk, dim=-1)
    topk_output = (
        topk_weights.float().contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        None,
    )
    return (
        x,
        w1,
        w2,
        topk_output,
        {
            "activation": "silu",
            "routed_scaling_factor": 1.0,
        },
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["hidden"],
        binding["inter"],
        binding["num_experts"],
        binding["topk"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return tuple(args), kwargs


def key_shape(binding):
    return [
        binding["num_tokens"],
        binding["hidden"],
        binding["inter"],
        binding["num_experts"],
        binding["topk"],
    ]


def config(binding, dtype):
    num_tokens, hidden, inter, num_experts, topk = (
        binding["num_tokens"],
        binding["hidden"],
        binding["inter"],
        binding["num_experts"],
        binding["topk"],
    )

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "hidden_states": tensor([num_tokens, hidden]),
            "w1": tensor([num_experts, 2 * inter, hidden]),
            "w2": tensor([num_experts, hidden, inter]),
            "topk_weights": tensor([num_tokens, topk], "torch.float32"),
            "topk_ids": tensor([num_tokens, topk], "torch.int32"),
            "activation": {"scalar": "silu"},
            "routed_scaling_factor": {"scalar": 1.0},
        },
        "outputs": {"out": tensor([num_tokens, hidden])},
        "dims": dict(binding),
    }
