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

"""Native SGLang v0.5.21 collection workload for topk_softmax."""

import torch

OP_NAME = "topk_softmax"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0, 1)

SHAPES = [(t, 256, 8) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(4096, 256, 8)]


def native():
    try:
        from sglang.kernels.ops.moe.moe_topk_softmax import topk_softmax as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("num_tokens", "num_experts", "k"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    num_tokens, num_experts, k = shape
    g = torch.Generator(device=device).manual_seed(9)
    # fp32 logits keep the top-k ordering well defined; see the test module.
    gating = torch.randn(num_tokens, num_experts, device=device, generator=g)
    topk_weights = torch.zeros(
        num_tokens, k, dtype=torch.float32, device=device
    )
    topk_ids = torch.zeros(num_tokens, k, dtype=torch.int32, device=device)
    return topk_weights, topk_ids, gating, {"renormalize": True}


def build_inputs(binding, dtype, device):
    shape = (binding["num_tokens"], binding["num_experts"], binding["k"])
    *args, kwargs = build_case(shape, dtype, device)
    return tuple(args), kwargs


def key_shape(binding):
    return [binding["num_tokens"], binding["num_experts"], binding["k"]]


def config(binding, dtype):
    num_tokens, num_experts, k = (
        binding["num_tokens"],
        binding["num_experts"],
        binding["k"],
    )

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "topk_weights": tensor([num_tokens, k], "torch.float32"),
            "topk_ids": tensor([num_tokens, k], "torch.int32"),
            "gating_output": tensor(
                [num_tokens, num_experts], "torch.float32"
            ),
            "renormalize": {"scalar": True},
        },
        "outputs": {
            "topk_weights": tensor([num_tokens, k], "torch.float32"),
            "topk_ids": tensor([num_tokens, k], "torch.int32"),
        },
        "dims": dict(binding),
    }
