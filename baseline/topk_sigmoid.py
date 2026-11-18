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

"""Native SGLang collection workload for topk_sigmoid."""

import torch

OP_NAME = "topk_sigmoid"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (0, 1)

SHAPES = [(t, 256, 8) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 256, 8) for t in (4096,)]


def native():
    try:
        from sglang.kernels.ops.moe.moe_topk_sigmoid import topk_sigmoid as _f
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _op(w, ids, gating_output, renormalize, routed_scaling_factor):
        _f(
            w,
            ids,
            gating_output,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )
        return w, ids

    return _op


def grid():
    return [
        {"num_tokens": num_tokens, "num_experts": num_experts, "k": k}
        for (num_tokens, num_experts, k) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, num_experts, k = shape
    g = torch.Generator(device=device).manual_seed(9)
    # fp32 logits keep the top-k ordering well defined; see the test module.
    gating = torch.randn(num_tokens, num_experts, device=device, generator=g)
    topk_weights = torch.zeros(
        num_tokens, k, dtype=torch.float32, device=device
    )
    topk_ids = torch.zeros(num_tokens, k, dtype=torch.int32, device=device)
    # renormalize is a bool, which unpack_to_args_kwargs only forwards as
    # part of a kwargs dict.
    return (
        topk_weights,
        topk_ids,
        gating,
        {
            "renormalize": True,
            "routed_scaling_factor": 2.5,
        },
    )


def build_inputs(binding, dtype, device):
    shape = (binding["num_tokens"], binding["num_experts"], binding["k"])
    weights, ids, gating, kwargs = build_case(shape, dtype, device)
    return (
        weights,
        ids,
        gating,
        kwargs["renormalize"],
        kwargs["routed_scaling_factor"],
    ), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["num_experts"], binding["k"]]


def config(binding, dtype):
    t, num_experts, k = (
        binding["num_tokens"],
        binding["num_experts"],
        binding["k"],
    )
    return {
        "inputs": {
            "topk_weights": {
                "shape": [t, k],
                "dtype": "torch.float32",
                "note": "原地写回（clone）",
            },
            "topk_ids": {
                "shape": [t, k],
                "dtype": "torch.int32",
                "note": "原地写回（clone）",
            },
            "gating_output": {
                "shape": [t, num_experts],
                "dtype": "torch.float32",
            },
            "renormalize": {"scalar": True},
            "routed_scaling_factor": {"scalar": 2.5},
        },
        "outputs": {
            "topk_weights": {"shape": [t, k], "dtype": "torch.float32"},
            "topk_ids": {"shape": [t, k], "dtype": "torch.int32"},
        },
        "dims": {"num_tokens": t, "num_experts": num_experts, "k": k},
    }
