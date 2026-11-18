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

"""Native SGLang collection workload for zero_experts_identity."""

import torch

OP_NAME = "zero_experts_identity"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (0, 1)

SHAPES = [(t, 8, 7168, 32) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 8, 7168, 32) for t in (4096,)]
_NUM_ZERO = 8


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            zero_experts_compute_triton as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "top_k": top_k,
            "hidden": hidden,
            "num_experts": num_experts,
            **{"num_zero": _NUM_ZERO},
        }
        for (num_tokens, top_k, hidden, num_experts) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, top_k, hidden, num_experts = shape
    g = torch.Generator(device=device).manual_seed(9)
    idx = torch.randint(
        0,
        num_experts + _NUM_ZERO,
        (num_tokens, top_k),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    scales = torch.rand(
        num_tokens, top_k, dtype=torch.float32, device=device, generator=g
    )
    h = torch.randn(num_tokens, hidden, device=device, generator=g).to(
        cur_dtype
    )
    # zero_expert_type is a str, which unpack_to_args_kwargs only forwards as
    # part of a kwargs dict.
    return (
        idx,
        scales,
        num_experts,
        {
            "zero_expert_type": "identity",
            "hidden_states": h,
        },
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["top_k"],
        binding["hidden"],
        binding["num_experts"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return (*args, kwargs["zero_expert_type"], kwargs["hidden_states"]), {}


def key_shape(binding):
    return (
        f"t{binding['num_tokens']}_tk{binding['top_k']}_h{binding['hidden']}"
        f"_E{binding['num_experts']}_z{binding['num_zero']}"
    )


def config(binding, dtype):
    t, top_k, hidden, num_experts, num_zero = (
        binding["num_tokens"],
        binding["top_k"],
        binding["hidden"],
        binding["num_experts"],
        binding["num_zero"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "expert_indices": {
                "shape": [t, top_k],
                "dtype": "torch.int32",
                "note": "原地写回（clone）",
            },
            "expert_scales": {
                "shape": [t, top_k],
                "dtype": "torch.float32",
                "note": "原地写回（clone）",
            },
            "num_experts": {"scalar": num_experts},
            "zero_expert_type": {"scalar": "identity"},
            "hidden_states": {"shape": [t, hidden], "dtype": dt},
        },
        "outputs": {
            "hidden_states_out": {"shape": [t, hidden], "dtype": dt},
        },
        "dims": {
            "num_tokens": t,
            "top_k": top_k,
            "hidden": hidden,
            "num_experts": num_experts,
            "num_zero": num_zero,
        },
    }
