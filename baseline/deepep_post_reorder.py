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

"""Native SGLang collection workload for deepep_post_reorder."""

import torch

OP_NAME = "deepep_post_reorder"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
_BLOCK_SIZE = 512
MUTATED_INPUTS = (1,)

SHAPES = [(t, 8, 7168, 1.0) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(t, 8, 7168, 1.0) for t in (2048,)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            deepep_post_reorder_triton_kernel as _k,
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
        hidden_size,
        routed_scaling_factor,
    ):
        _k[(out.shape[0],)](
            down_output,
            out,
            src2dst,
            topk_ids,
            topk_weights,
            topk,
            hidden_size,
            routed_scaling_factor,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        return out

    return _op


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "topk": topk,
            "hidden": hidden,
            "scaling": scaling,
        }
        for (num_tokens, topk, hidden, scaling) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, topk, hidden, scaling = shape
    g = torch.Generator(device=device).manual_seed(0)
    n = num_tokens * topk
    down = torch.randn(
        n, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    src2dst = (
        torch.randperm(
            n, generator=torch.Generator(device="cpu").manual_seed(0)
        )
        .reshape(num_tokens, topk)
        .to(torch.int32)
        .to(device)
    )
    output = torch.zeros(num_tokens, hidden, dtype=cur_dtype, device=device)
    topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)
    topk_weights = torch.rand(
        num_tokens, topk, dtype=torch.float32, device=device, generator=g
    )
    return (
        down,
        output,
        src2dst,
        topk_ids,
        topk_weights,
        topk,
        hidden,
        scaling,
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["topk"],
        binding["hidden"],
        binding["scaling"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return (
        f"t{binding['num_tokens']}_topk{binding['topk']}_h{binding['hidden']}"
    )


def config(binding, dtype):
    num_tokens, topk, hidden = (
        binding["num_tokens"],
        binding["topk"],
        binding["hidden"],
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
            "hidden_size": {"scalar": hidden},
            "routed_scaling_factor": {"scalar": binding["scaling"]},
        },
        "outputs": {
            "out": {"shape": [num_tokens, hidden], "dtype": dt},
        },
        "dims": {
            "num_tokens": num_tokens,
            "topk": topk,
            "hidden": hidden,
            "n": n,
        },
    }
