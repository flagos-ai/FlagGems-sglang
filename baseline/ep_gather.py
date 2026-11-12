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

"""Native SGLang collection workload for ep_gather."""

import torch

OP_NAME = "ep_gather"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (4,)

SHAPES = [(t, 4096, 8, max(t * 8, 16)) for t in (1, 8, 64, 512)]
MORE_SHAPES = [(4096, 4096, 8, 4096 * 8)]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import ep_gather as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "num_tokens": num_tokens,
            "hidden": hidden,
            "topk": topk,
            "num_src": num_src,
        }
        for (num_tokens, hidden, topk, num_src) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    num_tokens, hidden, topk, num_src = shape
    g = torch.Generator(device=device).manual_seed(9)
    input_tensor = torch.randn(
        num_src, hidden, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    ids = torch.randint(
        0,
        32,
        (num_tokens, topk),
        device=device,
        generator=g,
        dtype=torch.int32,
    )
    weight = torch.rand(
        num_tokens, topk, dtype=torch.float32, device=device, generator=g
    )
    index = torch.randint(
        0,
        num_src,
        (num_tokens, topk),
        device=device,
        generator=g,
        dtype=torch.int32,
    )
    out = torch.zeros(num_tokens, hidden, dtype=cur_dtype, device=device)
    return input_tensor, ids, weight, index, out


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_tokens"],
        binding["hidden"],
        binding["topk"],
        binding["num_src"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_h{binding['hidden']}_topk{binding['topk']}_src{binding['num_src']}"


def config(binding, dtype):
    num_tokens, hidden, topk, num_src = (
        binding["num_tokens"],
        binding["hidden"],
        binding["topk"],
        binding["num_src"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "input_tensor": {"shape": [num_src, hidden], "dtype": dt},
            "recv_topk_ids": {
                "shape": [num_tokens, topk],
                "dtype": "torch.int32",
            },
            "recv_topk_weight": {
                "shape": [num_tokens, topk],
                "dtype": "torch.float32",
            },
            "input_index": {
                "shape": [num_tokens, topk],
                "dtype": "torch.int32",
            },
            "output_tensor": {
                "shape": [num_tokens, hidden],
                "dtype": dt,
                "note": "原地写回",
            },
        },
        "outputs": {
            "output_tensor": {"shape": [num_tokens, hidden], "dtype": dt},
        },
        "dims": {
            "num_tokens": num_tokens,
            "hidden": hidden,
            "topk": topk,
            "num_src": num_src,
        },
    }
