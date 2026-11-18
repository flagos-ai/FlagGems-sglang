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

"""Native SGLang v0.5.21 collection workload for pack_topk_ids."""

import torch

OP_NAME = "pack_topk_ids"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(t, 8, 256) for t in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(t, 8, 256) for t in (16384,)]


def native():
    try:
        from sglang.kernels.ops.moe.trtllm_lora_temp.topk_pack import (
            fused_pack_topk as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("num_tokens", "topk", "num_experts"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    num_tokens, topk, num_experts = shape
    g = torch.Generator(device=device).manual_seed(9)
    ids = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    weights = torch.rand(
        num_tokens, topk, dtype=torch.float32, device=device, generator=g
    )
    return ids, weights


def build_inputs(binding, dtype, device):
    shape = (binding["num_tokens"], binding["topk"], binding["num_experts"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["num_tokens"], binding["topk"], binding["num_experts"]]


def config(binding, dtype):
    num_tokens, topk = binding["num_tokens"], binding["topk"]

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "topk_ids": tensor([num_tokens, topk], "torch.int32"),
            "topk_weights": tensor([num_tokens, topk], "torch.float32"),
        },
        "outputs": {"packed": tensor([num_tokens, topk], "torch.int32")},
        "dims": dict(binding),
    }
