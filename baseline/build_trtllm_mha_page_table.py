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

"""Native SGLang collection workload for build_trtllm_mha_page_table."""

import torch

OP_NAME = "build_trtllm_mha_page_table"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (3,)

MAX_BATCH = 256
MAX_CONTEXT = 131072

SHAPES = [(bs, 64, 16384) for bs in (1, 8, 32)]
MORE_SHAPES = [(bs, 64, 16384) for bs in (128,)]


def native():
    """解析 build_trtllm_mha_page_table。"""
    try:
        from sglang.kernels.ops.kvcache.trtllm_mha_page_table import (
            build_trtllm_mha_page_table as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "bs": bs,
            "page_size": page_size,
            "max_len": max_len,
            **{"max_batch": MAX_BATCH, "max_context": MAX_CONTEXT},
        }
        for (bs, page_size, max_len) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, page_size, max_len = shape
    max_batch, max_context = MAX_BATCH, MAX_CONTEXT
    g = torch.Generator(device=device).manual_seed(0)
    req_to_token = torch.randint(
        0,
        1 << 22,
        (max_batch, max_context),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    req_pool_indices = (
        torch.randperm(
            max_batch, generator=torch.Generator(device="cpu").manual_seed(0)
        )[:bs]
        .to(torch.int32)
        .to(device)
    )
    cache_seqlens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    width = (max_len + page_size - 1) // page_size
    page_table = torch.zeros(bs, width, dtype=torch.int32, device=device)
    return req_to_token, req_pool_indices, cache_seqlens, page_table, page_size


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["page_size"], binding["max_len"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return (
        f"bs{binding['bs']}_ml{binding['max_len']}_ps{binding['page_size']}"
        f"_mb{binding['max_batch']}_mc{binding['max_context']}"
    )


def config(binding, dtype):
    bs, max_len, page_size = (
        binding["bs"],
        binding["max_len"],
        binding["page_size"],
    )
    max_batch, max_context = binding["max_batch"], binding["max_context"]
    width = (max_len + page_size - 1) // page_size
    return {
        "inputs": {
            "req_to_token": {
                "shape": [max_batch, max_context],
                "dtype": "torch.int32",
            },
            "req_pool_indices": {"shape": [bs], "dtype": "torch.int32"},
            "cache_seqlens": {"shape": [bs], "dtype": "torch.int32"},
            "page_table": {
                "shape": [bs, width],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
            "page_size": {"scalar": page_size},
        },
        "outputs": {
            "page_table": {"shape": [bs, width], "dtype": "torch.int32"},
        },
        "dims": {
            "bs": bs,
            "max_len": max_len,
            "page_size": page_size,
            "width": width,
            "max_batch": max_batch,
            "max_context": max_context,
        },
    }
