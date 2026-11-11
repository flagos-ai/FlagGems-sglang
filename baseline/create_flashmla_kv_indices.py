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

"""Native SGLang collection workload for create_flashmla_kv_indices."""

import torch

OP_NAME = "create_flashmla_kv_indices"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (4,)

SHAPES = [(bs, 16384, 64) for bs in (1, 8, 32)]
MORE_SHAPES = [(bs, 16384, 64) for bs in (128,)]
_MAX_BATCH = 256
_MAX_CONTEXT = 131072


def native():
    """解析裸 kernel 并返回复刻 baseline 启动的 wrapper。"""
    try:
        from sglang.kernels.ops.kvcache.kv_indices import (
            create_flashmla_kv_indices_triton as _kernel,
        )
        from sglang.kernels.ops.kvcache.kv_indices import (
            get_num_kv_index_blocks_flashmla as _nblocks,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _call(
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_start_idx,
        kv_indices,
        page_size,
    ):
        bs, width = kv_indices.shape
        grid = (bs, _nblocks(width, page_size))
        _kernel[grid](
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            kv_start_idx,
            kv_indices,
            req_to_token.stride(0),
            kv_indices.stride(0),
            PAGED_SIZE=page_size,
        )
        return kv_indices

    return _call


def grid():
    return [
        {
            "bs": bs,
            "max_len": max_len,
            "page_size": page_size,
            **{"max_batch": _MAX_BATCH, "max_context": _MAX_CONTEXT},
        }
        for (bs, max_len, page_size) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, max_len, page_size = shape
    g = torch.Generator(device=device).manual_seed(7)
    req_to_token = torch.randint(
        0,
        1 << 22,
        (_MAX_BATCH, _MAX_CONTEXT),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    # torch_gcu randperm hangs on-device; generate on CPU then move.
    req_pool_indices = (
        torch.randperm(
            _MAX_BATCH, generator=torch.Generator(device="cpu").manual_seed(7)
        )[:bs]
        .to(torch.int32)
        .to(device)
    )
    lens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    width = (max_len + page_size - 1) // page_size
    kv_indices = torch.zeros(bs, width, dtype=torch.int32, device=device)
    return req_to_token, req_pool_indices, lens, None, kv_indices, page_size


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["max_len"], binding["page_size"])
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
            "page_kernel_lens": {"shape": [bs], "dtype": "torch.int32"},
            "kv_start_idx": {"scalar": None},
            "kv_indices": {
                "shape": [bs, width],
                "dtype": "torch.int32",
                "note": "原地写回",
            },
            "page_size": {"scalar": page_size},
        },
        "outputs": {
            "kv_indices": {"shape": [bs, width], "dtype": "torch.int32"},
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
