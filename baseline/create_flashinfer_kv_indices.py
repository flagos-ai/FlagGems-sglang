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

"""Native SGLang collection workload for create_flashinfer_kv_indices."""

import torch

OP_NAME = "create_flashinfer_kv_indices"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (5,)

SHAPES = [(bs, 16, 4096) for bs in (1, 8, 32)]
MORE_SHAPES = [(bs, 16, 4096) for bs in (128,)]


def native():
    """解析裸 kernel 并返回复刻 baseline 启动的 wrapper。"""
    try:
        from sglang.kernels.ops.kvcache.kv_indices import (
            create_flashinfer_kv_indices_triton as _kernel,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _call(
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
    ):
        bs = req_pool_indices.shape[0]
        _kernel[(bs,)](
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            req_to_token.stride(0),
        )
        return kv_indices

    return _call


def grid():
    return [
        {
            "bs": bs,
            "page_size": page_size,
            "max_len": max_len,
            **{"max_batch": 256, "max_context": 8192},
        }
        for (bs, page_size, max_len) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, page_size, max_len = shape
    max_batch, max_context = 256, 8192
    g = torch.Generator(device=device).manual_seed(0)
    req_to_token = torch.randint(
        0,
        1 << 20,
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
    lens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(lens, dim=0)
    kv_start_idx = None
    kv_indices = torch.zeros(
        int(kv_indptr[-1]), dtype=torch.int32, device=device
    )
    return (
        req_to_token,
        req_pool_indices,
        lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
    )


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["page_size"], binding["max_len"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return (
        f"bs{binding['bs']}_ml{binding['max_len']}"
        f"_mb{binding['max_batch']}_mc{binding['max_context']}"
    )


def config(binding, dtype):
    bs, max_len = binding["bs"], binding["max_len"]
    max_batch, max_context = binding["max_batch"], binding["max_context"]
    return {
        "inputs": {
            "req_to_token": {
                "shape": [max_batch, max_context],
                "dtype": "torch.int32",
            },
            "req_pool_indices": {"shape": [bs], "dtype": "torch.int32"},
            "page_kernel_lens": {"shape": [bs], "dtype": "torch.int32"},
            "kv_indptr": {"shape": [bs + 1], "dtype": "torch.int32"},
            "kv_start_idx": {"scalar": None},
            "kv_indices": {
                "shape": ["total_kv"],
                "dtype": "torch.int32",
                "note": "zeros(sum(page_kernel_lens)) int32，原地写回，数据相关",
            },
        },
        "outputs": {
            "kv_indices": {
                "shape": ["total_kv"],
                "dtype": "torch.int32",
                "note": "长度 sum(page_kernel_lens)，int32，数据相关",
            },
        },
        "dims": {
            "bs": bs,
            "max_len": max_len,
            "max_batch": max_batch,
            "max_context": max_context,
            "note": "lens 随机，kv_indices 长度数据相关",
        },
    }
