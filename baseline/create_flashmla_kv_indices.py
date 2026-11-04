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

"""create_flashmla_kv_indices baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.kvcache.kv_indices.create_flashmla_kv_indices_triton
    是裸 triton kernel，baseline.py 用二维 grid 启动并写 out：
        out = kv_indices.clone()
        bs, width = out.shape
        grid = (bs, get_num_kv_index_blocks_flashmla(width, page_size))
        create_flashmla_kv_indices_triton[grid](req_to_token, req_pool_indices,
            page_kernel_lens, kv_start_idx, out, req_to_token.stride(0),
            out.stride(0), PAGED_SIZE=page_size)
    native() 返回复刻该启动的 wrapper。kernel 写 out -> IS_INPLACE=True。

输入构造复刻 cases.py 的 BENCH_CASES：
    req_to_token = randint(0, 1<<22, (max_batch, max_context)) int32
    req_pool_indices = randperm(max_batch)[:bs] int32
    page_kernel_lens = randint(1, max_len+1, (bs,)) int32; kv_start_idx=None
    width = ceil(max_len/page_size); kv_indices = zeros(bs, width) int32
    其中 max_len=16384, page_size=64, max_batch=256, max_context=131072，bs in (1,8,32,128)。
DTYPES 记 int32。
"""

import torch

OP_NAME = "create_flashmla_kv_indices"
DTYPES = [torch.int32]
IS_INPLACE = True  # kernel 写 kv_indices（out）

_MAX_LEN = 16384
_PAGE_SIZE = 64
_MAX_BATCH = 256
_MAX_CONTEXT = 131072
_BS = [1, 8, 32, 128]


def native():
    """解析裸 kernel 并返回复刻 baseline 启动的 wrapper。"""
    try:
        from sglang.kernels.ops.kvcache.kv_indices import (
            create_flashmla_kv_indices_triton as _kernel,
            get_num_kv_index_blocks_flashmla as _nblocks,
        )
    except Exception:
        return None

    def _call(req_to_token, req_pool_indices, page_kernel_lens,
              kv_start_idx, kv_indices, page_size):
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
        {"bs": b, "max_len": _MAX_LEN, "page_size": _PAGE_SIZE,
         "max_batch": _MAX_BATCH, "max_context": _MAX_CONTEXT}
        for b in _BS
    ]


def build_inputs(binding, dtype, device):
    bs, max_len, page_size = binding["bs"], binding["max_len"], binding["page_size"]
    max_batch, max_context = binding["max_batch"], binding["max_context"]
    g = torch.Generator(device=device).manual_seed(7)
    req_to_token = torch.randint(
        0, 1 << 22, (max_batch, max_context), dtype=torch.int32, device=device, generator=g
    )
    req_pool_indices = torch.randperm(
        max_batch, device=device, generator=g
    )[:bs].to(torch.int32)
    lens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    width = (max_len + page_size - 1) // page_size
    kv_indices = torch.zeros(bs, width, dtype=torch.int32, device=device)
    return (req_to_token, req_pool_indices, lens, None, kv_indices, page_size), {}


def key_shape(binding):
    return (f"bs{binding['bs']}_ml{binding['max_len']}_ps{binding['page_size']}"
            f"_mb{binding['max_batch']}_mc{binding['max_context']}")


def config(binding, dtype):
    bs, max_len, page_size = binding["bs"], binding["max_len"], binding["page_size"]
    max_batch, max_context = binding["max_batch"], binding["max_context"]
    width = (max_len + page_size - 1) // page_size
    return {
        "inputs": {
            "req_to_token": {"shape": [max_batch, max_context], "dtype": "torch.int32"},
            "req_pool_indices": {"shape": [bs], "dtype": "torch.int32"},
            "page_kernel_lens": {"shape": [bs], "dtype": "torch.int32"},
            "kv_start_idx": {"scalar": None},
            "kv_indices": {"shape": [bs, width], "dtype": "torch.int32",
                           "note": "原地写回"},
            "page_size": {"scalar": page_size},
        },
        "outputs": {
            "kv_indices": {"shape": [bs, width], "dtype": "torch.int32"},
        },
        "dims": {
            "bs": bs, "max_len": max_len, "page_size": page_size, "width": width,
            "max_batch": max_batch, "max_context": max_context,
        },
    }
