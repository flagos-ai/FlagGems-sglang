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

"""build_trtllm_mha_page_table baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.kvcache.trtllm_mha_page_table.build_trtllm_mha_page_table
    调用形态复刻 problems/kvcache/build_trtllm_mha_page_table/baseline.py：
        out = page_table.clone()
        build_trtllm_mha_page_table(req_to_token, req_pool_indices,
            cache_seqlens, out, page_size)
    kernel 写 out（page_table）-> IS_INPLACE=True。

输入构造复刻 cases.py 的 BENCH_CASES：
    req_to_token = randint(0, 1<<22, (max_batch, max_context)) int32
    req_pool_indices = randperm(max_batch)[:bs] int32
    cache_seqlens = randint(1, max_len+1, (bs,)) int32
    width = ceil(max_len/page_size); page_table = zeros(bs, width) int32
    其中 max_len=16384, page_size=64, max_batch=256, max_context=131072，bs in (1,8,32,128)。
DTYPES 记 int32。
"""

import torch

OP_NAME = "build_trtllm_mha_page_table"
DTYPES = [torch.int32]
IS_INPLACE = True  # kernel 写 page_table（out）

_MAX_LEN = 16384
_PAGE_SIZE = 64
_MAX_BATCH = 256
_MAX_CONTEXT = 131072
_BS = [1, 8, 32, 128]


def native():
    """解析 build_trtllm_mha_page_table。"""
    try:
        from sglang.kernels.ops.kvcache.trtllm_mha_page_table import (
            build_trtllm_mha_page_table as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


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
    seqlens = torch.randint(
        1, max_len + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    width = (max_len + page_size - 1) // page_size
    page_table = torch.zeros(bs, width, dtype=torch.int32, device=device)
    return (req_to_token, req_pool_indices, seqlens, page_table, page_size), {}


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
            "cache_seqlens": {"shape": [bs], "dtype": "torch.int32"},
            "page_table": {"shape": [bs, width], "dtype": "torch.int32",
                           "note": "原地写回"},
            "page_size": {"scalar": page_size},
        },
        "outputs": {
            "page_table": {"shape": [bs, width], "dtype": "torch.int32"},
        },
        "dims": {
            "bs": bs, "max_len": max_len, "page_size": page_size, "width": width,
            "max_batch": max_batch, "max_context": max_context,
        },
    }
