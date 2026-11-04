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

"""create_chunked_prefix_cache_kv_indices baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.kvcache.kv_indices.create_chunked_prefix_cache_kv_indices
    是裸 triton kernel，baseline.py 用 kernel[(bs,)](...) 启动并写 out：
        out = chunk_kv_indices.clone()
        create_chunked_prefix_cache_kv_indices[(bs,)](req_to_token,
            req_pool_indices, chunk_start_idx, chunk_seq_lens, chunk_cu_seq_lens,
            out, req_to_token.stride(0))
    native() 返回复刻该启动的 wrapper。kernel 写 out -> IS_INPLACE=True。

输入构造复刻 cases.py 的 BENCH_CASES：
    req_to_token = randint(0, 1<<20, (max_batch, max_context)) int32
    req_pool_indices = randperm(max_batch)[:bs] int32
    chunk_seq_lens = randint(1, chunk+1, (bs,)) int32
    chunk_start_idx = randint(0, max_context-chunk-1, (bs,)) int32
    chunk_cu_seq_lens = [0, cumsum(seq_lens)] int32
    chunk_kv_indices = zeros(sum(seq_lens)) int32
    其中 chunk=2048, max_batch=256, max_context=16384，bs in (1,8,32,128)。
数据相关，key_shape 用字符串。DTYPES 记 int32。
"""

import torch

OP_NAME = "create_chunked_prefix_cache_kv_indices"
DTYPES = [torch.int32]
IS_INPLACE = True  # kernel 写 chunk_kv_indices（out）

_CHUNK = 2048
_MAX_BATCH = 256
_MAX_CONTEXT = 16384
_BS = [1, 8, 32, 128]


def native():
    """解析裸 kernel 并返回复刻 baseline 启动的 wrapper。"""
    try:
        from sglang.kernels.ops.kvcache.kv_indices import (
            create_chunked_prefix_cache_kv_indices as _kernel,
        )
    except Exception:
        return None

    def _call(req_to_token, req_pool_indices, chunk_start_idx,
              chunk_seq_lens, chunk_cu_seq_lens, chunk_kv_indices):
        bs = req_pool_indices.shape[0]
        _kernel[(bs,)](
            req_to_token,
            req_pool_indices,
            chunk_start_idx,
            chunk_seq_lens,
            chunk_cu_seq_lens,
            chunk_kv_indices,
            req_to_token.stride(0),
        )
        return chunk_kv_indices

    return _call


def grid():
    return [
        {"bs": b, "chunk": _CHUNK, "max_batch": _MAX_BATCH, "max_context": _MAX_CONTEXT}
        for b in _BS
    ]


def build_inputs(binding, dtype, device):
    bs, chunk = binding["bs"], binding["chunk"]
    max_batch, max_context = binding["max_batch"], binding["max_context"]
    g = torch.Generator(device=device).manual_seed(7)
    req_to_token = torch.randint(
        0, 1 << 20, (max_batch, max_context), dtype=torch.int32, device=device, generator=g
    )
    req_pool_indices = torch.randperm(
        max_batch, device=device, generator=g
    )[:bs].to(torch.int32)
    seq_lens = torch.randint(
        1, chunk + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    start_idx = torch.randint(
        0, max_context - chunk - 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    chunk_kv_indices = torch.zeros(int(cu[-1]), dtype=torch.int32, device=device)
    return (req_to_token, req_pool_indices, start_idx, seq_lens, cu, chunk_kv_indices), {}


def key_shape(binding):
    return (f"bs{binding['bs']}_chunk{binding['chunk']}"
            f"_mb{binding['max_batch']}_mc{binding['max_context']}")


def config(binding, dtype):
    bs, chunk = binding["bs"], binding["chunk"]
    max_batch, max_context = binding["max_batch"], binding["max_context"]
    return {
        "inputs": {
            "req_to_token": {"shape": [max_batch, max_context], "dtype": "torch.int32"},
            "req_pool_indices": {"shape": [bs], "dtype": "torch.int32"},
            "chunk_start_idx": {"shape": [bs], "dtype": "torch.int32"},
            "chunk_seq_lens": {"shape": [bs], "dtype": "torch.int32"},
            "chunk_cu_seq_lens": {"shape": [bs + 1], "dtype": "torch.int32"},
            "chunk_kv_indices": {"note": "zeros(sum(chunk_seq_lens)) int32，原地写回，数据相关"},
        },
        "outputs": {
            "chunk_kv_indices": {"note": "长度 sum(chunk_seq_lens)，int32，数据相关"},
        },
        "dims": {
            "bs": bs, "chunk": chunk, "max_batch": max_batch,
            "max_context": max_context, "note": "seq_lens 随机，输出长度数据相关",
        },
    }
