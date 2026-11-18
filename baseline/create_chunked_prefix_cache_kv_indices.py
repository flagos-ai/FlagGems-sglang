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

"""Native SGLang collection workload for create_chunked_prefix_cache_kv_indices."""

import torch

OP_NAME = "create_chunked_prefix_cache_kv_indices"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (5,)

SHAPES = [(bs, 2048) for bs in (1, 8, 32)]
MORE_SHAPES = [(bs, 2048) for bs in (128,)]
_MAX_BATCH = 256
_MAX_CONTEXT = 16384


def native():
    """解析裸 kernel 并返回复刻 baseline 启动的 wrapper。"""
    try:
        from sglang.kernels.ops.kvcache.kv_indices import (
            create_chunked_prefix_cache_kv_indices as _kernel,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _call(
        req_to_token,
        req_pool_indices,
        chunk_start_idx,
        chunk_seq_lens,
        chunk_cu_seq_lens,
        chunk_kv_indices,
    ):
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
        {
            "bs": bs,
            "chunk": chunk,
            **{"max_batch": _MAX_BATCH, "max_context": _MAX_CONTEXT},
        }
        for (bs, chunk) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    bs, chunk = shape
    g = torch.Generator(device=device).manual_seed(7)
    req_to_token = torch.randint(
        0,
        1 << 20,
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
    seq_lens = torch.randint(
        1, chunk + 1, (bs,), dtype=torch.int32, device=device, generator=g
    )
    start_idx = torch.randint(
        0,
        _MAX_CONTEXT - chunk - 1,
        (bs,),
        dtype=torch.int32,
        device=device,
        generator=g,
    )
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.zeros(int(cu[-1]), dtype=torch.int32, device=device)
    return (
        req_to_token,
        req_pool_indices,
        start_idx,
        seq_lens,
        cu,
        kv_indices,
    )


def build_inputs(binding, dtype, device):
    shape = (binding["bs"], binding["chunk"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return (
        f"bs{binding['bs']}_chunk{binding['chunk']}"
        f"_mb{binding['max_batch']}_mc{binding['max_context']}"
    )


def config(binding, dtype):
    bs, chunk = binding["bs"], binding["chunk"]
    max_batch, max_context = binding["max_batch"], binding["max_context"]
    return {
        "inputs": {
            "req_to_token": {
                "shape": [max_batch, max_context],
                "dtype": "torch.int32",
            },
            "req_pool_indices": {"shape": [bs], "dtype": "torch.int32"},
            "chunk_start_idx": {"shape": [bs], "dtype": "torch.int32"},
            "chunk_seq_lens": {"shape": [bs], "dtype": "torch.int32"},
            "chunk_cu_seq_lens": {"shape": [bs + 1], "dtype": "torch.int32"},
            "chunk_kv_indices": {
                "shape": ["total_kv"],
                "dtype": "torch.int32",
                "note": "zeros(sum(chunk_seq_lens)) int32，原地写回，数据相关",
            },
        },
        "outputs": {
            "chunk_kv_indices": {
                "shape": ["total_kv"],
                "dtype": "torch.int32",
                "note": "长度 sum(chunk_seq_lens)，int32，数据相关",
            },
        },
        "dims": {
            "bs": bs,
            "chunk": chunk,
            "max_batch": max_batch,
            "max_context": max_context,
            "note": "seq_lens 随机，输出长度数据相关",
        },
    }
