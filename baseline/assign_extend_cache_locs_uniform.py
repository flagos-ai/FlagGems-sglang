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

"""assign_extend_cache_locs_uniform baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.speculative.cache_locs.assign_extend_cache_locs_uniform_func
    调用形态复刻 problems/speculative/assign_extend_cache_locs_uniform/baseline.py：
        assign_extend_cache_locs_uniform_func(req_pool_indices=..., req_to_token=...,
            start_offset=..., batch_size=..., draft_token_num=..., device=...)
    返回新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    req_to_token = randint(0, 1<<20, (max_batch, pool_len)) int32
    pool_idx = randperm(max_batch)[:bs] int32
    start_offset = randint(0, pool_len-draft_token_num, (bs,)) int32
    其中 draft_token_num=8, max_batch=4096, pool_len=8192，bs in (1,8,64,512,2048)。
全整型，DTYPES 记 int32。
"""

import torch

OP_NAME = "assign_extend_cache_locs_uniform"
DTYPES = [torch.int32]
IS_INPLACE = False

_DRAFT = 8
_MAX_BATCH = 4096
_POOL_LEN = 8192
_BS = [1, 8, 64, 512, 2048]


def native():
    """解析 assign_extend_cache_locs_uniform_func。"""
    try:
        from sglang.kernels.ops.speculative.cache_locs import (
            assign_extend_cache_locs_uniform_func as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"bs": b, "draft_token_num": _DRAFT,
         "max_batch": _MAX_BATCH, "pool_len": _POOL_LEN}
        for b in _BS
    ]


def build_inputs(binding, dtype, device):
    bs, draft = binding["bs"], binding["draft_token_num"]
    max_batch, pool_len = binding["max_batch"], binding["pool_len"]
    g = torch.Generator(device=device).manual_seed(9)
    req_to_token = torch.randint(
        0, 1 << 20, (max_batch, pool_len), dtype=torch.int32, device=device, generator=g
    )
    pool_idx = torch.randperm(max_batch, device=device, generator=g)[:bs].to(torch.int32)
    start_offset = torch.randint(
        0, pool_len - draft, (bs,), dtype=torch.int32, device=device, generator=g
    )
    return (), {
        "req_pool_indices": pool_idx,
        "req_to_token": req_to_token,
        "start_offset": start_offset,
        "batch_size": bs,
        "draft_token_num": draft,
        "device": torch.device(device),
    }


def key_shape(binding):
    return (f"bs{binding['bs']}_dtn{binding['draft_token_num']}"
            f"_mb{binding['max_batch']}_pl{binding['pool_len']}")


def config(binding, dtype):
    bs, draft = binding["bs"], binding["draft_token_num"]
    max_batch, pool_len = binding["max_batch"], binding["pool_len"]
    return {
        "inputs": {
            "req_pool_indices": {"shape": [bs], "dtype": "torch.int32"},
            "req_to_token": {"shape": [max_batch, pool_len], "dtype": "torch.int32"},
            "start_offset": {"shape": [bs], "dtype": "torch.int32"},
            "batch_size": {"scalar": bs},
            "draft_token_num": {"scalar": draft},
            "device": {"note": "torch.device('cuda')"},
        },
        "outputs": {
            "out_cache_loc": {"shape": [bs * draft], "dtype": "torch.int32",
                              "note": "形状以 kernel 实际为准（bs*draft_token_num）"},
        },
        "dims": {
            "bs": bs, "draft_token_num": draft,
            "max_batch": max_batch, "pool_len": pool_len,
        },
    }
