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

"""assign_draft_cache_locs_contiguous baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.speculative.cache_locs.assign_draft_cache_locs_contiguous
    是裸 triton kernel，baseline.py 用 kernel[(bs,)](...) 启动并原地写 out：
        out = out_cache_loc.clone()
        assign_draft_cache_locs_contiguous[(req_pool_indices.shape[0],)](
            req_pool_indices, req_to_token, seq_lens, out,
            req_to_token.shape[1], topk, speculative_num_steps)
    采集器只会 op(*args)，故 native() 返回一个按 args 计算 launch grid 的
    wrapper，等价复刻该启动。kernel 原地写 out -> IS_INPLACE=True。

输入构造复刻 problems/speculative/assign_draft_cache_locs_contiguous/cases.py
的 BENCH_CASES：
    req_to_token = randint(0, 1<<20, (max_batch, pool_len)) int32
    pool_idx = randperm(max_batch)[:bs] int32
    seq_lens = randint(0, pool_len-topk*steps, (bs,)) int32
    out_cache_loc = zeros(bs*topk*steps) int32
    其中 topk=8, steps=5, max_batch=256, pool_len=8192，bs in (1,8,64,512,2048)。
全整型，DTYPES 记 int32。
"""

import torch

OP_NAME = "assign_draft_cache_locs_contiguous"
DTYPES = [torch.int32]
IS_INPLACE = True  # kernel 原地写 out_cache_loc

_TOPK = 8
_STEPS = 5
_MAX_BATCH = 256
_POOL_LEN = 8192
_BS = [1, 8, 64, 512, 2048]


def native():
    """解析裸 kernel 并返回按 args 计算 grid 的 wrapper。"""
    try:
        from sglang.kernels.ops.speculative.cache_locs import (
            assign_draft_cache_locs_contiguous as _kernel,
        )
    except Exception:
        return None

    def _call(req_pool_indices, req_to_token, seq_lens, out_cache_loc,
              topk, speculative_num_steps):
        _kernel[(req_pool_indices.shape[0],)](
            req_pool_indices,
            req_to_token,
            seq_lens,
            out_cache_loc,
            req_to_token.shape[1],
            topk,
            speculative_num_steps,
        )
        return out_cache_loc

    return _call


def grid():
    return [
        {"bs": b, "topk": _TOPK, "steps": _STEPS,
         "max_batch": _MAX_BATCH, "pool_len": _POOL_LEN}
        for b in _BS
    ]


def build_inputs(binding, dtype, device):
    bs, topk, steps = binding["bs"], binding["topk"], binding["steps"]
    max_batch, pool_len = binding["max_batch"], binding["pool_len"]
    g = torch.Generator(device=device).manual_seed(9)
    req_to_token = torch.randint(
        0, 1 << 20, (max_batch, pool_len), dtype=torch.int32, device=device, generator=g
    )
    pool_idx = torch.randperm(max_batch, device=device, generator=g)[:bs].to(torch.int32)
    seq_lens = torch.randint(
        0, pool_len - topk * steps, (bs,), dtype=torch.int32, device=device, generator=g
    )
    out_cache_loc = torch.zeros(bs * topk * steps, dtype=torch.int32, device=device)
    return (pool_idx, req_to_token, seq_lens, out_cache_loc, topk, steps), {}


def key_shape(binding):
    return (f"bs{binding['bs']}_topk{binding['topk']}_steps{binding['steps']}"
            f"_mb{binding['max_batch']}_pl{binding['pool_len']}")


def config(binding, dtype):
    bs, topk, steps = binding["bs"], binding["topk"], binding["steps"]
    max_batch, pool_len = binding["max_batch"], binding["pool_len"]
    return {
        "inputs": {
            "req_pool_indices": {"shape": [bs], "dtype": "torch.int32"},
            "req_to_token": {"shape": [max_batch, pool_len], "dtype": "torch.int32"},
            "seq_lens": {"shape": [bs], "dtype": "torch.int32"},
            "out_cache_loc": {"shape": [bs * topk * steps], "dtype": "torch.int32",
                              "note": "原地写回"},
            "topk": {"scalar": topk},
            "speculative_num_steps": {"scalar": steps},
        },
        "outputs": {
            "out_cache_loc": {"shape": [bs * topk * steps], "dtype": "torch.int32"},
        },
        "dims": {
            "bs": bs, "topk": topk, "steps": steps,
            "max_batch": max_batch, "pool_len": pool_len,
        },
    }
