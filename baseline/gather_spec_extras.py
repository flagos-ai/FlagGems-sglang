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

"""gather_spec_extras baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.speculative.gather_spec_extras.gather_spec_extras
    调用形态复刻 problems/speculative/gather_spec_extras/baseline.py：
        gather_spec_extras(indices, topk_p_buf, topk_index_buf,
                           output_tokens_buf, hidden_states_buf)
    返回新张量元组 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    indices = randperm(pool)[:m] int64
    topk_p_buf = rand(pool, topk) fp32
    topk_index_buf = randint(0, 128000, (pool, topk)) int64
    output_tokens_buf = randint(0, 128000, (pool,)) int64
    hidden_states_buf = randn(pool, hidden).bf16
    其中 pool=4096, topk=8, hidden=7168，m in (1,8,64,512,2048)。
dtype 取 hidden_states_buf 的 [bfloat16]。
"""

import torch

OP_NAME = "gather_spec_extras"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_POOL = 4096
_TOPK = 8
_HIDDEN = 7168
_M = [1, 8, 64, 512, 2048]


def native():
    """解析 sglang.kernels.ops.speculative.gather_spec_extras.gather_spec_extras。"""
    try:
        from sglang.kernels.ops.speculative.gather_spec_extras import (
            gather_spec_extras as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "pool": _POOL, "topk": _TOPK, "hidden": _HIDDEN} for m in _M]


def build_inputs(binding, dtype, device):
    m, pool, topk, hidden = (
        binding["m"], binding["pool"], binding["topk"], binding["hidden"],
    )
    g = torch.Generator(device=device).manual_seed(9)
    indices = torch.randperm(pool, device=device, generator=g)[:m].to(torch.int64)
    topk_p_buf = torch.rand(pool, topk, dtype=torch.float32, device=device, generator=g)
    topk_index_buf = torch.randint(
        0, 128000, (pool, topk), dtype=torch.int64, device=device, generator=g
    )
    output_tokens_buf = torch.randint(
        0, 128000, (pool,), dtype=torch.int64, device=device, generator=g
    )
    hidden_states_buf = torch.randn(
        pool, hidden, device=device, generator=g
    ).to(dtype)
    return (indices, topk_p_buf, topk_index_buf, output_tokens_buf,
            hidden_states_buf), {}


def key_shape(binding):
    return (f"m{binding['m']}_pool{binding['pool']}"
            f"_topk{binding['topk']}_h{binding['hidden']}")


def config(binding, dtype):
    m, pool, topk, hidden = (
        binding["m"], binding["pool"], binding["topk"], binding["hidden"],
    )
    dt = str(dtype)
    return {
        "inputs": {
            "indices": {"shape": [m], "dtype": "torch.int64"},
            "topk_p_buf": {"shape": [pool, topk], "dtype": "torch.float32"},
            "topk_index_buf": {"shape": [pool, topk], "dtype": "torch.int64"},
            "output_tokens_buf": {"shape": [pool], "dtype": "torch.int64"},
            "hidden_states_buf": {"shape": [pool, hidden], "dtype": dt},
        },
        "outputs": {
            "topk_p": {"shape": [m, topk], "dtype": "torch.float32"},
            "topk_index": {"shape": [m, topk], "dtype": "torch.int64"},
            "output_tokens": {"shape": [m], "dtype": "torch.int64"},
            "hidden_states": {"shape": [m, hidden], "dtype": dt},
        },
        "dims": {"m": m, "pool": pool, "topk": topk, "hidden": hidden},
    }
