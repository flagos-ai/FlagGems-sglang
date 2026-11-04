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

"""get_mla_kv_buffer baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.kvcache.mla_buffer.get_mla_kv_buffer_triton
    调用形态复刻 problems/kvcache/get_mla_kv_buffer/baseline.py：
        nope, rope = cache_k_nope.clone(), cache_k_rope.clone()
        get_mla_kv_buffer_triton(kv_buffer, loc, nope, rope)
    kernel 按 loc 从 kv_buffer 读出写入 nope/rope -> 原地写回后两个实参。
    IS_INPLACE=True（采集侧每次重建 nope/rope）。

输入构造复刻 cases.py 的 BENCH_CASES：
    kv_buffer = randn(num_slots, nope_dim+rope_dim).bf16
    loc = randperm(num_slots)[:n_loc] int64
    cache_k_nope = zeros(n_loc, nope_dim).bf16; cache_k_rope = zeros(n_loc, rope_dim).bf16
    其中 num_slots=65536, nope_dim=512, rope_dim=64，n_loc in (1,8,64,512,4096,16384)。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "get_mla_kv_buffer"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 写 cache_k_nope/cache_k_rope

_NUM_SLOTS = 65536
_NOPE = 512
_ROPE = 64
_N_LOC = [1, 8, 64, 512, 4096, 16384]


def native():
    """解析 get_mla_kv_buffer_triton。"""
    try:
        from sglang.kernels.ops.kvcache.mla_buffer import get_mla_kv_buffer_triton as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"n_loc": n, "num_slots": _NUM_SLOTS, "nope_dim": _NOPE, "rope_dim": _ROPE}
        for n in _N_LOC
    ]


def build_inputs(binding, dtype, device):
    n_loc, num_slots = binding["n_loc"], binding["num_slots"]
    nope_dim, rope_dim = binding["nope_dim"], binding["rope_dim"]
    g = torch.Generator(device=device).manual_seed(9)
    kv_buffer = torch.randn(
        num_slots, nope_dim + rope_dim, device=device, generator=g
    ).to(dtype)
    loc = torch.randperm(num_slots, device=device, generator=g)[:n_loc].to(torch.int64)
    nope = torch.zeros(n_loc, nope_dim, dtype=dtype, device=device)
    rope = torch.zeros(n_loc, rope_dim, dtype=dtype, device=device)
    return (kv_buffer, loc, nope, rope), {}


def key_shape(binding):
    return (f"nloc{binding['n_loc']}_slots{binding['num_slots']}"
            f"_nope{binding['nope_dim']}_rope{binding['rope_dim']}")


def config(binding, dtype):
    n_loc, num_slots = binding["n_loc"], binding["num_slots"]
    nope_dim, rope_dim = binding["nope_dim"], binding["rope_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "kv_buffer": {"shape": [num_slots, nope_dim + rope_dim], "dtype": dt},
            "loc": {"shape": [n_loc], "dtype": "torch.int64"},
            "cache_k_nope": {"shape": [n_loc, nope_dim], "dtype": dt, "note": "原地写回"},
            "cache_k_rope": {"shape": [n_loc, rope_dim], "dtype": dt, "note": "原地写回"},
        },
        "outputs": {
            "cache_k_nope": {"shape": [n_loc, nope_dim], "dtype": dt},
            "cache_k_rope": {"shape": [n_loc, rope_dim], "dtype": dt},
        },
        "dims": {
            "n_loc": n_loc, "num_slots": num_slots,
            "nope_dim": nope_dim, "rope_dim": rope_dim,
        },
    }
