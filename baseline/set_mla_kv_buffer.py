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

"""set_mla_kv_buffer baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.kvcache.mla_buffer.set_mla_kv_buffer_kernel
    是裸 triton kernel，baseline.py 用 kernel[(loc.numel(), 1)](...) 启动并写 out：
        out = kv_buffer.clone()
        block = triton.next_power_of_2(nope_dim + rope_dim)
        set_mla_kv_buffer_kernel[(loc.numel(), 1)](out, cache_k_nope, cache_k_rope,
            loc, out.stride(0), cache_k_nope.stride(0), cache_k_rope.stride(0),
            nope_dim, rope_dim, BLOCK=block, DCP_RANK=0, DCP_WORLD_SIZE=1)
    native() 返回复刻该启动的 wrapper。kernel 写 out -> IS_INPLACE=True。

输入构造复刻 cases.py 的 BENCH_CASES：
    kv_buffer = zeros(num_slots, nope_dim+rope_dim).bf16
    loc = randperm(num_slots)[:n_loc] int64
    cache_k_nope = randn(n_loc, nope_dim).bf16; cache_k_rope = randn(n_loc, rope_dim).bf16
    其中 num_slots=65536, nope_dim=512, rope_dim=64，n_loc in (1,8,64,512,4096,16384)。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "set_mla_kv_buffer"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 写 kv_buffer（out）

_NUM_SLOTS = 65536
_NOPE = 512
_ROPE = 64
_N_LOC = [1, 8, 64, 512, 4096, 16384]


def native():
    """解析裸 kernel 并返回复刻 baseline 启动的 wrapper。"""
    try:
        import triton

        from sglang.kernels.ops.kvcache.mla_buffer import (
            set_mla_kv_buffer_kernel as _kernel,
        )
    except Exception:
        return None

    def _call(kv_buffer, loc, cache_k_nope, cache_k_rope):
        nope_dim = cache_k_nope.shape[-1]
        rope_dim = cache_k_rope.shape[-1]
        block = triton.next_power_of_2(nope_dim + rope_dim)
        _kernel[(loc.numel(), 1)](
            kv_buffer,
            cache_k_nope,
            cache_k_rope,
            loc,
            kv_buffer.stride(0),
            cache_k_nope.stride(0),
            cache_k_rope.stride(0),
            nope_dim,
            rope_dim,
            BLOCK=block,
            DCP_RANK=0,
            DCP_WORLD_SIZE=1,
        )
        return kv_buffer

    return _call


def grid():
    return [
        {"n_loc": n, "num_slots": _NUM_SLOTS, "nope_dim": _NOPE, "rope_dim": _ROPE}
        for n in _N_LOC
    ]


def build_inputs(binding, dtype, device):
    n_loc, num_slots = binding["n_loc"], binding["num_slots"]
    nope_dim, rope_dim = binding["nope_dim"], binding["rope_dim"]
    g = torch.Generator(device=device).manual_seed(9)
    kv_buffer = torch.zeros(num_slots, nope_dim + rope_dim, dtype=dtype, device=device)
    loc = torch.randperm(num_slots, device=device, generator=g)[:n_loc].to(torch.int64)
    nope = torch.randn(n_loc, nope_dim, device=device, generator=g).to(dtype)
    rope = torch.randn(n_loc, rope_dim, device=device, generator=g).to(dtype)
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
            "kv_buffer": {"shape": [num_slots, nope_dim + rope_dim], "dtype": dt,
                          "note": "原地写回"},
            "loc": {"shape": [n_loc], "dtype": "torch.int64"},
            "cache_k_nope": {"shape": [n_loc, nope_dim], "dtype": dt},
            "cache_k_rope": {"shape": [n_loc, rope_dim], "dtype": dt},
        },
        "outputs": {
            "kv_buffer": {"shape": [num_slots, nope_dim + rope_dim], "dtype": dt},
        },
        "dims": {
            "n_loc": n_loc, "num_slots": num_slots,
            "nope_dim": nope_dim, "rope_dim": rope_dim,
        },
    }
