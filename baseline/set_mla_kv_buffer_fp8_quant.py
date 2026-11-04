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

"""set_mla_kv_buffer_fp8_quant baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.kvcache.mla_buffer.set_mla_kv_buffer_triton_fp8_quant
    调用形态复刻 problems/kvcache/set_mla_kv_buffer_fp8_quant/baseline.py：
        kv_buffer = kv_buffer.clone()
        set_mla_kv_buffer_triton_fp8_quant(kv_buffer, loc, cache_k_nope,
            cache_k_rope, fp8_dtype)
    kernel 量化写入 kv_buffer -> IS_INPLACE=True。

输入构造复刻 cases.py 的 BENCH_CASES（_NOPE=512, _ROPE=64）：
    kv_buffer = randn(slots, (nope+rope)//2).bf16  —— 源 dtype 存储、kernel 内按
        fp8 视图，故最后一维元素数减半。
    loc = randperm(slots)[:tokens] int64
    cache_k_nope = randn(tokens, nope).bf16; cache_k_rope = randn(tokens, rope).bf16
    fp8_dtype = torch.float8_e4m3fn
    其中 slots=max(tokens*2, 64)，tokens in (16,128,1024,4096,16384)。
dtype 取 cases.py 的 [bfloat16]（源张量 dtype）。
"""

import torch

OP_NAME = "set_mla_kv_buffer_fp8_quant"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 量化写入 kv_buffer

_NOPE = 512
_ROPE = 64
_TOKENS = [16, 128, 1024, 4096, 16384]


def native():
    """解析 set_mla_kv_buffer_triton_fp8_quant。"""
    try:
        from sglang.kernels.ops.kvcache.mla_buffer import (
            set_mla_kv_buffer_triton_fp8_quant as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"tokens": t, "slots": max(t * 2, 64), "nope_dim": _NOPE, "rope_dim": _ROPE}
        for t in _TOKENS
    ]


def build_inputs(binding, dtype, device):
    tokens, slots = binding["tokens"], binding["slots"]
    nope_dim, rope_dim = binding["nope_dim"], binding["rope_dim"]
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=device, generator=g).to(dtype)

    kv_buffer = _t(slots, (nope_dim + rope_dim) // 2).contiguous()
    loc = torch.randperm(slots, generator=g, device=device)[:tokens].to(torch.int64)
    cache_k_nope = _t(tokens, nope_dim).contiguous()
    cache_k_rope = _t(tokens, rope_dim).contiguous()
    return (kv_buffer, loc, cache_k_nope, cache_k_rope, torch.float8_e4m3fn), {}


def key_shape(binding):
    return (f"t{binding['tokens']}_slots{binding['slots']}"
            f"_nope{binding['nope_dim']}_rope{binding['rope_dim']}")


def config(binding, dtype):
    tokens, slots = binding["tokens"], binding["slots"]
    nope_dim, rope_dim = binding["nope_dim"], binding["rope_dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "kv_buffer": {"shape": [slots, (nope_dim + rope_dim) // 2], "dtype": dt,
                          "note": "原地写回（源 dtype 存储，kernel 内按 fp8 视图，末维半宽）"},
            "loc": {"shape": [tokens], "dtype": "torch.int64"},
            "cache_k_nope": {"shape": [tokens, nope_dim], "dtype": dt},
            "cache_k_rope": {"shape": [tokens, rope_dim], "dtype": dt},
            "fp8_dtype": {"scalar": "torch.float8_e4m3fn"},
        },
        "outputs": {
            "kv_buffer": {"shape": [slots, (nope_dim + rope_dim) // 2], "dtype": dt},
        },
        "dims": {
            "tokens": tokens, "slots": slots,
            "nope_dim": nope_dim, "rope_dim": rope_dim,
        },
    }
