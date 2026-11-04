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

"""quant_store_kv_mxfp8 baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.mxfp8_quant.quant_store_kv_mxfp8
    调用形态复刻 baseline.py：
        kc,vc,sk,sv = k_cache.clone(),v_cache.clone(),sfk.clone(),sfv.clone()
        quant_store_kv_mxfp8(k, v, loc, kc, vc, sk, sv, page_size=page_size)
        return kc, vc, sk, sv
    kernel 原地写入四个目标缓冲（baseline 先 clone）。IS_INPLACE=True：
    采样前重建实参，避免 warmup 原地累积污染。

输入构造复刻 cases.py 的 BENCH_CASES（page=128, d=128）：
    k/v=randn(t,nkv,d).to(bf16)；loc=randperm 前 t 个 slot int32；
    k_cache/v_cache=zeros(num_pages*128,nkv,d) fp8_e4m3；
    sfk/sfv=zeros(num_pages,nkv,32,128//32,d//32) uint8。
BENCH: _case(t,nkv,128,num_pages=max(2,t//128+1)) for t in (16,128,1024,4096)
       for nkv in (1,2)。
"""

import torch

OP_NAME = "quant_store_kv_mxfp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地写入 k_cache/v_cache/sfk/sfv

_PAGE = 128
_D = 128
_CASES = [
    (t, nkv, _D, max(2, t // _PAGE + 1))
    for t in (16, 128, 1024, 4096)
    for nkv in (1, 2)
]


def native():
    """解析 quant_store_kv_mxfp8。"""
    try:
        from sglang.kernels.ops.quantization.mxfp8_quant import (
            quant_store_kv_mxfp8 as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"t": t, "nkv": nkv, "d": d, "num_pages": np}
        for (t, nkv, d, np) in _CASES
    ]


def build_inputs(binding, dtype, device):
    t, nkv, d, num_pages = (
        binding["t"], binding["nkv"], binding["d"], binding["num_pages"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(t, nkv, d, dtype=torch.float32, device=device, generator=g).to(dtype)
    v = torch.randn(t, nkv, d, dtype=torch.float32, device=device, generator=g).to(dtype)
    slots = torch.randperm(num_pages * _PAGE, generator=g, device=device)[:t]
    loc = slots.to(torch.int32)
    sf = d // 32
    k_cache = torch.zeros(
        (num_pages * _PAGE, nkv, d), dtype=torch.float8_e4m3fn, device=device
    )
    v_cache = torch.zeros_like(k_cache)
    sfk = torch.zeros(
        (num_pages, nkv, 32, _PAGE // 32, sf), dtype=torch.uint8, device=device
    )
    sfv = torch.zeros_like(sfk)
    return (
        k.contiguous(), v.contiguous(), loc, k_cache, v_cache, sfk, sfv,
    ), {"page_size": _PAGE}


def key_shape(binding):
    return f"t{binding['t']}_nkv{binding['nkv']}_d{binding['d']}_np{binding['num_pages']}"


def config(binding, dtype):
    t, nkv, d, num_pages = (
        binding["t"], binding["nkv"], binding["d"], binding["num_pages"],
    )
    sf = d // 32
    dt = str(dtype)
    return {
        "inputs": {
            "k": {"shape": [t, nkv, d], "dtype": dt},
            "v": {"shape": [t, nkv, d], "dtype": dt},
            "loc": {"shape": [t], "dtype": "torch.int32"},
            "k_cache": {"shape": [num_pages * _PAGE, nkv, d],
                        "dtype": "torch.float8_e4m3fn", "note": "原地写回"},
            "v_cache": {"shape": [num_pages * _PAGE, nkv, d],
                        "dtype": "torch.float8_e4m3fn", "note": "原地写回"},
            "sfk": {"shape": [num_pages, nkv, 32, _PAGE // 32, sf],
                    "dtype": "torch.uint8", "note": "原地写回"},
            "sfv": {"shape": [num_pages, nkv, 32, _PAGE // 32, sf],
                    "dtype": "torch.uint8", "note": "原地写回"},
            "page_size": {"scalar": _PAGE},
        },
        "outputs": {
            "k_cache": {"shape": [num_pages * _PAGE, nkv, d],
                        "dtype": "torch.float8_e4m3fn"},
            "v_cache": {"shape": [num_pages * _PAGE, nkv, d],
                        "dtype": "torch.float8_e4m3fn"},
            "sfk": {"shape": [num_pages, nkv, 32, _PAGE // 32, sf],
                    "dtype": "torch.uint8"},
            "sfv": {"shape": [num_pages, nkv, 32, _PAGE // 32, sf],
                    "dtype": "torch.uint8"},
        },
        "dims": {"t": t, "nkv": nkv, "d": d, "num_pages": num_pages,
                 "page_size": _PAGE},
    }
