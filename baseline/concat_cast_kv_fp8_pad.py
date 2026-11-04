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

"""concat_cast_kv_fp8_pad baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.dsa.dequant_k_cache.concat_cast_kv_fp8_pad
    调用形态复刻 baseline.py：
        out = out.clone(); return concat_cast_kv_fp8_pad(out, k, k_rope, num_tokens)
    destination-passing kernel：原地写入 out（baseline 先 clone）。IS_INPLACE=True：
    采样前重建实参。

输入构造复刻 cases.py 的 BENCH_CASES（pad=128, nope=512, rope=64, 非 strided）：
    k=randn(num_tokens,512).to(bf16)；k_rope=randn(num_tokens,64).to(bf16)；
    out=full((num_tokens+pad, 576), 7.0).to(fp8_e4m3)。
BENCH: _case(t,128) for t in (16,128,1024,4096,16384)。
"""

import torch

OP_NAME = "concat_cast_kv_fp8_pad"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # destination-passing：原地写入 out

_NOPE = 512
_ROPE = 64
_PAD = 128
_SHAPES = [{"num_tokens": t, "pad": _PAD} for t in (16, 128, 1024, 4096, 16384)]


def native():
    """解析 concat_cast_kv_fp8_pad。"""
    try:
        from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
            concat_cast_kv_fp8_pad as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    num_tokens, pad = binding["num_tokens"], binding["pad"]
    g = torch.Generator(device=device).manual_seed(0)
    k = torch.randn(
        num_tokens, _NOPE, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    k_rope = torch.randn(
        num_tokens, _ROPE, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    out = torch.full(
        (num_tokens + pad, _NOPE + _ROPE), 7.0, dtype=torch.float32, device=device
    ).to(torch.float8_e4m3fn)
    return (out, k, k_rope, num_tokens), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_pad{binding['pad']}"


def config(binding, dtype):
    num_tokens, pad = binding["num_tokens"], binding["pad"]
    dt = str(dtype)
    return {
        "inputs": {
            "out": {"shape": [num_tokens + pad, _NOPE + _ROPE],
                    "dtype": "torch.float8_e4m3fn", "note": "原地写回"},
            "k": {"shape": [num_tokens, _NOPE], "dtype": dt},
            "k_rope": {"shape": [num_tokens, _ROPE], "dtype": dt},
            "num_tokens": {"scalar": num_tokens},
        },
        "outputs": {
            "out": {"shape": [num_tokens + pad, _NOPE + _ROPE],
                    "dtype": "torch.float8_e4m3fn"},
        },
        "dims": {"num_tokens": num_tokens, "pad": pad, "nope": _NOPE, "rope": _ROPE},
    }
