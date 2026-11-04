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

"""quantize_k_cache_fast_separate baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.dsa.quant_k_cache._quantize_k_cache_fast_separate
    调用形态复刻 baseline.py：
        _quantize_k_cache_fast_separate(k_nope, k_rope, group_size)
    返回 (nope_part_u8, rope_part_u8) 新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES（dim_nope=512, dim_rope=64, group=128）：
    k_nope=(randn(t,512)*2).to(bf16)；k_rope=randn(t,64).to(bf16)。
BENCH: _case(t) for t in (1,128,2048,8192)。
"""

import torch

OP_NAME = "quantize_k_cache_fast_separate"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_DIM_NOPE = 512
_DIM_ROPE = 64
_GROUP = 128
_SHAPES = [{"num_tokens": t, "group_size": _GROUP} for t in (1, 128, 2048, 8192)]


def native():
    """解析 _quantize_k_cache_fast_separate。"""
    try:
        from sglang.kernels.ops.attention.dsa.quant_k_cache import (
            _quantize_k_cache_fast_separate as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    num_tokens, group_size = binding["num_tokens"], binding["group_size"]
    g = torch.Generator(device=device).manual_seed(0)
    k_nope = (torch.randn(num_tokens, _DIM_NOPE, generator=g, device=device) * 2).to(dtype)
    k_rope = torch.randn(num_tokens, _DIM_ROPE, generator=g, device=device).to(dtype)
    return (k_nope, k_rope, group_size), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_g{binding['group_size']}"


def config(binding, dtype):
    num_tokens, group_size = binding["num_tokens"], binding["group_size"]
    num_tiles = _DIM_NOPE // group_size
    dt = str(dtype)
    return {
        "inputs": {
            "k_nope": {"shape": [num_tokens, _DIM_NOPE], "dtype": dt},
            "k_rope": {"shape": [num_tokens, _DIM_ROPE], "dtype": dt},
            "group_size": {"scalar": group_size},
        },
        "outputs": {
            "nope_part_u8": {"shape": [num_tokens, _DIM_NOPE + num_tiles * 4],
                             "dtype": "torch.uint8",
                             "note": "fp8 nope + fp32 scale 的字节视图"},
            "rope_part_u8": {"shape": [num_tokens, _DIM_ROPE * 2],
                             "dtype": "torch.uint8", "note": "bf16 rope 的字节视图"},
        },
        "dims": {"num_tokens": num_tokens, "dim_nope": _DIM_NOPE,
                 "dim_rope": _DIM_ROPE, "group_size": group_size,
                 "num_tiles": num_tiles},
    }
