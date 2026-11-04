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

"""dequantize_k_cache_fast baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.dsa.dequant_k_cache._dequantize_k_cache_fast
    调用形态复刻 baseline.py：
        _dequantize_k_cache_fast(quant_k_cache, group_size)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 _build_quant_k_cache（dim_nope=512, dim_rope=64,
group_size=128）：打包的 quant_k_cache 形如
    (num_tokens, dim_nope + num_tiles*4 + dim_rope*2) fp8_e4m3，
其中 num_tiles=dim_nope//group=4；前 512 列为 fp8 nope，中间 16 列 view 成
fp32 的 4 个 tile scale，末 128 列 view 成 bf16 的 rope。
BENCH: _case(t) for t in (1,128,2048,8192)。
"""

import torch

OP_NAME = "dequantize_k_cache_fast"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_DIM_NOPE = 512
_DIM_ROPE = 64
_GROUP = 128
_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_SHAPES = [{"num_tokens": t, "group_size": _GROUP} for t in (1, 128, 2048, 8192)]


def native():
    """解析 _dequantize_k_cache_fast。"""
    try:
        from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
            _dequantize_k_cache_fast as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def _build_quant_k_cache(num_tokens, group_size, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    k_nope = (torch.randn(num_tokens, _DIM_NOPE, generator=g, device=device) * 2).to(
        torch.bfloat16
    )
    k_rope = torch.randn(num_tokens, _DIM_ROPE, generator=g, device=device).to(
        torch.bfloat16
    )
    num_tiles = _DIM_NOPE // group_size
    buf = torch.empty(
        num_tokens, _DIM_NOPE + num_tiles * 4 + _DIM_ROPE * 2, dtype=_FP8, device=device
    )
    nope_q = buf[:, :_DIM_NOPE]
    nope_s = buf[:, _DIM_NOPE : _DIM_NOPE + num_tiles * 4].view(torch.float32)
    rope = buf[:, _DIM_NOPE + num_tiles * 4 :].view(torch.bfloat16)
    rope[:] = k_rope
    for t in range(num_tiles):
        tile = k_nope[:, t * group_size : (t + 1) * group_size].float()
        scale = (tile.abs().amax(dim=-1) / _FP8_MAX).clamp(min=1e-12)
        nope_s[:, t] = scale
        nope_q[:, t * group_size : (t + 1) * group_size] = (
            tile / scale.unsqueeze(-1)
        ).to(_FP8)
    return buf


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    num_tokens, group_size = binding["num_tokens"], binding["group_size"]
    buf = _build_quant_k_cache(num_tokens, group_size, device)
    return (buf, group_size), {}


def key_shape(binding):
    return f"t{binding['num_tokens']}_g{binding['group_size']}"


def config(binding, dtype):
    num_tokens, group_size = binding["num_tokens"], binding["group_size"]
    num_tiles = _DIM_NOPE // group_size
    packed = _DIM_NOPE + num_tiles * 4 + _DIM_ROPE * 2
    return {
        "inputs": {
            "quant_k_cache": {"shape": [num_tokens, packed],
                              "dtype": "torch.float8_e4m3fn",
                              "note": "打包 fp8 nope + fp32 tile scale + bf16 rope"},
            "group_size": {"scalar": group_size},
        },
        "outputs": {
            "out": {"shape": [num_tokens, _DIM_NOPE + _DIM_ROPE], "dtype": "torch.bfloat16"},
        },
        "dims": {"num_tokens": num_tokens, "dim_nope": _DIM_NOPE,
                 "dim_rope": _DIM_ROPE, "group_size": group_size,
                 "num_tiles": num_tiles},
    }
