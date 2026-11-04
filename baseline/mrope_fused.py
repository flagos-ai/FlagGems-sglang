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

"""mrope_fused baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.rotary_triton.triton_mrope_fused
    调用形态复刻 problems/rope/mrope_fused/baseline.py：
        axis_map = zeros(1, int32)
        triton_mrope_fused(q, k, cos_sin_cache, positions, mrope_section,
            head_size, rotary_dim, mrope_interleaved=False,
            mrope_interleaved_glm=False, is_neox_style=True, axis_map=axis_map)
    native 原地改写 q/k 并返回 None（baseline 先 clone 再调用）；故 IS_INPLACE=True。

输入构造复刻 problems/rope/mrope_fused/cases.py 的 BENCH_CASES：
    q=randn(num_tokens, n_qh*head_size); k=randn(num_tokens, n_kh*head_size)；
    cos_sin_cache=randn(max_pos, rotary_dim)；positions=randint(0,max_pos,(3,num_tokens))。
dtype 取 benchmark/test_mrope_fused.py 的 [bfloat16]。
"""

import torch

OP_NAME = "mrope_fused"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # native 原地改写 q/k

_MAX_POS = 4096
# BENCH_CASES: _case(t, 8, 2, 128, 128, [16, 24, 24]) for t in (1,128,2048,8192)
_SHAPES = [
    {"num_tokens": t, "n_qh": 8, "n_kh": 2, "head_size": 128,
     "rotary_dim": 128, "mrope_section": [16, 24, 24]}
    for t in (1, 128, 2048, 8192)
]


def native():
    """解析 triton_mrope_fused。"""
    try:
        from sglang.kernels.ops.attention.rotary_triton import (
            triton_mrope_fused as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    nt = binding["num_tokens"]
    n_qh, n_kh = binding["n_qh"], binding["n_kh"]
    head_size, rotary_dim = binding["head_size"], binding["rotary_dim"]
    mrope_section = binding["mrope_section"]
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(nt, n_qh * head_size, dtype=torch.float32, device=device, generator=g).to(dtype)
    k = torch.randn(nt, n_kh * head_size, dtype=torch.float32, device=device, generator=g).to(dtype)
    cos_sin_cache = torch.randn(
        _MAX_POS, rotary_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    positions = torch.randint(
        0, _MAX_POS, (3, nt), dtype=torch.int64, device=device, generator=g
    )
    axis_map = torch.zeros(1, dtype=torch.int32, device=device)
    args = (q, k, cos_sin_cache, positions, mrope_section, head_size, rotary_dim)
    kwargs = {
        "mrope_interleaved": False,
        "mrope_interleaved_glm": False,
        "is_neox_style": True,
        "axis_map": axis_map,
    }
    return args, kwargs


def key_shape(binding):
    return "nt{num_tokens}_qh{n_qh}_kh{n_kh}_hs{head_size}_rd{rotary_dim}".format(**binding)


def config(binding, dtype):
    nt = binding["num_tokens"]
    n_qh, n_kh = binding["n_qh"], binding["n_kh"]
    head_size, rotary_dim = binding["head_size"], binding["rotary_dim"]
    mrope_section = binding["mrope_section"]
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [nt, n_qh * head_size], "dtype": dt, "note": "原地写回"},
            "k": {"shape": [nt, n_kh * head_size], "dtype": dt, "note": "原地写回"},
            "cos_sin_cache": {"shape": [_MAX_POS, rotary_dim], "dtype": dt},
            "positions": {"shape": [3, nt], "dtype": "torch.int64"},
            "mrope_section": {"scalar": list(mrope_section)},
            "head_size": {"scalar": head_size},
            "rotary_dim": {"scalar": rotary_dim},
            "mrope_interleaved": {"scalar": False},
            "mrope_interleaved_glm": {"scalar": False},
            "is_neox_style": {"scalar": True},
            "axis_map": {"shape": [1], "dtype": "torch.int32"},
        },
        "outputs": {
            "q": {"shape": [nt, n_qh * head_size], "dtype": dt},
            "k": {"shape": [nt, n_kh * head_size], "dtype": dt},
        },
        "dims": {
            "num_tokens": nt, "n_qh": n_qh, "n_kh": n_kh,
            "head_size": head_size, "rotary_dim": rotary_dim,
            "max_pos": _MAX_POS,
        },
    }
