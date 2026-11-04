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

"""fused_norm_rope_stacked baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.speculative.fused_kv_materialize._fused_norm_rope_stacked
    调用形态复刻 problems/speculative/fused_norm_rope_stacked/baseline.py：
        k_out, v_out = _fused_norm_rope_stacked(
            kv, k_norm_weight, eps, cos_sin_cache, positions,
            num_kv_heads, head_dim, rotary_dim)
    返回新张量 (k_out, v_out)，不原地写回输入；故 IS_INPLACE=False。

输入构造复刻 problems/speculative/fused_norm_rope_stacked/cases.py 的 BENCH_CASES：
    _case(t, 8, 8, 128, 128)：kv_size=num_kv_heads*head_dim；
    kv=randn(t, n_layers, 2*kv_size).to(dtype)；k_norm_weight=randn(n_layers, head_dim) f32；
    eps=full((n_layers,), 1e-6) f32；cos_sin_cache=randn(max_pos, rotary_dim).to(dtype)；
    positions=randint(0,max_pos,(t,)) int64。
dtype 取 benchmark/test_fused_norm_rope_stacked.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_norm_rope_stacked"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_MAX_POS = 4096
# BENCH_CASES: _case(t, 8, 8, 128, 128) for t in (1,128,2048,8192)
#   => n_layers=8, num_kv_heads=8, head_dim=128, rotary_dim=128
_SHAPES = [
    {"t": t, "n_layers": 8, "num_kv_heads": 8, "head_dim": 128, "rotary_dim": 128}
    for t in (1, 128, 2048, 8192)
]


def native():
    """解析 _fused_norm_rope_stacked。"""
    try:
        from sglang.kernels.ops.speculative.fused_kv_materialize import (
            _fused_norm_rope_stacked as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    t = binding["t"]
    n_layers = binding["n_layers"]
    num_kv_heads = binding["num_kv_heads"]
    head_dim = binding["head_dim"]
    rotary_dim = binding["rotary_dim"]
    kv_size = num_kv_heads * head_dim
    g = torch.Generator(device=device).manual_seed(0)
    kv = torch.randn(
        t, n_layers, 2 * kv_size, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    k_norm_weight = torch.randn(
        n_layers, head_dim, dtype=torch.float32, device=device, generator=g
    )
    eps = torch.full((n_layers,), 1e-6, dtype=torch.float32, device=device)
    cos_sin_cache = torch.randn(
        _MAX_POS, rotary_dim, dtype=torch.float32, device=device, generator=g
    ).to(dtype)
    positions = torch.randint(
        0, _MAX_POS, (t,), dtype=torch.int64, device=device, generator=g
    )
    args = (
        kv,
        k_norm_weight,
        eps,
        cos_sin_cache,
        positions,
        num_kv_heads,
        head_dim,
        rotary_dim,
    )
    return args, {}


def key_shape(binding):
    return "t{t}_L{n_layers}_kvh{num_kv_heads}_hd{head_dim}_rd{rotary_dim}".format(**binding)


def config(binding, dtype):
    t = binding["t"]
    n_layers = binding["n_layers"]
    num_kv_heads = binding["num_kv_heads"]
    head_dim = binding["head_dim"]
    rotary_dim = binding["rotary_dim"]
    kv_size = num_kv_heads * head_dim
    dt = str(dtype)
    return {
        "inputs": {
            "kv": {"shape": [t, n_layers, 2 * kv_size], "dtype": dt},
            "k_norm_weight": {"shape": [n_layers, head_dim], "dtype": "torch.float32"},
            "eps": {"shape": [n_layers], "dtype": "torch.float32"},
            "cos_sin_cache": {"shape": [_MAX_POS, rotary_dim], "dtype": dt},
            "positions": {"shape": [t], "dtype": "torch.int64"},
            "num_kv_heads": {"scalar": num_kv_heads},
            "head_dim": {"scalar": head_dim},
            "rotary_dim": {"scalar": rotary_dim},
        },
        "outputs": {
            "k_out": {"shape": [t, n_layers, kv_size], "dtype": dt},
            "v_out": {"shape": [t, n_layers, kv_size], "dtype": dt},
        },
        "dims": {
            "t": t, "n_layers": n_layers, "num_kv_heads": num_kv_heads,
            "head_dim": head_dim, "rotary_dim": rotary_dim, "kv_size": kv_size,
            "max_pos": _MAX_POS,
        },
    }
