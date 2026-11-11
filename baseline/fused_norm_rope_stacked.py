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

"""Native SGLang collection workload for fused_norm_rope_stacked."""

import torch

OP_NAME = "fused_norm_rope_stacked"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(t, 8, 8, 128, 128) for t in (1, 128, 2048, 8192)]
MORE_SHAPES = [(1, 2, 2, 64, 64), (37, 3, 4, 128, 128), (129, 2, 2, 128, 64)]
MAX_POS = 4096
EPS = 1e-6


def native():
    """解析 _fused_norm_rope_stacked。"""
    try:
        from sglang.kernels.ops.speculative.fused_kv_materialize import (
            _fused_norm_rope_stacked as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "t": t,
            "n_layers": n_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
        }
        for (t, n_layers, num_kv_heads, head_dim, rotary_dim) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_inputs(binding, dtype, device):
    t, n_layers, num_kv_heads, head_dim, rotary_dim = (
        binding["t"],
        binding["n_layers"],
        binding["num_kv_heads"],
        binding["head_dim"],
        binding["rotary_dim"],
    )
    g = torch.Generator(device=device).manual_seed(0)
    kv_size = num_kv_heads * head_dim
    kv = torch.randn(
        t,
        n_layers,
        2 * kv_size,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    k_norm_weight = torch.randn(
        n_layers, head_dim, generator=g, device=device, dtype=torch.float32
    )
    eps = torch.full((n_layers,), EPS, device=device, dtype=torch.float32)
    cos_sin_cache = torch.randn(
        MAX_POS,
        rotary_dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(dtype)
    positions = torch.randint(
        0, MAX_POS, (t,), generator=g, device=device, dtype=torch.int64
    )
    return (
        kv,
        k_norm_weight,
        eps,
        cos_sin_cache,
        positions,
        num_kv_heads,
        head_dim,
        rotary_dim,
    ), {}


def key_shape(binding):
    return "t{t}_L{n_layers}_kvh{num_kv_heads}_hd{head_dim}_rd{rotary_dim}".format(
        **binding
    )


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
            "k_norm_weight": {
                "shape": [n_layers, head_dim],
                "dtype": "torch.float32",
            },
            "eps": {"shape": [n_layers], "dtype": "torch.float32"},
            "cos_sin_cache": {"shape": [MAX_POS, rotary_dim], "dtype": dt},
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
            "t": t,
            "n_layers": n_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
            "kv_size": kv_size,
            "max_pos": MAX_POS,
        },
    }
