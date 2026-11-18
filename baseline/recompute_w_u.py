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

"""Native SGLang collection workload for recompute_w_u."""

import torch

OP_NAME = "recompute_w_u"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(1, t, 16, 4, 128, 128, 64) for t in (64, 256, 1024)]
MORE_SHAPES = [(1, t, 16, 4, 128, 128, 64) for t in (4096,)]


def native():
    """解析 sglang.kernels.ops.attention.fla.wy_fast.recompute_w_u_fwd。"""
    try:
        from sglang.kernels.ops.attention.fla.wy_fast import (
            recompute_w_u_fwd as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "b": b,
            "t": t,
            "h": h,
            "hg": hg,
            "k_dim": k_dim,
            "v_dim": v_dim,
            "bt": bt,
        }
        for (b, t, h, hg, k_dim, v_dim, bt) in (*SHAPES, *MORE_SHAPES)
    ]


def build_inputs(binding, dtype, device):
    b, t, h, hg, k_dim, v_dim, bt = (
        binding["b"],
        binding["t"],
        binding["h"],
        binding["hg"],
        binding["k_dim"],
        binding["v_dim"],
        binding["bt"],
    )
    gen = torch.Generator(device=device).manual_seed(9)

    def r(*sizes, dtype=dtype):
        return torch.randn(*sizes, device=device, generator=gen).to(dtype)

    return (
        r(b, t, hg, k_dim),
        r(b, t, h, v_dim),
        r(b, t, h, dtype=torch.float32).sigmoid(),
        -r(b, t, h, dtype=torch.float32).abs(),
        r(b, t, h, bt),
        None,
    ), {}


def key_shape(binding):
    return "b%d_t%d_h%d_hg%d_k%d_v%d_bt%d" % (
        binding["b"],
        binding["t"],
        binding["h"],
        binding["hg"],
        binding["k_dim"],
        binding["v_dim"],
        binding["bt"],
    )


def config(binding, dtype):
    b, t, h, hg = binding["b"], binding["t"], binding["h"], binding["hg"]
    k_dim, v_dim, bt = binding["k_dim"], binding["v_dim"], binding["bt"]
    dt = str(dtype)
    return {
        "inputs": {
            "k": {"shape": [b, t, hg, k_dim], "dtype": dt},
            "v": {"shape": [b, t, h, v_dim], "dtype": dt},
            "beta": {"shape": [b, t, h], "dtype": "torch.float32"},
            "g_cumsum": {"shape": [b, t, h], "dtype": "torch.float32"},
            "A": {"shape": [b, t, h, bt], "dtype": dt},
            "cu_seqlens": {"scalar": None},
        },
        "outputs": {
            "w": {"shape": [b, t, h, k_dim], "dtype": dt},
            "u": {"shape": [b, t, h, v_dim], "dtype": dt},
        },
        "dims": {
            "b": b,
            "t": t,
            "h": h,
            "hg": hg,
            "k_dim": k_dim,
            "v_dim": v_dim,
            "bt": bt,
        },
    }
