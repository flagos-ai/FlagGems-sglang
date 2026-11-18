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

"""Native SGLang collection workload for hc_head."""

import torch

OP_NAME = "hc_head"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_NORM_EPS = 1e-6
_HC_EPS = 1e-3
# BENCH_CASES：hc_mult=4, hidden_size=7168。
SHAPES = [(t, 4, 7168) for t in (1, 128, 2048, 8192)]
MORE_SHAPES = [(1, 2, 128), (37, 4, 256), (129, 2, 512)]


def native():
    """解析 sglang.kernels.ops.layernorm.mhc_head.fused_hc_head。"""
    try:
        from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head as op
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"t": t, "hc_mult": hc_mult, "hidden": hidden}
        for (t, hc_mult, hidden) in dict.fromkeys((*SHAPES, *MORE_SHAPES))
    ]


def build_inputs(binding, dtype, device):
    t, hc_mult, hidden = binding["t"], binding["hc_mult"], binding["hidden"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        t, hc_mult, hidden, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    hc_fn = (
        torch.randn(
            hc_mult,
            hc_mult * hidden,
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        * 0.02
    )
    hc_scale = torch.tensor([1.5], device=device, dtype=torch.float32)
    hc_base = 0.1 * torch.randn(
        hc_mult, generator=g, device=device, dtype=torch.float32
    )
    return (x, hc_fn, hc_scale, hc_base, _NORM_EPS, _HC_EPS), {}


def key_shape(binding):
    return [binding["t"], binding["hc_mult"], binding["hidden"]]


def config(binding, dtype):
    t, hc_mult, hidden = binding["t"], binding["hc_mult"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [t, hc_mult, hidden], "dtype": dt},
            "hc_fn": {
                "shape": [hc_mult, hc_mult * hidden],
                "dtype": "torch.float32",
            },
            "hc_scale": {"shape": [1], "dtype": "torch.float32"},
            "hc_base": {"shape": [hc_mult], "dtype": "torch.float32"},
            "norm_eps": {"scalar": _NORM_EPS},
            "hc_eps": {"scalar": _HC_EPS},
        },
        "outputs": {
            "out": {"shape": [t, hc_mult, hidden], "dtype": dt},
        },
        "dims": {"t": t, "hc_mult": hc_mult, "hidden": hidden},
    }
