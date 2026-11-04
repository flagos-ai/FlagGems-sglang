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

"""gemma4_fused_routing baseline（方案 B，sglang）。

native：sglang.kernels.ops.layernorm.gemma4_fused_ops.gemma4_fused_routing。
复刻 problems/moe/gemma4_fused_routing/baseline.py：op(gating_output,
per_expert_scale, topk)，返回新的 (weights, ids)，不原地 -> IS_INPLACE=False。

输入复刻 cases.py BENCH_CASES（t∈{1,8,64,512,4096}, e∈{64,128,256}, topk=8,
dtype=fp32）：gating=randn(t,e).fp32; scale=(0.5+rand(e)).fp32。
"""

import torch

OP_NAME = "gemma4_fused_routing"
DTYPES = [torch.float32]
IS_INPLACE = False

_SHAPES = [(t, e, 8) for t in (1, 8, 64, 512, 4096) for e in (64, 128, 256)]


def native():
    try:
        from sglang.kernels.ops.layernorm.gemma4_fused_ops import (
            gemma4_fused_routing as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.moe.gemma4_routing import (
                gemma4_fused_routing as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [{"t": t, "e": e, "topk": tk} for (t, e, tk) in _SHAPES]


def build_inputs(binding, dtype, device):
    t, e, topk = binding["t"], binding["e"], binding["topk"]
    g = torch.Generator(device=device).manual_seed(0)
    gating = torch.randn(t, e, dtype=torch.float32, device=device, generator=g).to(dtype)
    scale = (0.5 + torch.rand(e, dtype=torch.float32, device=device, generator=g)).to(dtype)
    return (gating, scale, topk), {}


def key_shape(binding):
    return [binding["t"], binding["e"], binding["topk"]]


def config(binding, dtype):
    t, e, topk = binding["t"], binding["e"], binding["topk"]
    dt = str(dtype)
    return {
        "inputs": {
            "gating_output": {"shape": [t, e], "dtype": dt},
            "per_expert_scale": {"shape": [e], "dtype": dt},
            "topk": {"scalar": topk},
        },
        "outputs": {
            "weights": {"shape": [t, topk], "dtype": "torch.float32"},
            "ids": {"shape": [t, topk], "dtype": "torch.int32"},
        },
        "dims": {"t": t, "e": e, "topk": topk},
    }
