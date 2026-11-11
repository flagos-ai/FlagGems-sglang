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

"""Native SGLang collection workload for moe_topk_sum."""

import torch

OP_NAME = "moe_topk_sum"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (1,)

SHAPES = [(m, topk, 7168) for m in (1, 8, 64, 512) for topk in (4, 8)]
MORE_SHAPES = [(m, topk, 7168) for m in (4096,) for topk in (4, 8)]


def native():
    try:
        from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum as _f
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None

    def _op(x, dst):
        _f(x, dst)
        return dst

    return _op


def grid():
    return [
        {"m": m, "topk": topk, "k": k}
        for (m, topk, k) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    m, topk, k = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, topk, k, device=device, generator=g).to(cur_dtype)
    out = torch.zeros(m, k, dtype=cur_dtype, device=device)
    return x, out


def build_inputs(binding, dtype, device):
    shape = (binding["m"], binding["topk"], binding["k"])
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    return [binding["m"], binding["topk"], binding["k"]]


def config(binding, dtype):
    m, topk, k = binding["m"], binding["topk"], binding["k"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, topk, k], "dtype": dt},
            "out": {
                "shape": [m, k],
                "dtype": dt,
                "note": "原地写回（out.clone()）",
            },
        },
        "outputs": {
            "out": {"shape": [m, k], "dtype": dt},
        },
        "dims": {"m": m, "topk": topk, "k": k},
    }
