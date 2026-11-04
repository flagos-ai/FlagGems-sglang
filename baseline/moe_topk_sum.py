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

"""moe_topk_sum baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.moe_topk_sum.moe_topk_sum（destination-passing）。
复刻 problems/moe/moe_topk_sum/baseline.py：dst=out.clone()，调
moe_topk_sum(x, dst)，原地写 dst 并返回。

输入复刻 cases.py BENCH_CASES（m∈{1,8,64,512,4096}, topk∈{4,8}, k=7168）：
    x=randn(m,topk,k).bf16; out=zeros(m,k).bf16。IS_INPLACE=True（写回 dst）。
"""

import torch

OP_NAME = "moe_topk_sum"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

_SHAPES = [(m, topk, 7168) for m in (1, 8, 64, 512, 4096) for topk in (4, 8)]


def native():
    try:
        from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum as _f
    except Exception:
        return None

    def _op(x, dst):
        _f(x, dst)
        return dst

    return _op


def grid():
    return [{"m": m, "topk": tk, "k": k} for (m, tk, k) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, topk, k = binding["m"], binding["topk"], binding["k"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, topk, k, device=device, generator=g).to(dtype)
    dst = torch.zeros(m, k, dtype=dtype, device=device)
    return (x, dst), {}


def key_shape(binding):
    return [binding["m"], binding["topk"], binding["k"]]


def config(binding, dtype):
    m, topk, k = binding["m"], binding["topk"], binding["k"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, topk, k], "dtype": dt},
            "out": {"shape": [m, k], "dtype": dt, "note": "原地写回（out.clone()）"},
        },
        "outputs": {
            "out": {"shape": [m, k], "dtype": dt},
        },
        "dims": {"m": m, "topk": topk, "k": k},
    }
