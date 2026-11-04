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

"""gate_topk baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.gate_topk.gate_topk。复刻 problems/moe/
gate_topk/baseline.py：gate_topk(x, k)，返回新的 (weights, ids)，不原地 ->
IS_INPLACE=False。

输入复刻 cases.py BENCH_CASES（m∈{1,8,64,512,4096}, n=256, k∈{4,8}）：
    x = randperm(m*n).reshape(m,n).float()/(m*n)（distinct 值避免 tie-break）。
dtype：cases 用 fp32（randperm/float），故 DTYPES=[fp32]。
"""

import torch

OP_NAME = "gate_topk"
DTYPES = [torch.float32]
IS_INPLACE = False

_SHAPES = [(m, 256, k) for m in (1, 8, 64, 512, 4096) for k in (4, 8)]


def native():
    try:
        from sglang.kernels.ops.moe.gate_topk import gate_topk as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n, "k": k} for (m, n, k) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n, k = binding["m"], binding["n"], binding["k"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randperm(m * n, device=device, generator=g).reshape(m, n).float() / (m * n)
    return (x.contiguous(), k), {}


def key_shape(binding):
    return [binding["m"], binding["n"], binding["k"]]


def config(binding, dtype):
    m, n, k = binding["m"], binding["n"], binding["k"]
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": "torch.float32"},
            "k": {"scalar": k},
        },
        "outputs": {
            "weights": {"shape": [m, k], "dtype": "torch.float32"},
            "ids": {"shape": [m, k], "dtype": "torch.int32"},
        },
        "dims": {"m": m, "n": n, "k": k},
    }
