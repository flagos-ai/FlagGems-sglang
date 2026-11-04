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

"""swiglu_oai_split baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.minimax_m3_swiglu.swiglu_oai_split。复刻
problems/moe/swiglu_oai_split/baseline.py：op(gate_up, alpha, beta, limit,
out_dtype=out_dtype)，返回新张量 -> IS_INPLACE=False。

输入复刻 cases.py BENCH_CASES（m∈{1,8,64,512,4096}, n∈{1024,4096,12288},
alpha=1.702, beta=1.0, limit=None, out_dtype=None, 输入 dtype=bf16）：
    gate_up=(2*randn(m,2n)).bf16。
"""

import torch

OP_NAME = "swiglu_oai_split"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_ALPHA = 1.702
_BETA = 1.0
_SHAPES = [(m, n) for m in (1, 8, 64, 512, 4096) for n in (1024, 4096, 12288)]


def native():
    try:
        from sglang.kernels.ops.moe.minimax_m3_swiglu import swiglu_oai_split as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n_inter": n} for (m, n) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n_inter = binding["m"], binding["n_inter"]
    g = torch.Generator(device=device).manual_seed(0)
    gate_up = (
        2.0 * torch.randn(m, 2 * n_inter, dtype=torch.float32, device=device, generator=g)
    ).to(dtype).contiguous()
    return (gate_up, _ALPHA, _BETA, None), {"out_dtype": None}


def key_shape(binding):
    return [binding["m"], binding["n_inter"]]


def config(binding, dtype):
    m, n_inter = binding["m"], binding["n_inter"]
    dt = str(dtype)
    return {
        "inputs": {
            "gate_up": {"shape": [m, 2 * n_inter], "dtype": dt},
            "alpha": {"scalar": _ALPHA},
            "beta": {"scalar": _BETA},
            "limit": {"scalar": None},
            "out_dtype": {"scalar": None},
        },
        "outputs": {
            "out": {"shape": [m, n_inter], "dtype": dt},
        },
        "dims": {"m": m, "n_inter": n_inter, "gate_up_last": 2 * n_inter},
    }
