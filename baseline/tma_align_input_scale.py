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

"""tma_align_input_scale baseline（方案 B，sglang）。

native：sglang.kernels.ops.moe.ep_moe_kernels.tma_align_input_scale。复刻
problems/moe/tma_align_input_scale/baseline.py：op(input_scale)，返回新张量 ->
IS_INPLACE=False。

输入复刻 cases.py BENCH_CASES（m∈{1,8,64,512,4096,16384}, k_div_block=56）：
    input_scale = rand(m, k_div_block).fp32。
"""

import torch

OP_NAME = "tma_align_input_scale"
DTYPES = [torch.float32]
IS_INPLACE = False

_K_DIV_BLOCK = 56
_MS = [1, 8, 64, 512, 4096, 16384]


def native():
    try:
        from sglang.kernels.ops.moe.ep_moe_kernels import tma_align_input_scale as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "k_div_block": _K_DIV_BLOCK} for m in _MS]


def build_inputs(binding, dtype, device):
    m, k_div_block = binding["m"], binding["k_div_block"]
    g = torch.Generator(device=device).manual_seed(0)
    input_scale = torch.rand(m, k_div_block, dtype=torch.float32, device=device, generator=g)
    return (input_scale,), {}


def key_shape(binding):
    return [binding["m"], binding["k_div_block"]]


def config(binding, dtype):
    m, k_div_block = binding["m"], binding["k_div_block"]
    return {
        "inputs": {
            "input_scale": {"shape": [m, k_div_block], "dtype": "torch.float32"},
        },
        "outputs": {
            "note": "TMA 对齐后的 input_scale（转置/padding 布局）",
        },
        "dims": {"m": m, "k_div_block": k_div_block},
    }
