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

"""fused_eh_norm baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.layernorm.fused_eh_norm.fused_eh_norm
    调用形态复刻 problems/activation_norm/fused_eh_norm/baseline.py：
        fused_eh_norm(inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, eps)
    返回新张量，不原地写回 -> IS_INPLACE=False。

输入构造复刻 problems/activation_norm/fused_eh_norm/cases.py 的 BENCH_CASES：
    inputs_embeds/previous_hidden = randn(bs, hidden).to(dtype)（种子 seed/seed+1）
    enorm_weight/hnorm_weight = randn(hidden).to(dtype)（种子 seed+2/seed+3）
    eps = 1e-6。dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "fused_eh_norm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_SHAPES = [(bs, h) for bs in (1, 8, 64, 512, 4096) for h in (2048, 4096, 7168)]


def native():
    """解析 sglang.kernels.ops.layernorm.fused_eh_norm.fused_eh_norm。"""
    try:
        from sglang.kernels.ops.layernorm.fused_eh_norm import fused_eh_norm as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"bs": bs, "hidden": h} for (bs, h) in _SHAPES]


def _rand(shape, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32, device=device, generator=g).to(dtype)


def build_inputs(binding, dtype, device):
    bs, hidden = binding["bs"], binding["hidden"]
    inputs_embeds = _rand((bs, hidden), dtype, device, 0)
    previous_hidden = _rand((bs, hidden), dtype, device, 1)
    enorm_weight = _rand((hidden,), dtype, device, 2)
    hnorm_weight = _rand((hidden,), dtype, device, 3)
    return (inputs_embeds, previous_hidden, enorm_weight, hnorm_weight, _EPS), {}


def key_shape(binding):
    return [binding["bs"], binding["hidden"]]


def config(binding, dtype):
    bs, hidden = binding["bs"], binding["hidden"]
    dt = str(dtype)
    return {
        "inputs": {
            "inputs_embeds": {"shape": [bs, hidden], "dtype": dt},
            "previous_hidden": {"shape": [bs, hidden], "dtype": dt},
            "enorm_weight": {"shape": [hidden], "dtype": dt},
            "hnorm_weight": {"shape": [hidden], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [bs, hidden], "dtype": dt},
        },
        "dims": {"bs": bs, "hidden": hidden},
    }
