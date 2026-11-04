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

"""softcap_inplace_logits baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.activation.softcap.softcap_inplace_logits
    调用形态复刻 problems/activation_norm/softcap_inplace_logits/baseline.py：
        x = full_logits.clone()
        softcap_inplace_logits(x, final_logit_softcapping)
    算子原地改写第一个张量实参（baseline 先 clone 避免污染）->
    IS_INPLACE=True。采集器对原地算子会在采样前用 build_inputs 重建实参，故此处
    build_inputs 直接构造 contiguous full_logits 传入即可（无需额外 clone）。

输入构造复刻 problems/activation_norm/softcap_inplace_logits/cases.py BENCH_CASES：
    _case(m, n) for m in (1,8,64,512) for n in (4096,32000,128256)
    full_logits = (randn(m, n)*20).to(dtype)，cap=30.0
dtype 取 benchmark/test_softcap_inplace_logits.py 的 [float32]。
"""

import torch

OP_NAME = "softcap_inplace_logits"
DTYPES = [torch.float32]
IS_INPLACE = True  # 原地改写 full_logits

_CAP = 30.0
_SHAPES = [(m, n) for m in (1, 8, 64, 512) for n in (4096, 32000, 128256)]


def native():
    """解析 sglang.kernels.ops.activation.softcap.softcap_inplace_logits。"""
    try:
        from sglang.kernels.ops.activation.softcap import (
            softcap_inplace_logits as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n} for (m, n) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    full_logits = (
        torch.randn(m, n, generator=g, device=device, dtype=torch.float32) * 20
    ).to(dtype)
    return (full_logits, _CAP), {}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "full_logits": {"shape": [m, n], "dtype": dt, "note": "原地写回"},
            "final_logit_softcapping": {"scalar": _CAP},
        },
        "outputs": {
            "full_logits": {"shape": [m, n], "dtype": dt},
        },
        "dims": {"m": m, "n": n},
    }
